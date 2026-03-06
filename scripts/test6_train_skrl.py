from __future__ import annotations

import argparse
import copy
import gc
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# Reduce CUDA allocator fragmentation (must be set before importing torch)
# -----------------------------------------------------------------------------
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)

# -----------------------------------------------------------------------------
# CLI (do not add args already provided by AppLauncher)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser("Test6 skrl PPO training (IsaacLab) - manual loop")

parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-Lidar-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_obstacles", type=int, default=50)

parser.add_argument("--timesteps", type=int, default=2_000_000)
parser.add_argument("--feat_dim", type=int, default=256, help="Final feature dim after state/lidar fusion")

# Extra TensorBoard logging controls
parser.add_argument("--tb_interval", type=int, default=2000, help="Extra TB scalar logging interval (env steps)")
parser.add_argument(
    "--grad_hist_interval",
    type=int,
    default=50,
    help="Extra gradient histogram logging interval (PPO update iterations). 0 disables",
)
parser.add_argument("--grad_hist_samples", type=int, default=65536, help="Max samples per tensor for histogram")
parser.add_argument("--extra_tb_subdir", type=str, default="extra_tb", help="Subdir under experiment logdir")

# PPO hyperparameters
parser.add_argument("--rollouts", type=int, default=64, help="PPO rollouts (steps) before each update")
parser.add_argument("--learning_epochs", type=int, default=4)
parser.add_argument("--mini_batches", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=3e-4)

parser.add_argument("--checkpoint_interval", type=int, default=50000, help="Save checkpoint every N timesteps")

# Memory / leak mitigation toggles
parser.add_argument(
    "--keep_infos",
    action="store_true",
    default=False,
    help=(
        "If set, pass env infos dict into skrl.record_transition. "
        "Default is OFF to avoid memory growth if infos contains large tensors/objects."
    ),
)
parser.add_argument(
    "--cuda_clean_interval",
    type=int,
    default=2000,
    help="Call gc.collect() + torch.cuda.empty_cache() every N env steps. 0 disables.",
)

parser.set_defaults(log_cuda_mem=True)
parser.add_argument("--log_cuda_mem", action="store_true", help="Enable CUDA memory logging to TensorBoard")
parser.add_argument("--no_log_cuda_mem", action="store_false", dest="log_cuda_mem", help="Disable CUDA memory logging")

parser.add_argument(
    "--debug_act",
    action="store_true",
    default=False,
    help="Print agent.act return structure at the first step for debugging",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Launch Isaac Sim
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports AFTER app launch
# -----------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# register tasks
import omniperception_isaacdrone.tasks.test6_registry as _test6_registry  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg
from omniperception_isaacdrone.envs.test6_env import ObstacleSpawner

# skrl
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


# -----------------------------------------------------------------------------
# Initialization helpers
# -----------------------------------------------------------------------------
def _init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# -----------------------------------------------------------------------------
# Structured Feature Extractor with normalized state/lidar assumptions
# -----------------------------------------------------------------------------
class StructuredFeatureExtractor(nn.Module):
    """Two-tower encoder for normalized observations.

    Assumptions:
      - state terms are already normalized into [-1, 1]
      - lidar closeness terms are already normalized into [0, 1]
    """

    def __init__(self, state_dim: int, lidar_dim: int, feat_dim: int = 256):
        super().__init__()
        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)
        self.feat_dim = int(feat_dim)

        self.register_buffer(
            "state_scale",
            torch.ones((1, self.state_dim), dtype=torch.float32),
            persistent=False,
        )

        self.state_ln = nn.LayerNorm(self.state_dim)
        self.lidar_ln = nn.LayerNorm(self.lidar_dim)

        hidden = 256

        self.state_net = nn.Sequential(
            nn.Linear(self.state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.lidar_net = nn.Sequential(
            nn.Linear(self.lidar_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.fuse_net = nn.Sequential(
            nn.Linear(hidden + hidden, self.feat_dim),
            nn.SiLU(),
        )

        self.apply(_init_linear)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        s = obs[:, : self.state_dim]
        l = obs[:, self.state_dim : self.state_dim + self.lidar_dim]

        s = s * self.state_scale.to(s.device)
        s = torch.clamp(s, -1.0, 1.0)
        s = self.state_ln(s)

        l = torch.clamp(l, 0.0, 1.0)
        l = l * 2.0 - 1.0
        l = self.lidar_ln(l)

        s_feat = self.state_net(s)
        l_feat = self.lidar_net(l)
        fused = torch.cat([s_feat, l_feat], dim=-1)
        feat = self.fuse_net(fused)
        return feat


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, state_dim: int, lidar_dim: int, feat_dim: int = 256):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True)

        obs_dim = self.num_observations
        act_dim = self.num_actions

        expected = int(state_dim) + int(lidar_dim)
        if obs_dim != expected:
            raise RuntimeError(
                f"[Policy] Observation dim mismatch: obs_dim={obs_dim} but state_dim+lidar_dim={expected} "
                f"(state_dim={state_dim}, lidar_dim={lidar_dim})."
            )

        self.fe = StructuredFeatureExtractor(state_dim=state_dim, lidar_dim=lidar_dim, feat_dim=feat_dim)
        self.mean = nn.Linear(feat_dim, act_dim)
        self.log_std_parameter = nn.Parameter(torch.full((act_dim,), -0.5))

        self.apply(_init_linear)

    def compute(self, inputs, role):
        obs = inputs["states"]
        feat = self.fe(obs)
        mean = self.mean(feat)
        log_std = torch.clamp(self.log_std_parameter, min=-3.0, max=1.0).expand_as(mean)
        return mean, log_std, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, state_dim: int, lidar_dim: int, feat_dim: int = 256):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        obs_dim = self.num_observations
        expected = int(state_dim) + int(lidar_dim)
        if obs_dim != expected:
            raise RuntimeError(
                f"[Value] Observation dim mismatch: obs_dim={obs_dim} but state_dim+lidar_dim={expected} "
                f"(state_dim={state_dim}, lidar_dim={lidar_dim})."
            )

        self.fe = StructuredFeatureExtractor(state_dim=state_dim, lidar_dim=lidar_dim, feat_dim=feat_dim)
        self.v = nn.Linear(feat_dim, 1)

        self.apply(_init_linear)

    def compute(self, inputs, role):
        obs = inputs["states"]
        feat = self.fe(obs)
        v = self.v(feat)
        return v, {}


# -----------------------------------------------------------------------------
# Space helpers
# -----------------------------------------------------------------------------
def _extract_single_policy_obs_space(base_env) -> Box:
    sp = getattr(base_env, "single_observation_space", None)
    if isinstance(sp, gym.spaces.Dict):
        policy = sp.spaces.get("policy", None)
        if isinstance(policy, gym.spaces.Box):
            return policy

    obs_space = getattr(base_env, "observation_space", None)
    if isinstance(obs_space, gym.spaces.Dict):
        policy = obs_space.spaces.get("policy", None)
        if isinstance(policy, gym.spaces.Box):
            return policy
    if isinstance(obs_space, gym.spaces.Box):
        return obs_space

    raise RuntimeError(f"Cannot extract single policy observation space from env type {type(base_env)}")


def _extract_single_action_space(base_env) -> Box:
    sp = getattr(base_env, "single_action_space", None)
    if isinstance(sp, gym.spaces.Box):
        return sp

    act_space = getattr(base_env, "action_space", None)
    if isinstance(act_space, gym.spaces.Box):
        if len(act_space.shape) == 1:
            return act_space
        raise RuntimeError(
            f"Action space is batched or malformed: shape={act_space.shape}. "
            "Expected single_action_space to be defined on the environment."
        )

    raise RuntimeError(f"Cannot extract single action space from env type {type(base_env)}")


def _assert_finite_box_bounds(space: Box, name: str):
    if not isinstance(space, Box):
        raise RuntimeError(f"{name} is not a gymnasium.spaces.Box: {type(space)}")

    low_finite = np.isfinite(space.low).all()
    high_finite = np.isfinite(space.high).all()
    if not (low_finite and high_finite):
        raise RuntimeError(
            f"{name} is still unbounded: "
            f"low_finite={low_finite}, high_finite={high_finite}, "
            f"shape={space.shape}. "
            "Please verify that MyDroneRLEnv._configure_gym_env_spaces is the version that defines semantic bounds."
        )


def _box_bound_summary(space: Box) -> str:
    return (
        f"shape={space.shape}, "
        f"low[min,max]=({float(np.min(space.low)):.3f}, {float(np.max(space.low)):.3f}), "
        f"high[min,max]=({float(np.min(space.high)):.3f}, {float(np.max(space.high)):.3f})"
    )


def _get_state_lidar_dims(base_env, obs_dim: int) -> tuple[int, int]:
    state_dim = int(getattr(base_env, "policy_state_dim", 0))
    lidar_dim = int(getattr(base_env, "policy_lidar_dim", 0))

    if state_dim <= 0 or state_dim + lidar_dim != obs_dim:
        norm_cfg = getattr(getattr(base_env, "cfg", None), "normalization", None)
        state_dim = int(getattr(norm_cfg, "state_dim", 19))
        if state_dim <= 0 or state_dim > obs_dim:
            raise RuntimeError(
                f"Invalid state_dim inferred from env: state_dim={state_dim}, obs_dim={obs_dim}."
            )
        lidar_dim = obs_dim - state_dim

    return state_dim, lidar_dim


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def _sanitize_tb_tag(tag: str) -> str:
    return tag.replace(".", "/")


def _to_float(x) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.item())
        return float(x.float().mean().item())
    return float(x)


def _sample_flat(t: torch.Tensor, max_samples: int) -> torch.Tensor:
    x = t.detach().view(-1)
    if max_samples <= 0:
        return x
    if x.numel() <= max_samples:
        return x
    idx = torch.randint(low=0, high=x.numel(), size=(max_samples,), device=x.device)
    return x[idx]


def _nan_to_num_inplace(x: torch.Tensor, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> torch.Tensor:
    try:
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    except Exception:
        y = x
        if torch.isnan(y).any():
            y = torch.where(torch.isnan(y), torch.full_like(y, nan), y)
        if torch.isinf(y).any():
            y = torch.where(y == float("inf"), torch.full_like(y, posinf), y)
            y = torch.where(y == float("-inf"), torch.full_like(y, neginf), y)
        return y


def _extract_policy_obs(obs: Any) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs
    if isinstance(obs, dict):
        if "policy" in obs and isinstance(obs["policy"], torch.Tensor):
            return obs["policy"]
        for v in obs.values():
            if isinstance(v, torch.Tensor):
                return v
    raise RuntimeError(f"Unsupported observation type: {type(obs)}")


def _ensure_state_shape(states: torch.Tensor, num_envs: int, obs_dim: int) -> torch.Tensor:
    if not isinstance(states, torch.Tensor):
        raise RuntimeError(f"states is not a torch.Tensor: {type(states)}")

    if states.dim() == 2 and states.shape[0] == num_envs and states.shape[1] == obs_dim:
        return states

    if states.dim() == 1 and states.numel() == num_envs * obs_dim:
        return states.view(num_envs, obs_dim)

    if states.dim() == 2 and states.shape[0] == 1 and states.shape[1] == num_envs * obs_dim:
        return states.view(num_envs, obs_dim)

    raise RuntimeError(
        f"[FATAL] Invalid state shape: got {tuple(states.shape)}, expected ({num_envs}, {obs_dim}) "
        f"or flat {num_envs * obs_dim}"
    )


def log_cuda_memory(writer: SummaryWriter, step: int):
    if not args.log_cuda_mem:
        return
    if not torch.cuda.is_available():
        return
    try:
        writer.add_scalar("CUDA/allocated_mb", float(torch.cuda.memory_allocated() / (1024 ** 2)), step)
        writer.add_scalar("CUDA/reserved_mb", float(torch.cuda.memory_reserved() / (1024 ** 2)), step)
        writer.add_scalar("CUDA/max_allocated_mb", float(torch.cuda.max_memory_allocated() / (1024 ** 2)), step)
        if hasattr(torch.cuda, "max_memory_reserved"):
            writer.add_scalar("CUDA/max_reserved_mb", float(torch.cuda.max_memory_reserved() / (1024 ** 2)), step)
    except Exception:
        pass


def log_reward_terms(writer: SummaryWriter, base_env, step: int):
    rm = getattr(base_env, "reward_manager", None)
    if rm is None:
        return

    step_reward = getattr(rm, "_step_reward", None)
    term_names = getattr(rm, "active_terms", None)

    if step_reward is None or term_names is None:
        return
    if not isinstance(step_reward, torch.Tensor):
        return
    if len(term_names) == 0:
        return

    weights: List[float | None] = []
    try:
        term_cfgs = getattr(rm, "_term_cfgs", None)
        if isinstance(term_cfgs, list) and len(term_cfgs) == len(term_names):
            weights = [float(getattr(cfg, "weight", 0.0)) for cfg in term_cfgs]
        else:
            weights = [None] * len(term_names)
    except Exception:
        weights = [None] * len(term_names)

    for i, name in enumerate(term_names):
        v = step_reward[:, i]

        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            print(f"[WARN] Non-finite reward term '{name}' detected at step={step}", flush=True)
            try:
                writer.add_scalar(f"Debug/nonfinite_reward_term/{_sanitize_tb_tag(name)}", 1.0, step)
            except Exception:
                pass
            v = _nan_to_num_inplace(v, nan=0.0, posinf=0.0, neginf=0.0)

        writer.add_scalar(f"RewardTermsWeighted/{_sanitize_tb_tag(name)}/min", _to_float(v.min()), step)
        writer.add_scalar(f"RewardTermsWeighted/{_sanitize_tb_tag(name)}/mean", _to_float(v.mean()), step)
        writer.add_scalar(f"RewardTermsWeighted/{_sanitize_tb_tag(name)}/max", _to_float(v.max()), step)

        w = weights[i] if i < len(weights) else None
        if w is not None and abs(w) > 1e-12:
            raw = v / float(w)
            if isinstance(raw, torch.Tensor) and not torch.isfinite(raw).all():
                raw = _nan_to_num_inplace(raw, nan=0.0, posinf=0.0, neginf=0.0)
            writer.add_scalar(f"RewardTermsRaw/{_sanitize_tb_tag(name)}/min", _to_float(raw.min()), step)
            writer.add_scalar(f"RewardTermsRaw/{_sanitize_tb_tag(name)}/mean", _to_float(raw.mean()), step)
            writer.add_scalar(f"RewardTermsRaw/{_sanitize_tb_tag(name)}/max", _to_float(raw.max()), step)


def log_termination_ratios(writer: SummaryWriter, base_env, step: int):
    tm = getattr(base_env, "termination_manager", None)
    if tm is None:
        return

    last = getattr(tm, "_last_episode_dones", None)
    term_names = getattr(tm, "active_terms", None)
    if last is None or term_names is None:
        return
    if not isinstance(last, torch.Tensor) or len(term_names) == 0:
        return

    if not torch.isfinite(last).all():
        last = _nan_to_num_inplace(last, nan=0.0, posinf=0.0, neginf=0.0)

    ratios = last.float().mean(dim=0)
    for i, name in enumerate(term_names):
        writer.add_scalar(f"Terminations/ratio_{_sanitize_tb_tag(name)}", _to_float(ratios[i]), step)


def log_env_step_stats(
    writer: SummaryWriter,
    step: int,
    episode_steps_running: torch.Tensor,
    ended_lengths: List[int],
):
    if isinstance(episode_steps_running, torch.Tensor) and episode_steps_running.numel() > 0:
        es = episode_steps_running.float()
        if not torch.isfinite(es).all():
            es = _nan_to_num_inplace(es, nan=0.0, posinf=0.0, neginf=0.0)
        writer.add_scalar("Env/episode_steps_running_min", _to_float(es.min()), step)
        writer.add_scalar("Env/episode_steps_running_mean", _to_float(es.mean()), step)
        writer.add_scalar("Env/episode_steps_running_max", _to_float(es.max()), step)

    if len(ended_lengths) > 0:
        x = torch.tensor(ended_lengths, dtype=torch.float32)
        writer.add_scalar("Env/episode_length_done_min", float(x.min().item()), step)
        writer.add_scalar("Env/episode_length_done_mean", float(x.mean().item()), step)
        writer.add_scalar("Env/episode_length_done_max", float(x.max().item()), step)
        writer.add_scalar("Env/episodes_done_count", float(len(ended_lengths)), step)


def log_reward_action_stats(writer: SummaryWriter, step: int, rewards: torch.Tensor, actions: torch.Tensor):
    if isinstance(rewards, torch.Tensor) and rewards.numel() > 0:
        r = rewards.float()
        if not torch.isfinite(r).all():
            try:
                writer.add_scalar("Debug/nonfinite_rewards", 1.0, step)
            except Exception:
                pass
            r = _nan_to_num_inplace(r, nan=0.0, posinf=0.0, neginf=0.0)
        writer.add_scalar("Reward/total_min", _to_float(r.min()), step)
        writer.add_scalar("Reward/total_mean", _to_float(r.mean()), step)
        writer.add_scalar("Reward/total_max", _to_float(r.max()), step)

    if isinstance(actions, torch.Tensor) and actions.numel() > 0:
        a = actions.float()
        if not torch.isfinite(a).all():
            a = _nan_to_num_inplace(a, nan=0.0, posinf=0.0, neginf=0.0)
        writer.add_scalar("Action/raw_mean", _to_float(a.mean()), step)
        writer.add_scalar("Action/raw_std", _to_float(a.std(unbiased=False)), step)
        writer.add_scalar("Action/raw_abs_mean", _to_float(a.abs().mean()), step)
        if a.dim() == 2:
            for i in range(a.shape[1]):
                writer.add_scalar(f"Action/raw_dim_{i}_mean", _to_float(a[:, i].mean()), step)
                writer.add_scalar(f"Action/raw_dim_{i}_std", _to_float(a[:, i].std(unbiased=False)), step)


def log_action_processed_stats(writer: SummaryWriter, base_env, step: int):
    try:
        term = base_env.action_manager.get_term("root_twist")
        a = getattr(term, "processed_actions", None)
        if not isinstance(a, torch.Tensor):
            return
        a = a.float()
        writer.add_scalar("ActionProcessed/mean", _to_float(a.mean()), step)
        writer.add_scalar("ActionProcessed/std", _to_float(a.std(unbiased=False)), step)
        writer.add_scalar("ActionProcessed/abs_mean", _to_float(a.abs().mean()), step)
        if a.dim() == 2:
            for i in range(a.shape[1]):
                writer.add_scalar(f"ActionProcessed/dim_{i}_mean", _to_float(a[:, i].mean()), step)
                writer.add_scalar(f"ActionProcessed/dim_{i}_std", _to_float(a[:, i].std(unbiased=False)), step)
    except Exception:
        pass


def log_gradients(
    writer: SummaryWriter,
    models: Dict[str, nn.Module],
    step: int,
    max_samples: int,
):
    for key, model in models.items():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue

            g = p.grad.detach()

            try:
                g_norm = g.norm()
                if torch.isfinite(g_norm):
                    writer.add_scalar(f"Gradients/{key}/norm/{_sanitize_tb_tag(name)}", _to_float(g_norm), step)
                else:
                    writer.add_scalar(f"Gradients/{key}/norm/{_sanitize_tb_tag(name)}", float("nan"), step)
            except Exception:
                pass

            g_s = _sample_flat(g, max_samples=max_samples).float()
            finite = torch.isfinite(g_s)
            try:
                writer.add_scalar(
                    f"Gradients/{key}/finite_ratio/{_sanitize_tb_tag(name)}",
                    _to_float(finite.float().mean()) if g_s.numel() > 0 else 0.0,
                    step,
                )
            except Exception:
                pass

            if not finite.any():
                continue

            g_s_f = g_s[finite].detach().cpu()
            if g_s_f.numel() == 0:
                continue

            try:
                writer.add_histogram(f"Gradients/{key}/hist/{_sanitize_tb_tag(name)}", g_s_f, step)
            except ValueError:
                pass
            except Exception:
                pass


def _extract_actions_from_act_output(act_output: Any, act_dim: int) -> torch.Tensor:
    if isinstance(act_output, torch.Tensor):
        return act_output

    if isinstance(act_output, (tuple, list)):
        if len(act_output) == 0:
            raise RuntimeError("agent.act returned an empty tuple/list")
        if isinstance(act_output[0], torch.Tensor):
            return act_output[0]
        for item in act_output:
            if isinstance(item, torch.Tensor):
                if (item.dim() == 2 and item.shape[-1] == act_dim) or (item.dim() == 1 and item.shape[0] == act_dim):
                    return item
        raise RuntimeError(f"agent.act returned tuple/list but no tensor actions found: {[type(x) for x in act_output]}")

    if isinstance(act_output, dict):
        for k in ("actions", "action"):
            v = act_output.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        raise RuntimeError(f"agent.act returned dict but no 'actions' tensor found. keys={list(act_output.keys())}")

    raise RuntimeError(f"Unsupported agent.act return type: {type(act_output)}")


def _ensure_action_shape(actions: torch.Tensor, num_envs: int, act_dim: int) -> torch.Tensor:
    if not isinstance(actions, torch.Tensor):
        raise RuntimeError(f"actions is not a torch.Tensor: {type(actions)}")

    if actions.dim() == 1 and actions.shape[0] == act_dim:
        actions = actions.unsqueeze(0).repeat(num_envs, 1)
    elif actions.dim() == 2 and actions.shape[0] == 1 and actions.shape[1] == act_dim and num_envs > 1:
        actions = actions.repeat(num_envs, 1)

    if actions.dim() != 2 or actions.shape[0] != num_envs or actions.shape[1] != act_dim:
        raise RuntimeError(
            f"[FATAL] Invalid action shape: got {tuple(actions.shape)}, expected ({num_envs}, {act_dim})."
        )

    return actions


def main():
    print(f"[INFO] task={args.task}, num_envs={args.num_envs}, device={args.device}", flush=True)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    print("[INFO] env_cfg parsed", flush=True)

    print("[INFO] Spawning shared obstacles...", flush=True)
    ObstacleSpawner(num_obstacles=int(args.num_obstacles)).spawn_obstacles()

    print("[INFO] Creating env via gym.make(..., cfg=env_cfg)", flush=True)
    base_env = gym.make(args.task, cfg=env_cfg).unwrapped
    print(f"[INFO] Base env type: {type(base_env)}", flush=True)

    try:
        if hasattr(base_env, "scene") and hasattr(base_env.scene, "filter_collisions"):
            base_env.scene.filter_collisions(global_prim_paths=["/World/ground", "/World/Obstacles"])
            print("[INFO] Updated collision filtering to include /World/Obstacles", flush=True)
    except Exception as e:
        print(f"[WARN] scene.filter_collisions failed: {e}", flush=True)

    try:
        base_env.reset()
        print("[INFO] base_env.reset() ok", flush=True)
    except Exception as e:
        print(f"[WARN] base_env.reset() failed before training (will continue): {e}", flush=True)

    # -------------------------------------------------------------------------
    # Use finite semantic spaces defined by MyDroneRLEnv itself
    # -------------------------------------------------------------------------
    flat_obs_space = _extract_single_policy_obs_space(base_env)
    flat_act_space = _extract_single_action_space(base_env)

    _assert_finite_box_bounds(flat_obs_space, "single policy observation space")
    _assert_finite_box_bounds(flat_act_space, "single action space")

    obs_dim = int(np.prod(flat_obs_space.shape))
    act_dim = int(np.prod(flat_act_space.shape))
    state_dim, lidar_dim = _get_state_lidar_dims(base_env, obs_dim=obs_dim)

    print(
        f"[INFO] Finite policy observation space: {_box_bound_summary(flat_obs_space)}",
        flush=True,
    )
    print(
        f"[INFO] Finite action space: {_box_bound_summary(flat_act_space)}",
        flush=True,
    )
    print(
        f"[INFO] Using env-defined dims for skrl: obs_dim={obs_dim}, state_dim={state_dim}, "
        f"lidar_dim={lidar_dim}, act_dim={act_dim}, feat_dim={args.feat_dim}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Wrap env for skrl
    # -------------------------------------------------------------------------
    try:
        from skrl.envs.wrappers.torch import wrap_env
        env = wrap_env(base_env, wrapper="isaaclab")
    except Exception as e:
        print(f"[WARN] wrap_env failed, fallback to base_env directly: {e}", flush=True)
        env = base_env

    print(f"[INFO] Wrapped env type: {type(env)}", flush=True)
    print(f"[INFO] observation_space: {getattr(env, 'observation_space', None)}", flush=True)
    print(f"[INFO] action_space: {getattr(env, 'action_space', None)}", flush=True)
    print(f"[INFO] num_envs: {getattr(env, 'num_envs', None)}, device: {getattr(env, 'device', None)}", flush=True)

    wrapped_obs_space = getattr(env, "observation_space", None)
    wrapped_act_space = getattr(env, "action_space", None)
    if isinstance(wrapped_obs_space, Box):
        _assert_finite_box_bounds(wrapped_obs_space, "wrapped observation space")
    if isinstance(wrapped_act_space, Box):
        _assert_finite_box_bounds(wrapped_act_space, "wrapped action space")

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------
    models = {
        "policy": Policy(
            flat_obs_space,
            flat_act_space,
            getattr(env, "device", args.device),
            state_dim=state_dim,
            lidar_dim=lidar_dim,
            feat_dim=args.feat_dim,
        ),
        "value": Value(
            flat_obs_space,
            flat_act_space,
            getattr(env, "device", args.device),
            state_dim=state_dim,
            lidar_dim=lidar_dim,
            feat_dim=args.feat_dim,
        ),
    }

    # -------------------------------------------------------------------------
    # PPO config
    # -------------------------------------------------------------------------
    agent_cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
    agent_cfg["rollouts"] = int(args.rollouts)
    agent_cfg["learning_epochs"] = int(args.learning_epochs)
    agent_cfg["mini_batches"] = int(args.mini_batches)
    agent_cfg["discount_factor"] = 0.99
    agent_cfg["lambda"] = 0.95
    agent_cfg["learning_rate"] = float(args.learning_rate)
    agent_cfg["grad_norm_clip"] = 1.0
    agent_cfg["ratio_clip"] = 0.2
    agent_cfg["value_clip"] = 0.2
    agent_cfg["entropy_loss_scale"] = 0.02
    agent_cfg["value_loss_scale"] = 0.5

    # -------------------------------------------------------------------------
    # Experiment dir
    # -------------------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    log_root = project_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f") + "_PPO"
    experiment_dir = log_root / run_name

    agent_cfg["experiment"]["directory"] = str(log_root)
    agent_cfg["experiment"]["experiment_name"] = run_name
    agent_cfg["experiment"]["write_interval"] = int(args.tb_interval)
    agent_cfg["experiment"]["checkpoint_interval"] = int(args.checkpoint_interval)

    print(f"[INFO] TensorBoard logdir: {experiment_dir}", flush=True)

    extra_tb_dir = experiment_dir / args.extra_tb_subdir
    extra_tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(extra_tb_dir))

    try:
        writer.add_text("run/args", str(vars(args)), 0)
        writer.add_text(
            "run/obs_action",
            f"obs_dim={obs_dim}, state_dim={state_dim}, lidar_dim={lidar_dim}, act_dim={act_dim}",
            0,
        )
        writer.add_text("run/obs_bounds", _box_bound_summary(flat_obs_space), 0)
        writer.add_text("run/action_bounds", _box_bound_summary(flat_act_space), 0)
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Memory
    # -------------------------------------------------------------------------
    num_envs = int(getattr(env, "num_envs", args.num_envs))
    device = getattr(env, "device", torch.device(args.device))

    memory = RandomMemory(
        memory_size=int(agent_cfg["rollouts"]),
        num_envs=num_envs,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Agent
    # -------------------------------------------------------------------------
    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=flat_obs_space,
        action_space=flat_act_space,
        device=device,
    )
    agent.init()

    # -------------------------------------------------------------------------
    # Manual training loop
    # -------------------------------------------------------------------------
    print("[INFO] Starting training loop (manual, skrl-safe)...", flush=True)

    raw_obs, infos = env.reset()
    states = _ensure_state_shape(_extract_policy_obs(raw_obs), num_envs=num_envs, obs_dim=obs_dim)

    try:
        if hasattr(agent, "reset"):
            agent.reset()
    except Exception:
        pass

    episode_steps = torch.zeros((num_envs,), device=device, dtype=torch.int32)
    ended_lengths: List[int] = []

    rollouts = int(agent_cfg["rollouts"])
    last_log_t = 0
    last_log_time = time.time()

    total_steps = int(args.timesteps)
    pbar = tqdm(range(total_steps), ncols=100)

    try:
        for t in pbar:
            agent.pre_interaction(timestep=t, timesteps=total_steps)

            with torch.no_grad():
                act_output = agent.act(states, timestep=t, timesteps=total_steps)

            if args.debug_act and t == 0:
                print(f"[DEBUG] type(agent.act output)={type(act_output)}", flush=True)
                if isinstance(act_output, (tuple, list)):
                    print(f"[DEBUG] len(act_output)={len(act_output)}; elem types={[type(x) for x in act_output]}", flush=True)

            actions = _extract_actions_from_act_output(act_output, act_dim=act_dim)
            actions = _ensure_action_shape(actions, num_envs=num_envs, act_dim=act_dim)

            next_raw_obs, rewards, terminated, truncated, infos = env.step(actions)
            next_states = _ensure_state_shape(_extract_policy_obs(next_raw_obs), num_envs=num_envs, obs_dim=obs_dim)

            if isinstance(rewards, torch.Tensor) and not torch.isfinite(rewards).all():
                print(f"[WARN] Non-finite rewards detected at t={t}", flush=True)
                try:
                    writer.add_scalar("Debug/nonfinite_rewards_step", 1.0, t)
                except Exception:
                    pass
                rewards = _nan_to_num_inplace(rewards, nan=0.0, posinf=0.0, neginf=0.0)

            if isinstance(next_states, torch.Tensor) and not torch.isfinite(next_states).all():
                print(f"[WARN] Non-finite observations detected at t={t}", flush=True)
                try:
                    writer.add_scalar("Debug/nonfinite_obs_step", 1.0, t)
                except Exception:
                    pass
                next_states = _nan_to_num_inplace(next_states, nan=0.0, posinf=0.0, neginf=0.0)

            if isinstance(terminated, torch.Tensor) and not torch.isfinite(terminated).all():
                terminated = _nan_to_num_inplace(terminated, nan=0.0, posinf=0.0, neginf=0.0).to(torch.bool)

            if isinstance(truncated, torch.Tensor) and not torch.isfinite(truncated).all():
                truncated = _nan_to_num_inplace(truncated, nan=0.0, posinf=0.0, neginf=0.0).to(torch.bool)

            episode_steps += 1
            done = terminated | truncated
            if isinstance(done, torch.Tensor):
                done = done.squeeze(-1)
                if done.any():
                    lens = episode_steps[done].detach().cpu().tolist()
                    ended_lengths.extend([int(x) for x in lens])
                    episode_steps[done] = 0

            record_infos = infos if args.keep_infos else {}

            with torch.no_grad():
                agent.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=record_infos,
                    timestep=t,
                    timesteps=total_steps,
                )

            agent.post_interaction(timestep=t, timesteps=total_steps)

            if not args.headless:
                try:
                    env.render()
                except Exception:
                    pass

            if args.tb_interval > 0 and (t % int(args.tb_interval) == 0):
                now = time.time()
                dt = max(now - last_log_time, 1e-6)
                steps = max(t - last_log_t, 1)
                fps = float(steps) / dt
                writer.add_scalar("Perf/fps", fps, t)
                last_log_t = t
                last_log_time = now

                log_reward_action_stats(writer, t, rewards, actions)
                log_action_processed_stats(writer, base_env, t)
                log_reward_terms(writer, base_env, t)
                log_termination_ratios(writer, base_env, t)
                log_env_step_stats(writer, t, episode_steps, ended_lengths)
                ended_lengths.clear()

                log_cuda_memory(writer, t)
                writer.flush()

            if int(args.grad_hist_interval) > 0 and rollouts > 0:
                if (t + 1) % rollouts == 0:
                    update_idx = (t + 1) // rollouts
                    if update_idx % int(args.grad_hist_interval) == 0:
                        log_gradients(writer, models=models, step=t, max_samples=int(args.grad_hist_samples))
                        writer.flush()

            if int(args.cuda_clean_interval) > 0 and (t % int(args.cuda_clean_interval) == 0):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            try:
                if isinstance(rewards, torch.Tensor):
                    pbar.set_description(f"t={t} R(mean)={rewards.float().mean().item():.3f}")
            except Exception:
                pass

            states = next_states

    except KeyboardInterrupt:
        print("\n[WARN] KeyboardInterrupt: stopping training loop early", flush=True)

    print("[INFO] Training finished", flush=True)

    try:
        writer.close()
    except Exception:
        pass

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Unhandled exception:\n", flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
        print("[INFO] Simulation app closed", flush=True)
