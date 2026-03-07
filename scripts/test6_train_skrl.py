from __future__ import annotations

import argparse
import copy
import gc
import inspect
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from isaaclab.app import AppLauncher

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser("Stable skrl PPO trainer for IsaacLab drone lidar task")
parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-Lidar-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_obstacles", type=int, default=50)
parser.add_argument("--timesteps", type=int, default=2_000_000)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--state_dim", type=int, default=19)
parser.add_argument("--lidar_dim", type=int, default=432)
parser.add_argument("--feat_dim", type=int, default=256)

parser.add_argument("--rollouts", type=int, default=64)
parser.add_argument("--learning_epochs", type=int, default=4)
parser.add_argument("--mini_batches", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--_lambda", type=float, default=0.95)
parser.add_argument("--discount_factor", type=float, default=0.99)

parser.add_argument("--ratio_clip", type=float, default=0.2)
parser.add_argument("--value_clip", type=float, default=0.2)
parser.add_argument("--value_loss_scale", type=float, default=0.5)
parser.add_argument("--grad_norm_clip", type=float, default=0.5)
parser.add_argument("--entropy_coef", type=float, default=0.0)
parser.add_argument("--kl_threshold", type=float, default=0.02)
parser.add_argument("--clip_predicted_values", action="store_true")
parser.add_argument("--no_clip_predicted_values", dest="clip_predicted_values", action="store_false")
parser.set_defaults(clip_predicted_values=True)

parser.add_argument("--reward_scale", type=float, default=0.02)
parser.add_argument("--reward_clip", type=float, default=100.0)

parser.add_argument("--tb_interval", type=int, default=2000)
parser.add_argument("--checkpoint_interval", type=int, default=50000)
parser.add_argument("--cuda_clean_interval", type=int, default=2000)
parser.add_argument("--extra_tb_subdir", type=str, default="extra_tb")

parser.add_argument("--log_cuda_mem", dest="log_cuda_mem", action="store_true")
parser.add_argument("--no_log_cuda_mem", dest="log_cuda_mem", action="store_false")
parser.set_defaults(log_cuda_mem=True)

parser.add_argument(
    "--keep_infos",
    action="store_true",
    default=False,
    help="Pass env infos into skrl memory. Default off to avoid memory growth.",
)
parser.add_argument(
    "--grad_hist_interval",
    type=int,
    default=50,
    help="Log gradient histograms every N PPO updates. 0 disables.",
)
parser.add_argument(
    "--grad_hist_samples",
    type=int,
    default=65536,
    help="Maximum gradient samples per parameter for histogram logging.",
)

parser.add_argument(
    "--debug_act",
    action="store_true",
    default=False,
    help="Print agent.act return structure at step 0.",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Launch Isaac Sim, then import runtime deps
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import omniperception_isaacdrone.tasks.test6_registry as _test6_registry  # noqa: F401

from tqdm import tqdm
from gymnasium.spaces import Box
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from isaaclab_tasks.utils import parse_env_cfg
from torch.utils.tensorboard import SummaryWriter
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from omniperception_isaacdrone.envs.test6_env import ObstacleSpawner
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def format_array_preview(x: np.ndarray, max_items: int = 16) -> str:
    x = np.asarray(x).reshape(-1)
    if x.size <= max_items:
        return np.array2string(x, precision=3, separator=", ")
    head = x[:max_items]
    return f"{np.array2string(head, precision=3, separator=', ')} ... (total={x.size})"


def print_space_bounds(name: str, space: gym.Space) -> None:
    print(f"\n[SPACE] {name}: type={type(space).__name__}", flush=True)

    if isinstance(space, gym.spaces.Dict):
        print(f"[SPACE] {name}.keys={list(space.spaces.keys())}", flush=True)
        for k, subspace in space.spaces.items():
            print_space_bounds(f"{name}.{k}", subspace)
        return

    if isinstance(space, gym.spaces.Box):
        print(f"[SPACE] {name}.shape={space.shape}, dtype={space.dtype}", flush=True)
        print(f"[SPACE] {name}.low preview={format_array_preview(space.low)}", flush=True)
        print(f"[SPACE] {name}.high preview={format_array_preview(space.high)}", flush=True)
        print(f"[SPACE] {name}.low.min={float(np.min(space.low))}, low.max={float(np.max(space.low))}", flush=True)
        print(f"[SPACE] {name}.high.min={float(np.min(space.high))}, high.max={float(np.max(space.high))}", flush=True)
        return

    print(f"[SPACE] {name} = {space}", flush=True)


def init_hidden(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def init_policy_head(m: nn.Linear) -> None:
    nn.init.orthogonal_(m.weight, gain=0.01)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.0)


def init_value_head(m: nn.Linear) -> None:
    nn.init.orthogonal_(m.weight, gain=1.0)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.0)


def sanitize_tb_tag(tag: str) -> str:
    return str(tag).replace(".", "/").replace(" ", "_")


def to_float(x: Any) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, torch.Tensor):
        y = x.detach().float()
        if y.numel() == 0:
            return 0.0
        if not torch.isfinite(y).all():
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return float(y.mean().item())
    try:
        return float(x)
    except Exception:
        return 0.0


def value_sum(x: Any) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, torch.Tensor):
        y = x.detach().float()
        if y.numel() == 0:
            return 0.0
        if not torch.isfinite(y).all():
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return float(y.sum().item())
    try:
        return float(x)
    except Exception:
        return 0.0


def value_mean(x: Any) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, torch.Tensor):
        y = x.detach().float()
        if y.numel() == 0:
            return 0.0
        if not torch.isfinite(y).all():
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return float(y.mean().item())
    try:
        return float(x)
    except Exception:
        return 0.0


def sample_flat(x: torch.Tensor, max_samples: int) -> torch.Tensor:
    y = x.detach().reshape(-1)
    if max_samples <= 0 or y.numel() <= max_samples:
        return y
    idx = torch.randint(0, y.numel(), (max_samples,), device=y.device)
    return y[idx]


def extract_policy_obs(obs: Any) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs
    if isinstance(obs, dict):
        if "policy" in obs and isinstance(obs["policy"], torch.Tensor):
            return obs["policy"]
        for value in obs.values():
            if isinstance(value, torch.Tensor):
                return value
    raise RuntimeError(f"Unsupported observation type: {type(obs)}")


def extract_actions(act_output: Any, act_dim: int) -> torch.Tensor:
    if isinstance(act_output, torch.Tensor):
        return act_output
    if isinstance(act_output, (tuple, list)):
        if len(act_output) == 0:
            raise RuntimeError("agent.act returned empty tuple/list")
        for item in act_output:
            if isinstance(item, torch.Tensor):
                if item.dim() == 2 and item.shape[-1] == act_dim:
                    return item
                if item.dim() == 1 and item.shape[0] == act_dim:
                    return item
        raise RuntimeError(f"Unsupported tuple/list from agent.act: {[type(x) for x in act_output]}")
    if isinstance(act_output, dict):
        for key in ("actions", "action"):
            value = act_output.get(key, None)
            if isinstance(value, torch.Tensor):
                return value
    raise RuntimeError(f"Unsupported agent.act output: {type(act_output)}")


def ensure_obs_shape(x: torch.Tensor, num_envs: int, obs_dim: int) -> torch.Tensor:
    if x.dim() == 2 and x.shape == (num_envs, obs_dim):
        return x
    if x.dim() == 1 and x.numel() == num_envs * obs_dim:
        return x.view(num_envs, obs_dim)
    if x.dim() == 2 and x.shape == (1, num_envs * obs_dim):
        return x.view(num_envs, obs_dim)
    raise RuntimeError(f"Invalid obs shape {tuple(x.shape)}; expected ({num_envs}, {obs_dim})")


def ensure_action_shape(x: torch.Tensor, num_envs: int, act_dim: int) -> torch.Tensor:
    if x.dim() == 1 and x.shape[0] == act_dim:
        x = x.unsqueeze(0).repeat(num_envs, 1)
    elif x.dim() == 2 and x.shape == (1, act_dim) and num_envs > 1:
        x = x.repeat(num_envs, 1)
    if x.dim() != 2 or x.shape != (num_envs, act_dim):
        raise RuntimeError(f"Invalid action shape {tuple(x.shape)}; expected ({num_envs}, {act_dim})")
    return x


def ensure_vec_shape(x: torch.Tensor, num_envs: int, name: str) -> torch.Tensor:
    if x.dim() == 1 and x.shape[0] == num_envs:
        return x.unsqueeze(-1)
    if x.dim() == 2 and x.shape[0] == num_envs:
        return x
    raise RuntimeError(f"Invalid {name} shape {tuple(x.shape)}")


def sanitize_states(states: torch.Tensor, state_dim: int, lidar_dim: int) -> torch.Tensor:
    states = torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    state = torch.clamp(states[:, :state_dim], -1.0, 1.0)
    if lidar_dim <= 0:
        return state
    lidar = torch.clamp(states[:, state_dim: state_dim + lidar_dim], 0.0, 1.0)
    return torch.cat([state, lidar], dim=-1)


def sanitize_actions(actions: torch.Tensor) -> torch.Tensor:
    actions = torch.nan_to_num(actions.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(actions, -1.0, 1.0)


def scale_rewards(rewards: torch.Tensor, scale: float, clip: float) -> torch.Tensor:
    rewards = torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0) * float(scale)
    if clip > 0.0:
        rewards = torch.clamp(rewards, -float(clip), float(clip))
    return rewards


def infer_single_dim_from_box(space: Box, num_envs: int) -> int:
    total = int(np.prod(space.shape))
    if num_envs > 1 and total % num_envs == 0:
        candidate = total // num_envs
        if 0 < candidate != total:
            return candidate
    return total


def get_state_lidar_dims(base_env: Any, obs_dim: int) -> tuple[int, int]:
    state_dim = int(getattr(base_env, "policy_state_dim", 0))
    lidar_dim = int(getattr(base_env, "policy_lidar_dim", 0))
    if state_dim > 0 and state_dim + lidar_dim == obs_dim:
        return state_dim, lidar_dim
    norm_cfg = getattr(getattr(base_env, "cfg", None), "normalization", None)
    state_dim = int(getattr(norm_cfg, "state_dim", args.state_dim))
    if state_dim <= 0 or state_dim > obs_dim:
        raise RuntimeError(f"Invalid state_dim={state_dim} for obs_dim={obs_dim}")
    return state_dim, obs_dim - state_dim


def build_skrl_spaces(base_env: Any, state_dim: int, lidar_dim: int) -> tuple[int, int, gym.spaces.Dict, Box]:
    num_envs = int(getattr(base_env, "num_envs", 1))

    act_space = getattr(base_env, "single_action_space", None)
    if not isinstance(act_space, gym.spaces.Box):
        act_space = getattr(base_env, "action_space", None)

    act_dim = infer_single_dim_from_box(act_space, num_envs) if isinstance(act_space, gym.spaces.Box) else 4

    obs_dim = state_dim + lidar_dim
    obs_low = -np.ones((obs_dim,), dtype=np.float32)
    obs_high = np.ones((obs_dim,), dtype=np.float32)
    if lidar_dim > 0:
        obs_low[state_dim:] = 0.0

    obs_box = Box(low=obs_low, high=obs_high, dtype=np.float32)
    act_box = Box(
        low=-np.ones((act_dim,), dtype=np.float32),
        high=np.ones((act_dim,), dtype=np.float32),
        dtype=np.float32,
    )
    obs_space = gym.spaces.Dict({"policy": obs_box})
    return obs_dim, act_dim, obs_space, act_box


class SkrlSpaceAdapter(gym.Wrapper):
    """Expose skrl-friendly spaces/observations without mutating the base IsaacLab env."""

    def __init__(
        self,
        env: gym.Env,
        obs_space: gym.spaces.Dict,
        act_space: Box,
        state_dim: int,
        lidar_dim: int,
    ):
        super().__init__(env)
        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)
        self.obs_dim = self.state_dim + self.lidar_dim

        # Only patch the wrapper-facing spaces, never the base env itself.
        self.observation_space = obs_space
        self.single_observation_space = obs_space
        self.action_space = act_space
        self.single_action_space = act_space

        # Forward common vector-env attributes used by skrl / training code
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.device = getattr(env, "device", None)

    def _convert_obs(self, raw_obs: Any) -> dict[str, torch.Tensor]:
        x = extract_policy_obs(raw_obs)
        x = ensure_obs_shape(x, self.num_envs, self.obs_dim)
        x = sanitize_states(x, state_dim=self.state_dim, lidar_dim=self.lidar_dim)
        return {"policy": x}

    def reset(self, **kwargs):
        raw_obs, infos = self.env.reset(**kwargs)
        return self._convert_obs(raw_obs), infos

    def step(self, actions):
        raw_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        return self._convert_obs(raw_obs), rewards, terminated, truncated, infos


def models_are_finite(models: dict[str, nn.Module]) -> bool:
    for model in models.values():
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return False
    return True


def snapshot_models(models: dict[str, dict[str, nn.Module]]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for name, model in models.items()
    }


def log_cuda(writer: SummaryWriter, step: int) -> None:
    if args.log_cuda_mem and torch.cuda.is_available():
        writer.add_scalar("CUDA/allocated_mb", torch.cuda.memory_allocated() / 1024**2, step)
        writer.add_scalar("CUDA/reserved_mb", torch.cuda.memory_reserved() / 1024**2, step)
        writer.add_scalar("CUDA/max_allocated_mb", torch.cuda.max_memory_allocated() / 1024**2, step)
        if hasattr(torch.cuda, "max_memory_reserved"):
            writer.add_scalar("CUDA/max_reserved_mb", torch.cuda.max_memory_reserved() / 1024**2, step)


def extract_log_dict(infos: Any) -> Dict[str, Any]:
    if not isinstance(infos, dict):
        return {}
    log = infos.get("log", None)
    if isinstance(log, dict):
        return log
    extras = infos.get("extras", None)
    if isinstance(extras, dict):
        log = extras.get("log", None)
        if isinstance(log, dict):
            return log
    return {}


def extract_tb_reward_terms(base_env: Any) -> Dict[str, torch.Tensor]:
    data = getattr(base_env, "_tb_reward_terms", None)
    if isinstance(data, dict):
        return data
    return {}

def extract_reward_weights(base_env: Any) -> Dict[str, float]:
    """Extract reward term weights from env.cfg.rewards by field name."""
    out: Dict[str, float] = {}

    rewards_cfg = getattr(getattr(base_env, "cfg", None), "rewards", None)
    if rewards_cfg is None:
        return out

    # configclass instances typically expose term cfgs as attributes
    for name in dir(rewards_cfg):
        if name.startswith("_"):
            continue
        try:
            term_cfg = getattr(rewards_cfg, name)
        except Exception:
            continue

        weight = getattr(term_cfg, "weight", None)
        if weight is None:
            continue

        try:
            out[name] = float(weight)
        except Exception:
            pass

    return out



def extract_tb_aux_terms(base_env: Any) -> Dict[str, torch.Tensor]:
    data = getattr(base_env, "_tb_aux_terms", None)
    if isinstance(data, dict):
        return data
    return {}


def clear_tb_caches(base_env: Any) -> None:
    try:
        if isinstance(getattr(base_env, "_tb_reward_terms", None), dict):
            base_env._tb_reward_terms.clear()
    except Exception:
        pass
    try:
        if isinstance(getattr(base_env, "_tb_aux_terms", None), dict):
            base_env._tb_aux_terms.clear()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Reward / Episode accumulators
# -----------------------------------------------------------------------------
class RewardWindowAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.window_steps = 0

        # raw reward term statistics (before reward weight)
        self.term_sum: Dict[str, float] = {}
        self.term_min: Dict[str, float] = {}
        self.term_max: Dict[str, float] = {}

        # weighted reward term statistics (after reward weight)
        self.weighted_sum: Dict[str, float] = {}
        self.weighted_min: Dict[str, float] = {}
        self.weighted_max: Dict[str, float] = {}

        self.aux_sum: Dict[str, float] = {}
        self.aux_min: Dict[str, float] = {}
        self.aux_max: Dict[str, float] = {}

        self.raw_reward_mean_sum = 0.0
        self.raw_reward_min = None
        self.raw_reward_max = None

        self.train_reward_mean_sum = 0.0
        self.train_reward_min = None
        self.train_reward_max = None

    def update(
        self,
        reward_terms: Dict[str, torch.Tensor],
        reward_weights: Dict[str, float],
        aux_terms: Dict[str, torch.Tensor],
        rewards_raw: torch.Tensor,
        rewards_train: torch.Tensor,
    ) -> None:
        self.window_steps += 1

        r_raw = torch.nan_to_num(rewards_raw.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        r_train = torch.nan_to_num(rewards_train.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)

        raw_mean = float(r_raw.mean().item())
        raw_min = float(r_raw.min().item())
        raw_max = float(r_raw.max().item())
        train_mean = float(r_train.mean().item())
        train_min = float(r_train.min().item())
        train_max = float(r_train.max().item())

        self.raw_reward_mean_sum += raw_mean
        self.train_reward_mean_sum += train_mean
        self.raw_reward_min = raw_min if self.raw_reward_min is None else min(self.raw_reward_min, raw_min)
        self.raw_reward_max = raw_max if self.raw_reward_max is None else max(self.raw_reward_max, raw_max)
        self.train_reward_min = train_min if self.train_reward_min is None else min(self.train_reward_min, train_min)
        self.train_reward_max = train_max if self.train_reward_max is None else max(self.train_reward_max, train_max)

        # reward terms: raw + weighted
        for name, value in reward_terms.items():
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                continue

            v = torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)

            raw_mean_v = float(v.mean().item())
            raw_min_v = float(v.min().item())
            raw_max_v = float(v.max().item())

            self.term_sum[name] = self.term_sum.get(name, 0.0) + raw_mean_v
            self.term_min[name] = raw_min_v if name not in self.term_min else min(self.term_min[name], raw_min_v)
            self.term_max[name] = raw_max_v if name not in self.term_max else max(self.term_max[name], raw_max_v)

            w = float(reward_weights.get(name, 1.0))
            vw = v * w

            weighted_mean_v = float(vw.mean().item())
            weighted_min_v = float(vw.min().item())
            weighted_max_v = float(vw.max().item())

            self.weighted_sum[name] = self.weighted_sum.get(name, 0.0) + weighted_mean_v
            self.weighted_min[name] = weighted_min_v if name not in self.weighted_min else min(self.weighted_min[name], weighted_min_v)
            self.weighted_max[name] = weighted_max_v if name not in self.weighted_max else max(self.weighted_max[name], weighted_max_v)

        # aux terms
        for name, value in aux_terms.items():
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                continue
            v = torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            mean_v = float(v.mean().item())
            min_v = float(v.min().item())
            max_v = float(v.max().item())

            self.aux_sum[name] = self.aux_sum.get(name, 0.0) + mean_v
            self.aux_min[name] = min_v if name not in self.aux_min else min(self.aux_min[name], min_v)
            self.aux_max[name] = max_v if name not in self.aux_max else max(self.aux_max[name], max_v)

    def flush(self, writer: SummaryWriter, step: int) -> None:
        if self.window_steps <= 0:
            return

        writer.add_scalar("RewardWindow/raw_mean", self.raw_reward_mean_sum / float(self.window_steps), step)
        writer.add_scalar("RewardWindow/raw_min", float(self.raw_reward_min if self.raw_reward_min is not None else 0.0), step)
        writer.add_scalar("RewardWindow/raw_max", float(self.raw_reward_max if self.raw_reward_max is not None else 0.0), step)

        writer.add_scalar("RewardWindow/train_mean", self.train_reward_mean_sum / float(self.window_steps), step)
        writer.add_scalar("RewardWindow/train_min", float(self.train_reward_min if self.train_reward_min is not None else 0.0), step)
        writer.add_scalar("RewardWindow/train_max", float(self.train_reward_max if self.train_reward_max is not None else 0.0), step)

        # raw per-term reward outputs
        for name in sorted(self.term_sum.keys()):
            tag = sanitize_tb_tag(name)
            writer.add_scalar(f"RewardTermsRaw/{tag}/mean", self.term_sum[name] / float(self.window_steps), step)
            writer.add_scalar(f"RewardTermsRaw/{tag}/min", self.term_min[name], step)
            writer.add_scalar(f"RewardTermsRaw/{tag}/max", self.term_max[name], step)

        # weighted per-term reward outputs
        for name in sorted(self.weighted_sum.keys()):
            tag = sanitize_tb_tag(name)
            writer.add_scalar(f"RewardTermsWeighted/{tag}/mean", self.weighted_sum[name] / float(self.window_steps), step)
            writer.add_scalar(f"RewardTermsWeighted/{tag}/min", self.weighted_min[name], step)
            writer.add_scalar(f"RewardTermsWeighted/{tag}/max", self.weighted_max[name], step)

        for name in sorted(self.aux_sum.keys()):
            tag = sanitize_tb_tag(name)
            writer.add_scalar(f"AuxWindow/{tag}/mean", self.aux_sum[name] / float(self.window_steps), step)
            writer.add_scalar(f"AuxWindow/{tag}/min", self.aux_min[name], step)
            writer.add_scalar(f"AuxWindow/{tag}/max", self.aux_max[name], step)

class EpisodeInfoAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.reward_sum: Dict[str, float] = {}
        self.reward_mean_sum: Dict[str, float] = {}
        self.reward_count: Dict[str, int] = {}

        self.term_sum: Dict[str, float] = {}
        self.term_mean_sum: Dict[str, float] = {}
        self.term_count: Dict[str, int] = {}

        self.other_sum: Dict[str, float] = {}
        self.other_mean_sum: Dict[str, float] = {}
        self.other_count: Dict[str, int] = {}

    def update(self, infos: Any) -> None:
        log_dict = extract_log_dict(infos)
        if len(log_dict) == 0:
            return

        for key, value in log_dict.items():
            if key.startswith("Episode_Reward/"):
                self.reward_sum[key] = self.reward_sum.get(key, 0.0) + value_sum(value)
                self.reward_mean_sum[key] = self.reward_mean_sum.get(key, 0.0) + value_mean(value)
                self.reward_count[key] = self.reward_count.get(key, 0) + 1
            elif key.startswith("Episode_Termination/"):
                self.term_sum[key] = self.term_sum.get(key, 0.0) + value_sum(value)
                self.term_mean_sum[key] = self.term_mean_sum.get(key, 0.0) + value_mean(value)
                self.term_count[key] = self.term_count.get(key, 0) + 1
            else:
                self.other_sum[key] = self.other_sum.get(key, 0.0) + value_sum(value)
                self.other_mean_sum[key] = self.other_mean_sum.get(key, 0.0) + value_mean(value)
                self.other_count[key] = self.other_count.get(key, 0) + 1

    def flush(self, writer: SummaryWriter, step: int) -> None:
        for key in sorted(self.reward_sum.keys()):
            tag = sanitize_tb_tag(key.split("/", 1)[1])
            cnt = max(self.reward_count.get(key, 0), 1)
            writer.add_scalar(f"Episode_Reward/{tag}", self.reward_mean_sum[key] / float(cnt), step)
            writer.add_scalar(f"Episode_Reward/{tag}/sum", self.reward_sum[key], step)
            writer.add_scalar(f"Episode_Reward/{tag}/count", float(self.reward_count[key]), step)

        for key in sorted(self.term_sum.keys()):
            tag = sanitize_tb_tag(key.split("/", 1)[1])
            cnt = max(self.term_count.get(key, 0), 1)
            writer.add_scalar(f"Episode_Termination/{tag}", self.term_sum[key], step)
            writer.add_scalar(f"Episode_Termination/{tag}/mean", self.term_mean_sum[key] / float(cnt), step)
            writer.add_scalar(f"Episode_Termination/{tag}/count", float(self.term_count[key]), step)

        for key in sorted(self.other_sum.keys()):
            tag = sanitize_tb_tag(key)
            cnt = max(self.other_count.get(key, 0), 1)
            writer.add_scalar(f"Episode_Info/{tag}", self.other_mean_sum[key] / float(cnt), step)
            writer.add_scalar(f"Episode_Info/{tag}/sum", self.other_sum[key], step)
            writer.add_scalar(f"Episode_Info/{tag}/count", float(self.other_count[key]), step)


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
def log_reward_action_stats(
    writer: SummaryWriter,
    step: int,
    rewards_raw: torch.Tensor,
    rewards_train: torch.Tensor,
    actions: torch.Tensor,
) -> None:
    if isinstance(rewards_raw, torch.Tensor) and rewards_raw.numel() > 0:
        r = torch.nan_to_num(rewards_raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
        writer.add_scalar("Reward/raw_min", to_float(r.min()), step)
        writer.add_scalar("Reward/raw_mean", to_float(r.mean()), step)
        writer.add_scalar("Reward/raw_max", to_float(r.max()), step)

    if isinstance(rewards_train, torch.Tensor) and rewards_train.numel() > 0:
        r = torch.nan_to_num(rewards_train.float(), nan=0.0, posinf=0.0, neginf=0.0)
        writer.add_scalar("Reward/train_min", to_float(r.min()), step)
        writer.add_scalar("Reward/train_mean", to_float(r.mean()), step)
        writer.add_scalar("Reward/train_max", to_float(r.max()), step)

    if isinstance(actions, torch.Tensor) and actions.numel() > 0:
        a = torch.nan_to_num(actions.float(), nan=0.0, posinf=0.0, neginf=0.0)
        writer.add_scalar("Action/mean", to_float(a.mean()), step)
        writer.add_scalar("Action/std", to_float(a.std(unbiased=False)), step)
        writer.add_scalar("Action/abs_mean", to_float(a.abs().mean()), step)
        if a.dim() == 2:
            for i in range(a.shape[1]):
                writer.add_scalar(f"Action/dim_{i}_mean", to_float(a[:, i].mean()), step)
                writer.add_scalar(f"Action/dim_{i}_std", to_float(a[:, i].std(unbiased=False)), step)


def log_action_processed_stats(writer: SummaryWriter, base_env: Any, step: int) -> None:
    try:
        term = base_env.action_manager.get_term("root_twist")
        a = getattr(term, "processed_actions", None)
        if not isinstance(a, torch.Tensor) or a.numel() == 0:
            return
        a = torch.nan_to_num(a.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        writer.add_scalar("ActionProcessed/mean", to_float(a.mean()), step)
        writer.add_scalar("ActionProcessed/std", to_float(a.std(unbiased=False)), step)
        writer.add_scalar("ActionProcessed/abs_mean", to_float(a.abs().mean()), step)
        if a.dim() == 2:
            for i in range(a.shape[1]):
                writer.add_scalar(f"ActionProcessed/dim_{i}_mean", to_float(a[:, i].mean()), step)
                writer.add_scalar(f"ActionProcessed/dim_{i}_std", to_float(a[:, i].std(unbiased=False)), step)
    except Exception:
        pass


def log_env_step_stats(
    writer: SummaryWriter,
    step: int,
    episode_steps_running: torch.Tensor,
    ended_lengths: List[int],
) -> None:
    if isinstance(episode_steps_running, torch.Tensor) and episode_steps_running.numel() > 0:
        es = torch.nan_to_num(episode_steps_running.float(), nan=0.0, posinf=0.0, neginf=0.0)
        writer.add_scalar("Env/episode_steps_running_min", to_float(es.min()), step)
        writer.add_scalar("Env/episode_steps_running_mean", to_float(es.mean()), step)
        writer.add_scalar("Env/episode_steps_running_max", to_float(es.max()), step)

    if len(ended_lengths) > 0:
        x = torch.tensor(ended_lengths, dtype=torch.float32)
        writer.add_scalar("Env/episode_length_done_min", float(x.min().item()), step)
        writer.add_scalar("Env/episode_length_done_mean", float(x.mean().item()), step)
        writer.add_scalar("Env/episode_length_done_max", float(x.max().item()), step)
        writer.add_scalar("Env/episodes_done_count", float(len(ended_lengths)), step)


def log_policy_stats(writer: SummaryWriter, models: dict[str, nn.Module], step: int) -> None:
    policy = models.get("policy", None)
    value = models.get("value", None)
    if policy is not None and hasattr(policy, "log_std_parameter"):
        try:
            x = policy.log_std_parameter.detach().float()
            writer.add_scalar("Policy/log_std_mean", float(x.mean().item()), step)
            writer.add_scalar("Policy/log_std_min", float(x.min().item()), step)
            writer.add_scalar("Policy/log_std_max", float(x.max().item()), step)
        except Exception:
            pass

    if value is not None:
        try:
            total_norm_sq = 0.0
            count = 0
            for p in value.parameters():
                y = p.detach().float()
                if y.numel() == 0:
                    continue
                total_norm_sq += float((y * y).sum().item())
                count += y.numel()
            if count > 0:
                writer.add_scalar("Value/param_rms", (total_norm_sq / float(count)) ** 0.5, step)
        except Exception:
            pass


def log_gradients(writer: SummaryWriter, models: dict[str, nn.Module], step: int, max_samples: int) -> None:
    for model_key, model in models.items():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue

            g = p.grad.detach()
            if g.numel() == 0:
                continue

            try:
                g_norm = g.norm()
                writer.add_scalar(f"Gradients/{model_key}/norm/{sanitize_tb_tag(name)}", to_float(g_norm), step)
            except Exception:
                pass

            g_s = sample_flat(g, max_samples=max_samples).float()
            finite = torch.isfinite(g_s)
            finite_ratio = float(finite.float().mean().item()) if g_s.numel() > 0 else 0.0
            writer.add_scalar(
                f"Gradients/{model_key}/finite_ratio/{sanitize_tb_tag(name)}",
                finite_ratio,
                step,
            )

            if finite.any():
                g_f = g_s[finite].detach().cpu()
                if g_f.numel() > 0:
                    try:
                        writer.add_histogram(f"Gradients/{model_key}/hist/{sanitize_tb_tag(name)}", g_f, step)
                    except Exception:
                        pass


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class StructuredFeatureExtractor(nn.Module):
    def __init__(self, state_dim: int, lidar_dim: int, feat_dim: int = 256):
        super().__init__()
        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)

        self.state_ln = nn.LayerNorm(self.state_dim)
        self.state_net = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        if self.lidar_dim > 0:
            self.lidar_ln = nn.LayerNorm(self.lidar_dim)
            self.lidar_net = nn.Sequential(
                nn.Linear(self.lidar_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
            )
            fuse_in = 128 + 256
        else:
            self.lidar_ln = nn.Identity()
            self.lidar_net = None
            fuse_in = 128

        self.fuse_net = nn.Sequential(
            nn.Linear(fuse_in, feat_dim),
            nn.Tanh(),
        )
        self.apply(init_hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state = self.state_net(self.state_ln(torch.clamp(obs[:, : self.state_dim], -1.0, 1.0)))
        if self.lidar_dim <= 0:
            return self.fuse_net(state)

        lidar = torch.clamp(obs[:, self.state_dim: self.state_dim + self.lidar_dim], 0.0, 1.0)
        lidar = self.lidar_net(self.lidar_ln(lidar * 2.0 - 1.0))
        return self.fuse_net(torch.cat([state, lidar], dim=-1))


def gaussian_mixin_kwargs() -> dict[str, Any]:
    params = inspect.signature(GaussianMixin.__init__).parameters
    kwargs = {"clip_actions": True}
    if "clip_mean_actions" in params:
        kwargs["clip_mean_actions"] = True
    return kwargs


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, state_dim: int, lidar_dim: int, feat_dim: int = 256):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **gaussian_mixin_kwargs())

        if self.num_observations != state_dim + lidar_dim:
            raise RuntimeError(
                f"[Policy] obs dim mismatch: got {self.num_observations}, expected {state_dim + lidar_dim}"
            )

        self.fe = StructuredFeatureExtractor(state_dim, lidar_dim, feat_dim)
        self.mean = nn.Linear(feat_dim, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -1.0))
        init_policy_head(self.mean)

    def compute(self, inputs, role):
        feat = self.fe(inputs["states"])
        mean = torch.tanh(self.mean(feat))
        log_std = torch.clamp(self.log_std_parameter, min=-5.0, max=0.0).expand_as(mean)
        return mean, log_std, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, state_dim: int, lidar_dim: int, feat_dim: int = 256):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        if self.num_observations != state_dim + lidar_dim:
            raise RuntimeError(
                f"[Value] obs dim mismatch: got {self.num_observations}, expected {state_dim + lidar_dim}"
            )

        self.fe = StructuredFeatureExtractor(state_dim, lidar_dim, feat_dim)
        self.value = nn.Linear(feat_dim, 1)
        init_value_head(self.value)

    def compute(self, inputs, role):
        return self.value(self.fe(inputs["states"])), {}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"[INFO] task={args.task}, num_envs={args.num_envs}, device={args.device}", flush=True)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    try:
        setattr(env_cfg, "seed", int(args.seed))
    except Exception:
        pass

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    print("[INFO] Spawning shared obstacles...", flush=True)
    ObstacleSpawner(num_obstacles=int(args.num_obstacles), seed=int(args.seed)).spawn_obstacles()

    print("[INFO] Creating env...", flush=True)
    base_env = gym.make(args.task, cfg=env_cfg).unwrapped
    base_env.scene.filter_collisions(global_prim_paths=["/World/ground", "/World/Obstacles"])

    space = getattr(base_env, "single_observation_space", None)
    policy_space = (
        space.spaces.get("policy", None)
        if isinstance(space, gym.spaces.Dict)
        else getattr(base_env, "observation_space", None)
    )

    if policy_space is None:
        raise RuntimeError("policy observation space not found")

    obs_dim_raw = int(np.prod(policy_space.shape))
    state_dim, lidar_dim = get_state_lidar_dims(base_env, obs_dim_raw)
    obs_dim, act_dim, obs_space, act_space = build_skrl_spaces(base_env, state_dim, lidar_dim)

    print_space_bounds("skrl_obs_space", obs_space)
    print_space_bounds("skrl_act_space", act_space)


    adapted_env = SkrlSpaceAdapter(
        base_env,
        obs_space=obs_space,
        act_space=act_space,
        state_dim=state_dim,
        lidar_dim=lidar_dim,
    )

    env = wrap_env(adapted_env, wrapper="isaaclab")

    num_envs = int(getattr(env, "num_envs", args.num_envs))
    device = torch.device(getattr(env, "device", args.device))

    print(
        f"[INFO] skrl spaces -> obs={obs_dim} (state={state_dim}, lidar={lidar_dim}), act={act_dim}",
        flush=True,
    )

    models = {
        "policy": Policy(obs_space, act_space, device, state_dim, lidar_dim, args.feat_dim),
        "value": Value(obs_space, act_space, device, state_dim, lidar_dim, args.feat_dim),
    }

    cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
    cfg["rollouts"] = int(args.rollouts)
    cfg["learning_epochs"] = int(args.learning_epochs)
    cfg["mini_batches"] = int(args.mini_batches)
    cfg["discount_factor"] = float(args.discount_factor)
    cfg["lambda"] = float(args._lambda)
    cfg["learning_rate"] = float(args.learning_rate)
    cfg["ratio_clip"] = float(args.ratio_clip)
    cfg["value_clip"] = float(args.value_clip)
    cfg["value_loss_scale"] = float(args.value_loss_scale)
    cfg["entropy_loss_scale"] = float(args.entropy_coef)
    cfg["grad_norm_clip"] = float(args.grad_norm_clip)
    cfg["clip_predicted_values"] = bool(args.clip_predicted_values)
    cfg["kl_threshold"] = float(args.kl_threshold)

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    log_root = project_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f") + "_PPO"
    exp_dir = log_root / run_name
    tb_dir = exp_dir / args.extra_tb_subdir
    tb_dir.mkdir(parents=True, exist_ok=True)

    cfg["experiment"]["directory"] = str(log_root)
    cfg["experiment"]["experiment_name"] = run_name
    cfg["experiment"]["write_interval"] = int(args.tb_interval)
    cfg["experiment"]["checkpoint_interval"] = int(args.checkpoint_interval)

    print(f"[INFO] TensorBoard logdir: {exp_dir}", flush=True)

    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("run/args", str(vars(args)), 0)
    writer.add_text("run/dims", f"obs={obs_dim}, state={state_dim}, lidar={lidar_dim}, act={act_dim}", 0)

    memory = RandomMemory(memory_size=int(args.rollouts), num_envs=num_envs, device=device)
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )
    agent.init()

    # Only reset once: remove the old base_env.reset()
    raw_obs, infos = env.reset()
    states = sanitize_states(
        ensure_obs_shape(extract_policy_obs(raw_obs), num_envs, obs_dim),
        state_dim=state_dim,
        lidar_dim=lidar_dim,
    )

    episode_steps = torch.zeros((num_envs,), device=device, dtype=torch.int32)
    ended_lengths: List[int] = []
    last_log_step = 0
    last_log_time = time.time()
    last_good_snapshot = snapshot_models(models)

    reward_weights = extract_reward_weights(base_env)
    print(f"[INFO] reward weights: {reward_weights}", flush=True)


    reward_window = RewardWindowAccumulator()
    info_window = EpisodeInfoAccumulator()

    print("[INFO] Starting training loop...", flush=True)
    pbar = tqdm(range(int(args.timesteps)), ncols=110)

    try:
        for t in pbar:
            agent.pre_interaction(timestep=t, timesteps=int(args.timesteps))

            with torch.no_grad():
                act_output = agent.act(states, timestep=t, timesteps=int(args.timesteps))

            if args.debug_act and t == 0:
                print(f"[DEBUG] type(agent.act output)={type(act_output)}", flush=True)
                if isinstance(act_output, (tuple, list)):
                    print(f"[DEBUG] len(act_output)={len(act_output)}; elem types={[type(x) for x in act_output]}", flush=True)

            actions = ensure_action_shape(extract_actions(act_output, act_dim), num_envs, act_dim).float()
            if not torch.isfinite(actions).all():
                raise RuntimeError(f"Non-finite actions detected before env.step at t={t}")
            actions = sanitize_actions(actions)

            rollout_boundary = ((t + 1) % int(args.rollouts) == 0)
            if rollout_boundary:
                last_good_snapshot = snapshot_models(models)

            next_obs, rewards, terminated, truncated, infos = env.step(actions)

            next_states = sanitize_states(
                ensure_obs_shape(extract_policy_obs(next_obs), num_envs, obs_dim),
                state_dim=state_dim,
                lidar_dim=lidar_dim,
            )
            rewards = ensure_vec_shape(
                torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0),
                num_envs,
                "rewards",
            )
            terminated = ensure_vec_shape(
                torch.nan_to_num(terminated.float(), nan=0.0, posinf=0.0, neginf=0.0),
                num_envs,
                "terminated",
            ).bool()
            truncated = ensure_vec_shape(
                torch.nan_to_num(truncated.float(), nan=0.0, posinf=0.0, neginf=0.0),
                num_envs,
                "truncated",
            ).bool()
            train_rewards = scale_rewards(rewards, scale=args.reward_scale, clip=args.reward_clip)

            reward_terms = extract_tb_reward_terms(base_env)
            aux_terms = extract_tb_aux_terms(base_env)
            reward_window.update(
                reward_terms=reward_terms,
                reward_weights=reward_weights,
                aux_terms=aux_terms,
                rewards_raw=rewards,
                rewards_train=train_rewards,
            )
            info_window.update(infos)

            record_infos = infos if args.keep_infos else {}

            with torch.no_grad():
                agent.record_transition(
                    states=states,
                    actions=actions,
                    rewards=train_rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=record_infos,
                    timestep=t,
                    timesteps=int(args.timesteps),
                )

            agent.post_interaction(timestep=t, timesteps=int(args.timesteps))

            if rollout_boundary and not models_are_finite(models):
                debug_dir = exp_dir / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                torch.save(last_good_snapshot, debug_dir / f"last_good_before_nan_t{t}.pt")
                raise RuntimeError(
                    f"Non-finite model parameters detected immediately after PPO update at t={t}. "
                    f"Snapshot saved to {debug_dir}."
                )

            done = (terminated | truncated).squeeze(-1)
            episode_steps += 1
            if done.any():
                ended_lengths.extend([int(x) for x in episode_steps[done].detach().cpu().tolist()])
                episode_steps[done] = 0

            if not args.headless:
                try:
                    env.render()
                except Exception:
                    pass

            should_log = int(args.tb_interval) > 0 and (
                ((t + 1) % int(args.tb_interval) == 0) or ((t + 1) == int(args.timesteps))
            )
            if should_log:
                now = time.time()
                fps = float(max((t + 1) - last_log_step, 1)) / max(now - last_log_time, 1e-6)

                writer.add_scalar("Perf/fps", fps, t)
                log_reward_action_stats(writer, t, rewards_raw=rewards, rewards_train=train_rewards, actions=actions)
                log_action_processed_stats(writer, base_env, t)
                log_env_step_stats(writer, t, episode_steps_running=episode_steps, ended_lengths=ended_lengths)
                log_policy_stats(writer, models, t)
                reward_window.flush(writer, t)
                info_window.flush(writer, t)
                log_cuda(writer, t)

                writer.flush()

                ended_lengths.clear()
                reward_window.reset()
                info_window.reset()
                clear_tb_caches(base_env)

                last_log_time = now
                last_log_step = t + 1

            if int(args.grad_hist_interval) > 0 and rollout_boundary:
                update_idx = (t + 1) // int(args.rollouts)
                if update_idx % int(args.grad_hist_interval) == 0:
                    log_gradients(writer, models, t, int(args.grad_hist_samples))
                    writer.flush()

            if int(args.checkpoint_interval) > 0 and ((t + 1) % int(args.checkpoint_interval) == 0):
                ckpt_dir = exp_dir / "manual_checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {name: model.state_dict() for name, model in models.items()},
                    ckpt_dir / f"models_t{t+1}.pt",
                )

            if int(args.cuda_clean_interval) > 0 and ((t + 1) % int(args.cuda_clean_interval) == 0):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.set_description(
                f"t={t} rawR={rewards.mean().item():+.3f} trainR={train_rewards.mean().item():+.3f}"
            )
            states = next_states

    except KeyboardInterrupt:
        print("\n[WARN] KeyboardInterrupt: stopping training early", flush=True)
    finally:
        print("[INFO] Training loop exited", flush=True)
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
        raise
    finally:
        simulation_app.close()
        print("[INFO] Simulation app closed", flush=True)
