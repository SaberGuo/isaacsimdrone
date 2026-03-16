from __future__ import annotations

import argparse
import copy
import gc
import inspect
import os
import traceback
from collections import deque
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
parser.add_argument(
    "--num_obstacles",
    type=int,
    default=100,
    help="Maximum number of shared obstacles to spawn. The curriculum only activates a subset.",
)
parser.add_argument("--timesteps", type=int, default=2_000_000)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--state_dim", type=int, default=16)
parser.add_argument("--lidar_dim", type=int, default=432)
parser.add_argument("--feat_dim", type=int, default=256)

parser.add_argument("--rollouts", type=int, default=256)
parser.add_argument("--learning_epochs", type=int, default=4)
parser.add_argument("--mini_batches", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--_lambda", type=float, default=0.95)
parser.add_argument("--discount_factor", type=float, default=0.999)

parser.add_argument("--ratio_clip", type=float, default=0.15)
parser.add_argument("--value_clip", type=float, default=0.15)
parser.add_argument("--value_loss_scale", type=float, default=0.5)
parser.add_argument("--grad_norm_clip", type=float, default=0.5)
parser.add_argument("--entropy_coef", type=float, default=5e-3)
parser.add_argument("--kl_threshold", type=float, default=0.01)
parser.add_argument("--clip_predicted_values", action="store_true")
parser.add_argument("--no_clip_predicted_values", dest="clip_predicted_values", action="store_false")
parser.set_defaults(clip_predicted_values=True)

parser.add_argument("--reward_scale", type=float, default=0.1)
parser.add_argument("--reward_clip", type=float, default=500.0)

parser.add_argument("--tb_interval", type=int, default=500)
parser.add_argument("--dist_interval", type=int, default=500)
parser.add_argument("--dist_window", type=int, default=10)
parser.add_argument("--dist_max_samples", type=int, default=2048)
parser.add_argument("--checkpoint_interval", type=int, default=50000)
parser.add_argument("--cuda_clean_interval", type=int, default=2000)
parser.add_argument("--extra_tb_subdir", type=str, default="extra_tb")

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

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import omniperception_isaacdrone.tasks.test6_registry as _test6_registry  # noqa: F401
from gymnasium.spaces import Box
from isaaclab_tasks.utils import parse_env_cfg
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from omniperception_isaacdrone.envs.test6_env import ObstacleSpawner, WallSpawner


STATE_OBS_NAMES_16 = [
    "root_quat_w",
    "root_quat_x",
    "root_quat_y",
    "root_quat_z",
    "root_lin_vel_x",
    "root_lin_vel_y",
    "root_lin_vel_z",
    "root_ang_vel_x",
    "root_ang_vel_y",
    "root_ang_vel_z",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    "goal_delta_x",
    "goal_delta_y",
    "goal_delta_z",
]

ACTION_NAMES_4 = ["vx_cmd", "vy_cmd", "vz_cmd", "yaw_rate_cmd"]

DEBUG_PRINT = False

from pxr import UsdGeom, Gf
import isaacsim.core.utils.prims as prim_utils


def get_cfg_obstacle_curriculum_levels(env_cfg: Any) -> tuple[int, ...]:
    curriculum_cfg = getattr(env_cfg, "obstacle_curriculum", None)
    if curriculum_cfg is None:
        return ()
    try:
        return tuple(int(v) for v in getattr(curriculum_cfg, "levels", ()))
    except Exception:
        return ()


def scale_robot_visual_only(num_envs: int, visual_scale=(20.0, 20.0, 10.0)) -> None:
    stage = prim_utils.get_prim_at_path("/World").GetStage()
    sx, sy, sz = map(float, visual_scale)

    for i in range(int(num_envs)):
        visual_path = f"/World/envs/env_{i}/Robot/body/body_visual"
        prim = stage.GetPrimAtPath(visual_path)

        if not prim.IsValid():
            print(f"[WARN] visual prim not found: {visual_path}", flush=True)
            continue

        xform = UsdGeom.Xformable(prim)
        scale_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]

        if len(scale_ops) > 0:
            scale_ops[0].Set(Gf.Vec3f(sx, sy, sz))
        else:
            xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))

        print(f"[INFO] visual-only scale set on {visual_path}: {(sx, sy, sz)}", flush=True)


def debug_print(msg: str) -> None:
    if DEBUG_PRINT:
        print(msg, flush=True)


def tensor_stats_str(name: str, x: torch.Tensor, max_items: int = 8) -> str:
    if not isinstance(x, torch.Tensor):
        return f"{name}: <non-tensor {type(x)}>"
    y = x.detach()
    if y.numel() == 0:
        return f"{name}: shape={tuple(y.shape)} EMPTY"
    y = torch.nan_to_num(y.float(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = y.reshape(-1)
    preview = flat[:max_items].cpu().numpy()
    return (
        f"{name}: shape={tuple(y.shape)}, dtype={y.dtype}, "
        f"min={float(y.min().item()):+.4f}, max={float(y.max().item()):+.4f}, "
        f"mean={float(y.mean().item()):+.4f}, std={float(y.std().item()):+.4f}, "
        f"preview={np.array2string(preview, precision=3, separator=', ')}"
    )


def print_env0_transition(
    step: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    next_states: torch.Tensor,
    state_dim: int,
    lidar_dim: int,
) -> None:
    env_id = 0
    s0 = states[env_id]
    ns0 = next_states[env_id]
    a0 = actions[env_id]
    r0 = rewards[env_id]

    debug_print(f"\n[STEP {step}] ENV0 transition")
    debug_print(f"  state[:{state_dim}] = {s0[:state_dim].detach().cpu().numpy()}")
    if lidar_dim > 0:
        lidar0 = s0[state_dim:state_dim + lidar_dim]
        debug_print(
            "  lidar stats: "
            f"min={float(lidar0.min().item()):.4f}, "
            f"max={float(lidar0.max().item()):.4f}, "
            f"mean={float(lidar0.mean().item()):.4f}, "
            f"nonzero_ratio={float((lidar0 > 1e-6).float().mean().item()):.4f}"
        )
    debug_print(f"  action = {a0.detach().cpu().numpy()}")
    debug_print(f"  reward = {r0.detach().cpu().numpy()}")
    debug_print(
        f"  terminated={terminated[env_id].item()}, truncated={truncated[env_id].item()}"
    )
    debug_print(f"  next_state[:{state_dim}] = {ns0[:state_dim].detach().cpu().numpy()}")


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
    return int(np.prod(space.shape))


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

        self.observation_space = obs_space
        self.single_observation_space = obs_space
        self.action_space = act_space
        self.single_action_space = act_space

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


def snapshot_models(models: dict[str, nn.Module]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for name, model in models.items()
    }


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


def extract_termination_count_dict(infos: Any) -> Dict[str, float]:
    log_dict = extract_log_dict(infos)
    if len(log_dict) == 0:
        return {}

    out: Dict[str, float] = {}
    for key, value in log_dict.items():
        if not key.startswith("Episode_Termination/"):
            continue
        name = key.split("/", 1)[1]
        out[name] = to_float(value)
    return out


def extract_termination_ratio_dict(infos: Any, done_count: int) -> Dict[str, float]:
    if int(done_count) <= 0:
        return {}

    count_dict = extract_termination_count_dict(infos)
    if len(count_dict) == 0:
        return {}

    denom = max(float(done_count), 1.0)
    return {name: float(value) / denom for name, value in count_dict.items()}


def extract_prefixed_log_scalars(infos: Any, prefix: str) -> Dict[str, float]:
    log_dict = extract_log_dict(infos)
    if len(log_dict) == 0:
        return {}

    out: Dict[str, float] = {}
    for key, value in log_dict.items():
        if key.startswith(prefix):
            out[key] = to_float(value)
    return out


def get_env_step_dt(base_env: Any) -> float:
    if hasattr(base_env, "step_dt"):
        try:
            return float(base_env.step_dt)
        except Exception:
            pass
    try:
        return float(base_env.cfg.sim.dt) * float(base_env.cfg.decimation)
    except Exception:
        return 1.0 / 60.0


def extract_reward_weights(base_env: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}

    reward_manager = getattr(base_env, "reward_manager", None)
    if reward_manager is not None:
        try:
            for name in list(getattr(reward_manager, "active_terms", [])):
                term_cfg = reward_manager.get_term_cfg(name)
                out[name] = float(term_cfg.weight)
            if len(out) > 0:
                return out
        except Exception:
            pass

    rewards_cfg = getattr(getattr(base_env, "cfg", None), "rewards", None)
    if rewards_cfg is None:
        return out

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


def extract_tb_reward_terms(base_env: Any) -> Dict[str, torch.Tensor]:
    data = getattr(base_env, "_tb_reward_terms", None)
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


def extract_reward_manager_weighted_terms(base_env: Any) -> Dict[str, torch.Tensor]:
    reward_manager = getattr(base_env, "reward_manager", None)
    if reward_manager is None:
        return {}

    term_names = list(getattr(reward_manager, "_term_names", []))
    step_reward = getattr(reward_manager, "_step_reward", None)
    if not isinstance(step_reward, torch.Tensor):
        return {}
    if step_reward.dim() != 2 or step_reward.shape[1] != len(term_names):
        return {}

    out: Dict[str, torch.Tensor] = {}
    for i, name in enumerate(term_names):
        out[name] = step_reward[:, i].detach()
    return out


def build_reward_term_views(
    base_env: Any,
    reward_weights: Dict[str, float],
    reward_scale: float,
    reward_clip: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    raw_cache = extract_tb_reward_terms(base_env)
    weighted_terms = extract_reward_manager_weighted_terms(base_env)

    term_names = set(reward_weights.keys()) | set(raw_cache.keys()) | set(weighted_terms.keys())
    if len(term_names) == 0:
        return {}, {}, {}

    raw_terms: Dict[str, torch.Tensor] = {}
    weighted_out: Dict[str, torch.Tensor] = {}
    for name in sorted(term_names):
        w = float(reward_weights.get(name, 1.0))
        if name in weighted_terms:
            weighted = torch.nan_to_num(weighted_terms[name].detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        elif name in raw_cache:
            weighted = torch.nan_to_num(raw_cache[name].detach().float(), nan=0.0, posinf=0.0, neginf=0.0) * w
        else:
            continue
        weighted_out[name] = weighted

        if name in raw_cache:
            raw = torch.nan_to_num(raw_cache[name].detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        elif abs(w) > 1e-12:
            raw = weighted / w
        else:
            raw = torch.zeros_like(weighted)
        raw_terms[name] = raw

    if len(weighted_out) == 0:
        return raw_terms, weighted_out, {}

    dt = float(get_env_step_dt(base_env))
    total_preclip = None
    for value in weighted_out.values():
        contrib = value * dt * float(reward_scale)
        total_preclip = contrib if total_preclip is None else (total_preclip + contrib)

    if total_preclip is None:
        return raw_terms, weighted_out, {}

    if reward_clip > 0.0:
        total_clipped = torch.clamp(total_preclip, -float(reward_clip), float(reward_clip))
    else:
        total_clipped = total_preclip

    clip_factor = torch.ones_like(total_preclip)
    nz = total_preclip.abs() > 1e-8
    clip_factor[nz] = total_clipped[nz] / total_preclip[nz]

    scaled_terms: Dict[str, torch.Tensor] = {}
    for name, value in weighted_out.items():
        scaled_terms[name] = value * dt * float(reward_scale) * clip_factor

    return raw_terms, weighted_out, scaled_terms


class TensorDictStats:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.window_steps = 0
        self.sum: Dict[str, float] = {}
        self.min: Dict[str, float] = {}
        self.max: Dict[str, float] = {}

    def update(self, values: Dict[str, torch.Tensor]) -> None:
        if len(values) == 0:
            return
        self.window_steps += 1
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                continue
            v = torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            mean_v = float(v.mean().item())
            min_v = float(v.min().item())
            max_v = float(v.max().item())
            self.sum[name] = self.sum.get(name, 0.0) + mean_v
            self.min[name] = min_v if name not in self.min else min(self.min[name], min_v)
            self.max[name] = max_v if name not in self.max else max(self.max[name], max_v)

    def flush(self, writer: SummaryWriter, prefix: str, step: int) -> None:
        if self.window_steps <= 0:
            return
        for name in sorted(self.sum.keys()):
            tag = sanitize_tb_tag(name)
            writer.add_scalar(f"{prefix}/{tag}/mean", self.sum[name] / float(self.window_steps), step)
            writer.add_scalar(f"{prefix}/{tag}/min", self.min[name], step)
            writer.add_scalar(f"{prefix}/{tag}/max", self.max[name], step)


class RewardBreakdownAccumulator:
    def __init__(self) -> None:
        self.raw = TensorDictStats()
        self.weighted = TensorDictStats()
        self.scaled = TensorDictStats()

    def reset(self) -> None:
        self.raw.reset()
        self.weighted.reset()
        self.scaled.reset()

    def update(
        self,
        raw_terms: Dict[str, torch.Tensor],
        weighted_terms: Dict[str, torch.Tensor],
        scaled_terms: Dict[str, torch.Tensor],
    ) -> None:
        self.raw.update(raw_terms)
        self.weighted.update(weighted_terms)
        self.scaled.update(scaled_terms)

    def flush(self, writer: SummaryWriter, step: int) -> None:
        self.raw.flush(writer, "RewardRaw", step)
        self.weighted.flush(writer, "RewardWeighted", step)
        self.scaled.flush(writer, "RewardScaled", step)


class InfoTerminationRatioAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.update_steps = 0
        self.sum_ratios: Dict[str, float] = {}
        self.last_ratios: Dict[str, float] = {}

    def update(self, infos: Any, done_count: int) -> None:
        if int(done_count) <= 0:
            return

        ratio_dict = extract_termination_ratio_dict(infos, done_count=done_count)
        if len(ratio_dict) == 0:
            return

        self.update_steps += 1
        for name, value in ratio_dict.items():
            value_f = float(value)
            self.sum_ratios[name] = self.sum_ratios.get(name, 0.0) + value_f
            self.last_ratios[name] = value_f

    def flush(self, writer: SummaryWriter, step: int) -> None:
        if self.update_steps <= 0:
            return

        writer.add_scalar("TerminationInfoRatio/update_steps", float(self.update_steps), step)

        for name in sorted(self.sum_ratios.keys()):
            tag = sanitize_tb_tag(name)
            mean_ratio = self.sum_ratios[name] / float(self.update_steps)
            latest_ratio = self.last_ratios.get(name, 0.0)
            writer.add_scalar(f"TerminationInfoRatio/{tag}/mean", mean_ratio, step)
            writer.add_scalar(f"TerminationInfoRatio/{tag}/latest", latest_ratio, step)


class RollingHistogramLogger:
    def __init__(self, obs_names: List[str], action_names: List[str], window: int, max_samples: int = 0) -> None:
        self.obs_names = list(obs_names)
        self.action_names = list(action_names)
        self.window = max(int(window), 1)
        self.max_samples = int(max_samples)
        self.obs_buffers = [deque(maxlen=self.window) for _ in self.obs_names]
        self.action_buffers = [deque(maxlen=self.window) for _ in self.action_names]

    def _prepare_column(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nan_to_num(x.detach().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if self.max_samples > 0 and y.numel() > self.max_samples:
            idx = torch.randint(0, y.numel(), (self.max_samples,), device=y.device)
            y = y[idx]
        return y.cpu()

    def update(self, states: torch.Tensor, actions: torch.Tensor, state_dim: int) -> None:
        obs_dim = min(len(self.obs_names), int(state_dim), int(states.shape[1]))
        for i in range(obs_dim):
            self.obs_buffers[i].append(self._prepare_column(states[:, i]))

        act_dim = min(len(self.action_names), int(actions.shape[1]))
        for i in range(act_dim):
            self.action_buffers[i].append(self._prepare_column(actions[:, i]))

    def flush(self, writer: SummaryWriter, step: int) -> None:
        for name, buffer in zip(self.obs_names, self.obs_buffers):
            if len(buffer) == 0:
                continue
            writer.add_histogram(f"ObservationDist/{sanitize_tb_tag(name)}", torch.cat(list(buffer), dim=0), step)

        for name, buffer in zip(self.action_names, self.action_buffers):
            if len(buffer) == 0:
                continue
            writer.add_histogram(f"ActionDist/{sanitize_tb_tag(name)}", torch.cat(list(buffer), dim=0), step)


class SkrlLossMirror:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []

    def bind(self, agent: PPO) -> None:
        def wrapped_track_data(tag: str, value: float):
            low = str(tag).lower()
            value_f = to_float(value)
            if "loss" in low and "policy" in low:
                self.policy_losses.append(value_f)
            elif "loss" in low and "value" in low:
                self.value_losses.append(value_f)

        agent.track_data = wrapped_track_data

    def flush(self, writer: SummaryWriter, step: int) -> None:
        if len(self.policy_losses) > 0:
            writer.add_scalar("Loss/policy", float(np.mean(self.policy_losses)), step)
        if len(self.value_losses) > 0:
            writer.add_scalar("Loss/value", float(np.mean(self.value_losses)), step)
        self.reset()


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
        state = self.state_net(self.state_ln(torch.clamp(obs[:, :self.state_dim], -1.0, 1.0)))
        if self.lidar_dim <= 0:
            return self.fuse_net(state)

        lidar = torch.clamp(obs[:, self.state_dim:self.state_dim + self.lidar_dim], 0.0, 1.0)
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


def build_state_names(state_dim: int) -> List[str]:
    if int(state_dim) == len(STATE_OBS_NAMES_16):
        return list(STATE_OBS_NAMES_16)
    return [f"state_{i}" for i in range(int(state_dim))]


def build_action_names(act_dim: int) -> List[str]:
    if int(act_dim) == len(ACTION_NAMES_4):
        return list(ACTION_NAMES_4)
    return [f"action_{i}" for i in range(int(act_dim))]


def main() -> None:
    print(f"[INFO] task={args.task}, num_envs={args.num_envs}, device={args.device}", flush=True)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )

    curriculum_levels = get_cfg_obstacle_curriculum_levels(env_cfg)
    required_shared_obstacles = max(
        int(args.num_obstacles),
        max(curriculum_levels) if len(curriculum_levels) > 0 else 0,
    )

    print(f"[INFO] obstacle curriculum levels={curriculum_levels}", flush=True)
    if required_shared_obstacles != int(args.num_obstacles):
        print(
            f"[INFO] Expanding shared obstacle pool from requested {args.num_obstacles} "
            f"to {required_shared_obstacles} to satisfy curriculum levels.",
            flush=True,
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
    ObstacleSpawner(
        num_obstacles=int(required_shared_obstacles),
        seed=int(args.seed),
    ).spawn_obstacles()

    print("[INFO] Spawning workspace walls...", flush=True)
    WallSpawner(
        x_bounds=(-80.0, 80.0),
        y_bounds=(-80.0, 80.0),
        z_bounds=(0.0, 10.0),
        wall_thickness=0.5,
        color=(0.7, 0.7, 0.2),
        wall_colors={
        "Wall_XMin": (0.5, 1.0, 1.0),  
        "Wall_XMax": (1.0, 1.0, 0.5),  
        "Wall_YMin": (0.0, 1.0, 1.0),  
        "Wall_YMax": (1.0, 1.0, 0.0),  
        "Wall_ZMin": (1.0, 1.0, 1.0),  
        "Wall_ZMax": (0.0, 0.0, 0.0),  
    },
    ).spawn_walls()

    print("[INFO] Creating env...", flush=True)
    base_env = gym.make(args.task, cfg=env_cfg).unwrapped

    scale_robot_visual_only(
        num_envs=base_env.num_envs,
        visual_scale=(20.0, 20.0, 10.0),
    )

    base_env.scene.filter_collisions(
        global_prim_paths=[
            "/World/ground",
            "/World/Obstacles",
            "/World/Wall",
        ]
    )

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
    step_dt = get_env_step_dt(base_env)

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

    writer = SummaryWriter(log_dir=str(tb_dir))
    if len(curriculum_levels) > 0:
        writer.add_text("run/obstacle_curriculum_levels", str(curriculum_levels), 0)
    writer.add_text("run/args", str(vars(args)), 0)
    writer.add_text("run/dims", f"obs={obs_dim}, state={state_dim}, lidar={lidar_dim}, act={act_dim}", 0)
    writer.add_text("run/step_dt", f"{step_dt:.8f}", 0)

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

    loss_mirror = SkrlLossMirror()
    loss_mirror.bind(agent)

    raw_obs, infos = env.reset()
    states = sanitize_states(
        ensure_obs_shape(extract_policy_obs(raw_obs), num_envs, obs_dim),
        state_dim=state_dim,
        lidar_dim=lidar_dim,
    )

    last_good_snapshot = snapshot_models(models)

    reward_weights = extract_reward_weights(base_env)
    effective_per_step = {k: float(v) * step_dt * float(args.reward_scale) for k, v in reward_weights.items()}
    print(f"[INFO] reward weights(raw): {reward_weights}", flush=True)
    print(f"[INFO] reward weights(effective per-step before total clip): {effective_per_step}", flush=True)

    reward_window = RewardBreakdownAccumulator()
    termination_ratio_window = InfoTerminationRatioAccumulator()
    hist_logger = RollingHistogramLogger(
        obs_names=build_state_names(state_dim),
        action_names=build_action_names(act_dim),
        window=int(args.dist_window),
        max_samples=int(args.dist_max_samples),
    )

    latest_curriculum_log: Dict[str, float] = {}
    pbar = tqdm(range(int(args.timesteps)), ncols=110)

    try:
        for t in pbar:
            global_step = t + 1
            agent.pre_interaction(timestep=t, timesteps=int(args.timesteps))

            with torch.no_grad():
                act_output = agent.act(states, timestep=t, timesteps=int(args.timesteps))

            actions = ensure_action_shape(extract_actions(act_output, act_dim), num_envs, act_dim).float()
            if not torch.isfinite(actions).all():
                raise RuntimeError(f"Non-finite actions detected before env.step at t={t}")
            actions = sanitize_actions(actions)

            rollout_boundary = (global_step % int(args.rollouts) == 0)
            if rollout_boundary:
                last_good_snapshot = snapshot_models(models)

            states_before_step = states.clone()
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

            raw_terms, weighted_terms, scaled_terms = build_reward_term_views(
                base_env,
                reward_weights=reward_weights,
                reward_scale=float(args.reward_scale),
                reward_clip=float(args.reward_clip),
            )

            reward_window.update(raw_terms=raw_terms, weighted_terms=weighted_terms, scaled_terms=scaled_terms)
            clear_tb_caches(base_env)

            done_mask_tensor = terminated | truncated
            done_count = int(done_mask_tensor.sum().item())
            has_done = done_count > 0

            termination_ratio_window.update(infos, done_count=done_count)

            if has_done:
                curriculum_log_dict = extract_prefixed_log_scalars(infos, "Curriculum/")
                if len(curriculum_log_dict) > 0:
                    latest_curriculum_log.update(curriculum_log_dict)

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
            if rollout_boundary:
                loss_mirror.flush(writer, global_step)

            if rollout_boundary and not models_are_finite(models):
                debug_dir = exp_dir / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                torch.save(last_good_snapshot, debug_dir / f"last_good_before_nan_t{t}.pt")
                raise RuntimeError(
                    f"Non-finite model parameters detected immediately after PPO update at t={t}. "
                    f"Snapshot saved to {debug_dir}."
                )

            if not args.headless:
                try:
                    env.render()
                except Exception:
                    pass

            should_log_dist = int(args.dist_interval) > 0 and (
                (global_step % int(args.dist_interval) == 0) or (global_step == int(args.timesteps))
            )
            should_log_scalars = int(args.tb_interval) > 0 and (
                (global_step % int(args.tb_interval) == 0) or (global_step == int(args.timesteps))
            )

            if should_log_dist:
                hist_logger.update(states=states, actions=actions, state_dim=state_dim)
                hist_logger.flush(writer, global_step)
                if not should_log_scalars:
                    writer.flush()

            if should_log_scalars:
                for key, value in sorted(latest_curriculum_log.items()):
                    writer.add_scalar(sanitize_tb_tag(key), value, global_step)
                reward_window.flush(writer, global_step)
                termination_ratio_window.flush(writer, global_step)
                writer.flush()
                reward_window.reset()
                termination_ratio_window.reset()

            if int(args.grad_hist_interval) > 0 and rollout_boundary:
                update_idx = global_step // int(args.rollouts)
                if update_idx % int(args.grad_hist_interval) == 0:
                    log_gradients(writer, models, global_step, int(args.grad_hist_samples))
                    writer.flush()

            if int(args.checkpoint_interval) > 0 and (global_step % int(args.checkpoint_interval) == 0):
                ckpt_dir = exp_dir / "manual_checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {name: model.state_dict() for name, model in models.items()},
                    ckpt_dir / f"models_t{global_step}.pt",
                )

            if int(args.cuda_clean_interval) > 0 and (global_step % int(args.cuda_clean_interval) == 0):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.set_description(
                f"t={t} envR={rewards.mean().item():+.3f} trainR={train_rewards.mean().item():+.3f} done={done_count}"
            )

            if global_step <= 5 or global_step % 200 == 0:
                print_env0_transition(
                    step=global_step,
                    states=states_before_step,
                    actions=actions,
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated,
                    next_states=next_states,
                    state_dim=state_dim,
                    lidar_dim=lidar_dim,
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
