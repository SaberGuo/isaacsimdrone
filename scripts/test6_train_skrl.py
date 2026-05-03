from __future__ import annotations

# Standard-library imports must stay above AppLauncher construction.
import argparse
import copy
import gc
import json
import os
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from isaaclab.app import AppLauncher

# Keep CUDA allocator policy visible before torch is imported.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)

# -----------------------------------------------------------------------------
# CLI and Isaac application bootstrap
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser("Stable skrl PPO trainer for IsaacLab drone lidar task")
parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-Lidar-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--num_obstacles", type=int, default=100)
parser.add_argument("--timesteps", type=int, default=10_000_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--state_dim", type=int, default=18)
parser.add_argument("--lidar_dim", type=int, default=432)
parser.add_argument("--feat_dim", type=int, default=256)
parser.add_argument("--model_cfg_path", type=str, default="")
parser.add_argument("--model_cfg_json", type=str, default="")
parser.add_argument("--rollouts", type=int, default=512)
parser.add_argument("--learning_epochs", type=int, default=6)
parser.add_argument("--mini_batches", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1.0e-5)
parser.add_argument("--_lambda", type=float, default=0.95)
parser.add_argument("--discount_factor", type=float, default=0.995)
parser.add_argument("--ratio_clip", type=float, default=0.15)
parser.add_argument("--value_clip", type=float, default=0.2)
parser.add_argument("--value_loss_scale", type=float, default=0.5)
parser.add_argument("--grad_norm_clip", type=float, default=0.8)
parser.add_argument("--entropy_coef", type=float, default=6.0e-3)
parser.add_argument("--kl_threshold", type=float, default=0.008)
parser.add_argument("--use_kl_adaptive_lr", action="store_true")
parser.add_argument("--no_use_kl_adaptive_lr", dest="use_kl_adaptive_lr", action="store_false")
parser.set_defaults(use_kl_adaptive_lr=True)
parser.add_argument("--kl_adaptive_lr_threshold", type=float, default=0.006)
parser.add_argument("--kl_adaptive_min_lr", type=float, default=2e-6)
parser.add_argument("--kl_adaptive_max_lr", type=float, default=2e-5)
parser.add_argument("--kl_adaptive_kl_factor", type=float, default=2.0)
parser.add_argument("--kl_adaptive_lr_factor", type=float, default=1.5)
parser.add_argument("--clip_predicted_values", action="store_true")
parser.add_argument("--no_clip_predicted_values", dest="clip_predicted_values", action="store_false")
parser.set_defaults(clip_predicted_values=True)
parser.add_argument("--reward_scale", type=float, default=1.0)
parser.add_argument("--reward_clip", type=float, default=500.0)
parser.add_argument("--tb_interval", type=int, default=500)
parser.add_argument("--dist_interval", type=int, default=0)
parser.add_argument("--dist_window", type=int, default=10)
parser.add_argument("--dist_max_samples", type=int, default=2048)
parser.add_argument("--checkpoint_interval", type=int, default=50000)
parser.add_argument("--cuda_clean_interval", type=int, default=0)
parser.add_argument("--extra_tb_subdir", type=str, default="extra_tb")
parser.add_argument("--keep_infos", action="store_true", default=False)
parser.add_argument("--grad_hist_interval", type=int, default=0)
parser.add_argument("--grad_hist_samples", type=int, default=65536)
parser.add_argument("--debug_act", action="store_true", default=False)
parser.add_argument("--pbar_interval", type=int, default=50)
parser.add_argument("--render_interval", type=int, default=0)
parser.add_argument("--finite_check_interval", type=int, default=0)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Imports below require the Isaac/Omniverse application to be running.
import gymnasium as gym
import isaacsim.core.utils.bounds as bounds_utils
import isaacsim.core.utils.prims as prim_utils
import numpy as np
import torch
import torch.nn as nn
import omniperception_isaacdrone.tasks.test6_registry as _test6_registry  # noqa: F401
from gymnasium.spaces import Box
from isaaclab_tasks.utils import parse_env_cfg
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from omniperception_isaacdrone.envs.test6_env import WallSpawner, setup_global_obstacles
from omniperception_isaacdrone.models import Policy, Value, model_cfg_to_dict, resolve_model_cfg

# -----------------------------------------------------------------------------
# Runtime metadata and backend settings
# -----------------------------------------------------------------------------
STATE_OBS_NAMES_18 = [
    "root_pos_z", "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z",
    "root_lin_vel_x", "root_lin_vel_y", "root_lin_vel_z",
    "root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z",
    "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
    "goal_dir_x", "goal_dir_y", "goal_dir_z", "goal_dist",
]
ACTION_NAMES_4 = ["vx_cmd", "vy_cmd", "vz_cmd", "yaw_rate_cmd"]

DEBUG_PRINT = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# -----------------------------------------------------------------------------
# USD and debug helpers
# -----------------------------------------------------------------------------
def scale_robot_visual_only(num_envs: int, visual_scale=(20.0, 20.0, 10.0)) -> None:
    """仅缩放机器人视觉网格，不影响碰撞体。"""
    stage = prim_utils.get_prim_at_path("/World").GetStage()
    sx, sy, sz = map(float, visual_scale)
    for i in range(int(num_envs)):
        visual_path = f"/World/envs/env_{i}/Robot/body/body_visual"
        prim = stage.GetPrimAtPath(visual_path)
        if not prim.IsValid():
            continue
        xform = UsdGeom.Xformable(prim)
        scale_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
        if len(scale_ops) > 0:
            scale_ops[0].Set(Gf.Vec3f(sx, sy, sz))
        else:
            xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))


def _format_vec3(v) -> str:
    return f"({float(v[0]):.6f}, {float(v[1]):.6f}, {float(v[2]):.6f})"

def _get_world_scale_from_prim(prim) -> np.ndarray:
    xform_cache = UsdGeom.XformCache()
    world_m = xform_cache.GetLocalToWorldTransform(prim)
    sx = Gf.Vec3d(world_m[0][0], world_m[0][1], world_m[0][2]).GetLength()
    sy = Gf.Vec3d(world_m[1][0], world_m[1][1], world_m[1][2]).GetLength()
    sz = Gf.Vec3d(world_m[2][0], world_m[2][1], world_m[2][2]).GetLength()
    return np.array([float(sx), float(sy), float(sz)], dtype=np.float64)

def _get_world_translation_from_prim(prim) -> np.ndarray:
    xform_cache = UsdGeom.XformCache()
    world_m = xform_cache.GetLocalToWorldTransform(prim)
    t = world_m.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)

def _read_collision_geom_world_size(prim) -> dict:
    info = {"geom_type": prim.GetTypeName(), "authored": {}, "world": {}}
    world_scale = _get_world_scale_from_prim(prim)
    world_translation = _get_world_translation_from_prim(prim)
    info["world"]["scale_xyz"] = world_scale.tolist()
    info["world"]["translation_xyz"] = world_translation.tolist()
    return info

def print_robot_collision_shapes(env_index: int = 0, robot_rel_path: str = "Robot") -> None:
    pass

def debug_print(msg: str) -> None:
    if DEBUG_PRINT:
        print(msg, flush=True)

def print_env0_transition(step, states, actions, rewards, terminated, truncated, next_states, state_dim, lidar_dim):
    pass


# -----------------------------------------------------------------------------
# Observation/action shape and value sanitizers
# -----------------------------------------------------------------------------
def format_array_preview(x: np.ndarray, max_items: int = 16) -> str:
    x = np.asarray(x).reshape(-1)
    if x.size <= max_items:
        return np.array2string(x, precision=3, separator=", ")
    return f"{np.array2string(x[:max_items], precision=3, separator=', ')} ..."

def print_space_bounds(name: str, space: gym.Space) -> None:
    print(f"\n[SPACE] {name}: type={type(space).__name__}", flush=True)

def sanitize_tb_tag(tag: str) -> str:
    return str(tag).replace(".", "/").replace(" ", "_")

def to_float(x: Any) -> float:
    if isinstance(x, (float, int)): return float(x)
    if isinstance(x, torch.Tensor):
        y = x.detach().float()
        if y.numel() == 0: return 0.0
        if not torch.isfinite(y).all(): y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return float(y.mean().item())
    try: return float(x)
    except: return 0.0

def sample_flat(x: torch.Tensor, max_samples: int) -> torch.Tensor:
    y = x.detach().reshape(-1)
    if max_samples <= 0 or y.numel() <= max_samples: return y
    idx = torch.randint(0, y.numel(), (max_samples,), device=y.device)
    return y[idx]

def extract_policy_obs(obs: Any) -> torch.Tensor:
    if isinstance(obs, torch.Tensor): return obs
    if isinstance(obs, dict):
        if "policy" in obs and isinstance(obs["policy"], torch.Tensor): return obs["policy"]
        for value in obs.values():
            if isinstance(value, torch.Tensor): return value
    raise RuntimeError(f"Unsupported observation type: {type(obs)}")

def extract_actions(act_output: Any, act_dim: int) -> torch.Tensor:
    if isinstance(act_output, torch.Tensor): return act_output
    if isinstance(act_output, (tuple, list)):
        for item in act_output:
            if isinstance(item, torch.Tensor): return item
    if isinstance(act_output, dict):
        for key in ("actions", "action"):
            if isinstance(value := act_output.get(key, None), torch.Tensor): return value
    raise RuntimeError("Unsupported agent.act output")

def ensure_obs_shape(x: torch.Tensor, num_envs: int, obs_dim: int) -> torch.Tensor:
    return x.view(num_envs, obs_dim)

def ensure_action_shape(x: torch.Tensor, num_envs: int, act_dim: int) -> torch.Tensor:
    if x.dim() == 1 and x.shape[0] == act_dim: x = x.unsqueeze(0).repeat(num_envs, 1)
    elif x.dim() == 2 and x.shape == (1, act_dim) and num_envs > 1: x = x.repeat(num_envs, 1)
    return x

def ensure_vec_shape(x: torch.Tensor, num_envs: int, name: str) -> torch.Tensor:
    if x.dim() == 1 and x.shape[0] == num_envs: return x.unsqueeze(-1)
    if x.dim() == 2 and x.shape[0] == num_envs: return x
    raise RuntimeError(f"Invalid {name} shape")

def sanitize_states(states: torch.Tensor, state_dim: int, lidar_dim: int) -> torch.Tensor:
    states = torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    state = torch.clamp(states[:, :state_dim], -1.0, 1.0)
    if lidar_dim <= 0: return state
    lidar = torch.clamp(states[:, state_dim: state_dim + lidar_dim], 0.0, 1.0)
    return torch.cat([state, lidar], dim=-1)

def sanitize_actions(actions: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.nan_to_num(actions.float(), nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

def scale_rewards(rewards: torch.Tensor, scale: float, clip: float) -> torch.Tensor:
    rewards = torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0) * float(scale)
    if clip > 0.0: rewards = torch.clamp(rewards, -float(clip), float(clip))
    return rewards

def infer_single_dim_from_box(space: Box, num_envs: int) -> int:
    return int(np.prod(space.shape))

def get_state_lidar_dims(base_env: Any, obs_dim: int) -> tuple[int, int]:
    state_dim = int(getattr(base_env, "policy_state_dim", 0))
    lidar_dim = int(getattr(base_env, "policy_lidar_dim", 0))
    if state_dim > 0 and state_dim + lidar_dim == obs_dim: return state_dim, lidar_dim
    norm_cfg = getattr(getattr(base_env, "cfg", None), "normalization", None)
    state_dim = int(getattr(norm_cfg, "state_dim", args.state_dim))
    return state_dim, obs_dim - state_dim


def infer_lidar_grid_shape(base_env: Any, lidar_dim: int) -> tuple[int, int] | None:
    if int(lidar_dim) <= 0:
        return None
    try:
        obs_cfg = getattr(getattr(getattr(base_env, "cfg", None), "observations", None), "policy", None)
        lidar_term = getattr(obs_cfg, "lidar_grid", None)
        params = getattr(lidar_term, "params", None) or {}
        theta_min = float(params["theta_min"])
        theta_max = float(params["theta_max"])
        delta_theta = float(params["delta_theta"])
        phi_min = float(params["phi_min"])
        phi_max = float(params["phi_max"])
        delta_phi = float(params["delta_phi"])
        theta_bins = max(int(round((theta_max - theta_min) / delta_theta)), 1)
        phi_bins = max(int(round((phi_max - phi_min) / delta_phi)), 1)
        if theta_bins * phi_bins == int(lidar_dim):
            return theta_bins, phi_bins
    except Exception:
        pass
    return None


# -----------------------------------------------------------------------------
# skrl space adapter
# -----------------------------------------------------------------------------
def build_skrl_spaces(base_env: Any, state_dim: int, lidar_dim: int) -> tuple[int, int, gym.spaces.Dict, Box]:
    num_envs = int(getattr(base_env, "num_envs", 1))
    act_space = getattr(base_env, "single_action_space", getattr(base_env, "action_space", None))
    act_dim = infer_single_dim_from_box(act_space, num_envs) if isinstance(act_space, gym.spaces.Box) else 4
    obs_dim = state_dim + lidar_dim
    obs_low, obs_high = -np.ones((obs_dim,), dtype=np.float32), np.ones((obs_dim,), dtype=np.float32)
    if lidar_dim > 0: obs_low[state_dim:] = 0.0
    return obs_dim, act_dim, gym.spaces.Dict({"policy": Box(low=obs_low, high=obs_high, dtype=np.float32)}), Box(low=-np.ones((act_dim,), dtype=np.float32), high=np.ones((act_dim,), dtype=np.float32), dtype=np.float32)


class SkrlSpaceAdapter(gym.Wrapper):
    """将原始环境观测/动作空间适配为 skrl 所需格式。"""
    def __init__(self, env: gym.Env, obs_space: gym.spaces.Dict, act_space: Box, state_dim: int, lidar_dim: int):
        super().__init__(env)
        self.state_dim, self.lidar_dim, self.obs_dim = int(state_dim), int(lidar_dim), int(state_dim) + int(lidar_dim)
        self.observation_space = self.single_observation_space = obs_space
        self.action_space = self.single_action_space = act_space
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.device = getattr(env, "device", None)

    def _convert_obs(self, raw_obs: Any) -> dict[str, torch.Tensor]:
        return {"policy": sanitize_states(ensure_obs_shape(extract_policy_obs(raw_obs), self.num_envs, self.obs_dim), self.state_dim, self.lidar_dim)}

    def reset(self, **kwargs):
        raw_obs, infos = self.env.reset(**kwargs)
        return self._convert_obs(raw_obs), infos

    def step(self, actions):
        raw_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        return self._convert_obs(raw_obs), rewards, terminated, truncated, infos


# -----------------------------------------------------------------------------
# Model safety and rollout log extraction
# -----------------------------------------------------------------------------
def models_are_finite(models: dict[str, nn.Module]) -> bool:
    for model in models.values():
        for p in model.parameters():
            if not torch.isfinite(p).all(): return False
    return True

def snapshot_models(models: dict[str, nn.Module]) -> dict[str, dict[str, torch.Tensor]]:
    return {name: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} for name, model in models.items()}

def extract_log_dict(infos: Any) -> Dict[str, Any]:
    if not isinstance(infos, dict): return {}
    if isinstance(log := infos.get("log", None), dict): return log
    if isinstance(extras := infos.get("extras", None), dict):
        if isinstance(log := extras.get("log", None), dict): return log
    return {}

def _get_done_mask(terminated: torch.Tensor, truncated: torch.Tensor) -> torch.Tensor:
    num_envs = int(terminated.shape[0])
    terminated_mask = ensure_vec_shape(torch.nan_to_num(terminated.float(), nan=0.0, posinf=0.0, neginf=0.0), num_envs, "terminated").squeeze(-1).bool()
    truncated_mask = ensure_vec_shape(torch.nan_to_num(truncated.float(), nan=0.0, posinf=0.0, neginf=0.0), num_envs, "truncated").squeeze(-1).bool()
    return terminated_mask | truncated_mask

def _get_termination_term_mask(base_env: Any, name: str, num_envs: int) -> torch.Tensor | None:
    try:
        term = base_env.termination_manager.get_term(name)
    except Exception:
        return None
    if not isinstance(term, torch.Tensor):
        return None
    mask = ensure_vec_shape(torch.nan_to_num(term.float(), nan=0.0, posinf=0.0, neginf=0.0), num_envs, f"termination_{name}")
    return mask.squeeze(-1).bool()

def build_primary_termination_count_dict(base_env: Any, terminated: torch.Tensor, truncated: torch.Tensor) -> Dict[str, float]:
    done_mask = _get_done_mask(terminated, truncated)
    if not done_mask.any():
        return {}

    num_envs = int(done_mask.shape[0])
    truncated_mask = ensure_vec_shape(torch.nan_to_num(truncated.float(), nan=0.0, posinf=0.0, neginf=0.0), num_envs, "truncated").squeeze(-1).bool()

    raw_masks: Dict[str, torch.Tensor] = {}
    for name in ("reached_goal", "collision", "oob"):
        if (mask := _get_termination_term_mask(base_env, name, num_envs)) is not None:
            raw_masks[name] = mask & done_mask

    time_out_mask = _get_termination_term_mask(base_env, "time_out", num_envs)
    if time_out_mask is None:
        raw_masks["time_out"] = truncated_mask & done_mask
    else:
        raw_masks["time_out"] = (time_out_mask | truncated_mask) & done_mask

    counts: Dict[str, float] = {}
    remaining = done_mask.clone()

    # 使用互斥主因统计，保证各终止原因求和恰好等于 done_count。
    for name in ("reached_goal", "collision", "oob", "time_out"):
        mask = raw_masks.get(name)
        if mask is None:
            continue
        primary = remaining & mask
        count = int(primary.sum().item())
        if count > 0:
            counts[name] = float(count)
        remaining &= ~primary

    if remaining.any():
        counts["other"] = float(remaining.sum().item())

    return counts

def build_termination_ratio_dict(count_dict: Dict[str, float], done_count: int) -> Dict[str, float]:
    if int(done_count) <= 0 or len(count_dict) == 0:
        return {}
    denom = max(float(done_count), 1.0)
    return {k: float(v) / denom for k, v in count_dict.items()}

def extract_prefixed_log_scalars(infos: Any, prefix: str) -> Dict[str, float]:
    return {k: to_float(v) for k, v in extract_log_dict(infos).items() if k.startswith(prefix)}

def get_env_step_dt(base_env: Any) -> float:
    if hasattr(base_env, "step_dt"): return float(base_env.step_dt)
    try: return float(base_env.cfg.sim.dt) * float(base_env.cfg.decimation)
    except Exception: return 1.0 / 60.0

def extract_reward_weights(base_env: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if (reward_manager := getattr(base_env, "reward_manager", None)):
        try:
            for name in list(getattr(reward_manager, "active_terms", [])): out[name] = float(reward_manager.get_term_cfg(name).weight)
            if len(out) > 0: return out
        except Exception: pass
    if (rewards_cfg := getattr(getattr(base_env, "cfg", None), "rewards", None)):
        for name in dir(rewards_cfg):
            if name.startswith("_"): continue
            try: out[name] = float(getattr(getattr(rewards_cfg, name), "weight", None))
            except Exception: pass
    return out

def extract_tb_reward_terms(base_env: Any) -> Dict[str, torch.Tensor]:
    return data if isinstance(data := getattr(base_env, "_tb_reward_terms", None), dict) else {}

def extract_tb_aux_terms(base_env: Any) -> Dict[str, torch.Tensor]:
    return data if isinstance(data := getattr(base_env, "_tb_aux_terms", None), dict) else {}

def clear_tb_caches(base_env: Any) -> None:
    for attr in ("_tb_reward_terms", "_tb_aux_terms"):
        if isinstance(d := getattr(base_env, attr, None), dict): d.clear()

def extract_reward_manager_weighted_terms(base_env: Any) -> Dict[str, torch.Tensor]:
    if not (rm := getattr(base_env, "reward_manager", None)): return {}
    term_names, step_reward = list(getattr(rm, "_term_names", [])), getattr(rm, "_step_reward", None)
    if not isinstance(step_reward, torch.Tensor) or step_reward.dim() != 2 or step_reward.shape[1] != len(term_names): return {}
    return {name: step_reward[:, i].detach() for i, name in enumerate(term_names)}

def build_reward_term_views(base_env: Any, reward_weights: Dict[str, float], reward_scale: float, reward_clip: float) -> tuple:
    raw_cache, weighted_terms = extract_tb_reward_terms(base_env), extract_reward_manager_weighted_terms(base_env)
    term_names = set(reward_weights.keys()) | set(raw_cache.keys()) | set(weighted_terms.keys())
    if len(term_names) == 0: return {}, {}, {}
    raw_terms, weighted_out = {}, {}
    for name in sorted(term_names):
        w = float(reward_weights.get(name, 1.0))
        if name in weighted_terms: weighted = torch.nan_to_num(weighted_terms[name].detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        elif name in raw_cache: weighted = torch.nan_to_num(raw_cache[name].detach().float(), nan=0.0, posinf=0.0, neginf=0.0) * w
        else: continue
        weighted_out[name] = weighted
        if name in raw_cache: raw_terms[name] = torch.nan_to_num(raw_cache[name].detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        elif abs(w) > 1e-12: raw_terms[name] = weighted / w
        else: raw_terms[name] = torch.zeros_like(weighted)
    if len(weighted_out) == 0: return raw_terms, weighted_out, {}
    dt, total_preclip = float(get_env_step_dt(base_env)), None
    for value in weighted_out.values(): total_preclip = (value * dt * reward_scale) if total_preclip is None else (total_preclip + value * dt * reward_scale)
    if total_preclip is None: return raw_terms, weighted_out, {}
    total_clipped = torch.clamp(total_preclip, -float(reward_clip), float(reward_clip)) if reward_clip > 0.0 else total_preclip
    clip_factor = torch.ones_like(total_preclip)
    nz = total_preclip.abs() > 1e-8
    clip_factor[nz] = total_clipped[nz] / total_preclip[nz]
    return raw_terms, weighted_out, {name: value * dt * reward_scale * clip_factor for name, value in weighted_out.items()}


# -----------------------------------------------------------------------------
# TensorBoard accumulators
# -----------------------------------------------------------------------------
class TensorDictStats:
    """按步累积张量字典统计量（均值/最小/最大）。"""
    def __init__(self): self.reset()
    def reset(self): self.window_steps, self.sum, self.min, self.max = 0, {}, {}, {}
    def update(self, values: Dict[str, torch.Tensor]):
        if len(values) == 0: return
        self.window_steps += 1
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.numel() == 0: continue
            v = torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            mean_v, min_v, max_v = v.mean(), v.min(), v.max()
            self.sum[name] = mean_v if name not in self.sum else (self.sum[name] + mean_v)
            self.min[name] = min_v if name not in self.min else torch.minimum(self.min[name], min_v)
            self.max[name] = max_v if name not in self.max else torch.maximum(self.max[name], max_v)
    def flush(self, writer: SummaryWriter, prefix: str, step: int):
        if self.window_steps <= 0: return
        for name in sorted(self.sum.keys()):
            tag = sanitize_tb_tag(name)
            writer.add_scalar(f"{prefix}/{tag}/mean", float((self.sum[name] / float(self.window_steps)).item()), step)
            writer.add_scalar(f"{prefix}/{tag}/min", float(self.min[name].item()), step)
            writer.add_scalar(f"{prefix}/{tag}/max", float(self.max[name].item()), step)


class RewardBreakdownAccumulator:
    """分项累积原始/加权/缩放奖励统计。"""
    def __init__(self): self.raw, self.weighted, self.scaled = TensorDictStats(), TensorDictStats(), TensorDictStats()
    def reset(self): self.raw.reset(); self.weighted.reset(); self.scaled.reset()
    def update(self, raw_terms, weighted_terms, scaled_terms): self.raw.update(raw_terms); self.weighted.update(weighted_terms); self.scaled.update(scaled_terms)
    def flush(self, writer, step): self.raw.flush(writer, "RewardRaw", step); self.weighted.flush(writer, "RewardWeighted", step); self.scaled.flush(writer, "RewardScaled", step)


class InfoTerminationRatioAccumulator:
    """累积各终止原因的比率统计。"""
    def __init__(self): self.reset()
    def reset(self): self.update_steps, self.sum_ratios, self.last_ratios = 0, {}, {}
    def update(self, ratio_dict: Dict[str, float]):
        if len(ratio_dict) == 0: return
        self.update_steps += 1
        for name, value in ratio_dict.items():
            self.sum_ratios[name] = self.sum_ratios.get(name, 0.0) + float(value)
            self.last_ratios[name] = float(value)
    def flush(self, writer, step):
        if self.update_steps <= 0: return
        writer.add_scalar("TerminationInfoRatio/update_steps", float(self.update_steps), step)
        for name in sorted(self.sum_ratios.keys()):
            writer.add_scalar(f"TerminationInfoRatio/{sanitize_tb_tag(name)}/mean", self.sum_ratios[name] / float(self.update_steps), step)
            writer.add_scalar(f"TerminationInfoRatio/{sanitize_tb_tag(name)}/latest", self.last_ratios.get(name, 0.0), step)


class RollingHistogramLogger:
    """滑动窗口内观测与动作的直方图记录器。"""
    def __init__(self, obs_names, action_names, window, max_samples=0):
        self.obs_names, self.action_names, self.window, self.max_samples = list(obs_names), list(action_names), max(int(window), 1), int(max_samples)
        self.obs_buffers = [deque(maxlen=self.window) for _ in self.obs_names]
        self.action_buffers = [deque(maxlen=self.window) for _ in self.action_names]
    def _prepare_column(self, x: torch.Tensor):
        y = torch.nan_to_num(x.detach().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        return y[torch.randint(0, y.numel(), (self.max_samples,), device=y.device)].cpu() if self.max_samples > 0 and y.numel() > self.max_samples else y.cpu()
    def update(self, states, actions, state_dim):
        for i in range(min(len(self.obs_names), int(state_dim), int(states.shape[1]))): self.obs_buffers[i].append(self._prepare_column(states[:, i]))
        for i in range(min(len(self.action_names), int(actions.shape[1]))): self.action_buffers[i].append(self._prepare_column(actions[:, i]))
    def flush(self, writer, step):
        for name, buffer in zip(self.obs_names, self.obs_buffers):
            if len(buffer) > 0: writer.add_histogram(f"ObservationDist/{sanitize_tb_tag(name)}", torch.cat(list(buffer), dim=0), step)
        for name, buffer in zip(self.action_names, self.action_buffers):
            if len(buffer) > 0: writer.add_histogram(f"ActionDist/{sanitize_tb_tag(name)}", torch.cat(list(buffer), dim=0), step)


class SkrlLossMirror:
    """拦截 skrl agent.track_data 以记录策略/价值损失。"""
    def __init__(self): self.reset()
    def reset(self): self.policy_losses, self.value_losses = [], []
    def bind(self, agent: PPO):
        original_track_data = agent.track_data
        def wrapped_track_data(tag, value):
            if "loss" in (low := str(tag).lower()) and "policy" in low: self.policy_losses.append(to_float(value))
            elif "loss" in low and "value" in low: self.value_losses.append(to_float(value))
            original_track_data(tag, value)
        agent.track_data = wrapped_track_data
    def flush(self, writer, step):
        if len(self.policy_losses) > 0: writer.add_scalar("Loss/policy", float(np.mean(self.policy_losses)), step)
        if len(self.value_losses) > 0: writer.add_scalar("Loss/value", float(np.mean(self.value_losses)), step)
        self.reset()


def log_gradients(writer, models, step, max_samples):
    """记录各模型参数梯度的范数、有限比率与直方图。"""
    for model_key, model in models.items():
        for name, p in model.named_parameters():
            if p.grad is None or (g := p.grad.detach()).numel() == 0: continue
            try: writer.add_scalar(f"Gradients/{model_key}/norm/{sanitize_tb_tag(name)}", to_float(g.norm()), step)
            except Exception: pass
            g_s = sample_flat(g, max_samples=max_samples).float()
            writer.add_scalar(f"Gradients/{model_key}/finite_ratio/{sanitize_tb_tag(name)}", float((finite := torch.isfinite(g_s)).float().mean().item()) if g_s.numel() > 0 else 0.0, step)
            if finite.any() and (g_f := g_s[finite].detach().cpu()).numel() > 0:
                try: writer.add_histogram(f"Gradients/{model_key}/hist/{sanitize_tb_tag(name)}", g_f, step)
                except Exception: pass


# -----------------------------------------------------------------------------
# Run configuration helpers
# -----------------------------------------------------------------------------
def build_state_names(state_dim: int) -> List[str]:
    return list(STATE_OBS_NAMES_18) if int(state_dim) == len(STATE_OBS_NAMES_18) else [f"state_{i}" for i in range(int(state_dim))]

def build_action_names(act_dim: int) -> List[str]:
    return list(ACTION_NAMES_4) if int(act_dim) == len(ACTION_NAMES_4) else [f"action_{i}" for i in range(int(act_dim))]


def json_dumps_pretty(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def write_run_config_snapshot(
    config_path: Path,
    args_ns: argparse.Namespace,
    env_cfg: Any,
    runtime_meta: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> None:
    try:
        env_cfg_dict = env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else {"_error": "env_cfg has no to_dict()"}
    except Exception as exc:
        env_cfg_dict = {"_error": f"Failed to serialize env_cfg: {exc}"}
    content = "\n".join([
        "[test6_train_skrl.py args]",
        json_dumps_pretty(vars(args_ns)),
        "",
        "[test6_env_cfg.py resolved_cfg]",
        json_dumps_pretty(env_cfg_dict),
        "",
        "[runtime_meta]",
        json_dumps_pretty(runtime_meta),
        "",
        "[model_cfg]",
        json_dumps_pretty(model_cfg),
        "",
    ])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")


# -----------------------------------------------------------------------------
# Training entry point
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"[INFO] task={args.task}, num_envs={args.num_envs}, device={args.device}", flush=True)
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.scene.replicate_physics = True
    env_cfg.scene.filter_collisions = True
    try:
        action_params = getattr(env_cfg.actions.root_twist, "params", None) or {}
        if isinstance(action_params, dict):
            action_params["debug_print"] = bool(args.debug_act)
            env_cfg.actions.root_twist.params = action_params
    except Exception:
        pass
    try:
        setattr(env_cfg, "seed", int(args.seed))
    except Exception:
        pass
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
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
    print("[INFO] Setting up global obstacles template...", flush=True)
    setup_global_obstacles(int(args.num_obstacles))
    print("[INFO] Creating env...", flush=True)
    base_env = gym.make(args.task, cfg=env_cfg).unwrapped
    setattr(base_env, "_enable_tb_reward_terms", int(args.tb_interval) > 0)
    setattr(base_env, "_enable_tb_aux_terms", int(args.tb_interval) > 0)
    setattr(base_env, "_collision_print_enabled", False)
    if bool(args.headless) or int(args.num_envs) > 1:
        setattr(base_env, "_goal_vis_enabled", False)
    scale_robot_visual_only(num_envs=base_env.num_envs, visual_scale=(20.0, 20.0, 10.0))
    space = getattr(base_env, "single_observation_space", None)
    policy_space = space.spaces.get("policy", None) if isinstance(space, gym.spaces.Dict) else getattr(base_env, "observation_space", None)
    if policy_space is None:
        raise RuntimeError("policy observation space not found")
    obs_dim_raw = int(np.prod(policy_space.shape))
    state_dim, lidar_dim = get_state_lidar_dims(base_env, obs_dim_raw)
    lidar_grid_shape = infer_lidar_grid_shape(base_env, lidar_dim)
    model_cfg = resolve_model_cfg(
        feat_dim=int(args.feat_dim),
        lidar_grid_shape=lidar_grid_shape,
        model_cfg_path=args.model_cfg_path or None,
        model_cfg_json=args.model_cfg_json or None,
    )
    model_cfg_dict = model_cfg_to_dict(model_cfg)
    obs_dim, act_dim, obs_space, act_space = build_skrl_spaces(base_env, state_dim, lidar_dim)
    adapted_env = SkrlSpaceAdapter(base_env, obs_space=obs_space, act_space=act_space, state_dim=state_dim, lidar_dim=lidar_dim)
    env = wrap_env(adapted_env, wrapper="isaaclab")
    num_envs = int(getattr(env, "num_envs", args.num_envs))
    device = torch.device(getattr(env, "device", args.device))
    step_dt = get_env_step_dt(base_env)
    print(f"[INFO] lidar_grid_shape={lidar_grid_shape}", flush=True)
    models = {
        "policy": Policy(
            obs_space,
            act_space,
            device,
            state_dim,
            lidar_dim,
            args.feat_dim,
            lidar_grid_shape=lidar_grid_shape,
            model_cfg=model_cfg,
        ),
        "value": Value(
            obs_space,
            act_space,
            device,
            state_dim,
            lidar_dim,
            args.feat_dim,
            lidar_grid_shape=lidar_grid_shape,
            model_cfg=model_cfg,
        ),
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
    if bool(args.use_kl_adaptive_lr):
        cfg["learning_rate_scheduler"] = KLAdaptiveLR
        cfg["learning_rate_scheduler_kwargs"] = {
            "kl_threshold": float(args.kl_adaptive_lr_threshold),
            "min_lr": float(args.kl_adaptive_min_lr),
            "max_lr": float(args.kl_adaptive_max_lr),
            "kl_factor": float(args.kl_adaptive_kl_factor),
            "lr_factor": float(args.kl_adaptive_lr_factor),
        }
    else:
        cfg["learning_rate_scheduler"] = None
        cfg["learning_rate_scheduler_kwargs"] = {}
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    log_root = project_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f") + "_PPO"
    exp_dir = log_root / run_name
    tb_dir = exp_dir / args.extra_tb_subdir
    tb_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "config" / "config.txt"
    cfg["experiment"]["directory"] = str(log_root)
    cfg["experiment"]["experiment_name"] = run_name
    cfg["experiment"]["write_interval"] = int(args.tb_interval)
    cfg["experiment"]["checkpoint_interval"] = int(args.checkpoint_interval)
    write_run_config_snapshot(
        config_path=config_path,
        args_ns=args,
        env_cfg=env_cfg,
        runtime_meta={
            "task": str(args.task),
            "num_envs": int(num_envs),
            "obs_dim": int(obs_dim),
            "state_dim": int(state_dim),
            "lidar_dim": int(lidar_dim),
            "base_learning_rate": float(args.learning_rate),
            "learning_rate_scheduler": "KLAdaptiveLR" if bool(args.use_kl_adaptive_lr) else None,
            "learning_rate_scheduler_kwargs": dict(cfg["learning_rate_scheduler_kwargs"]),
            "feat_dim": int(model_cfg.feature_dim),
            "lidar_grid_shape": list(model_cfg.lidar_encoder.grid_shape) if model_cfg.lidar_encoder.grid_shape is not None else None,
            "lidar_encoder": str(model_cfg.lidar_encoder.type),
            "act_dim": int(act_dim),
            "step_dt": float(step_dt),
            "run_name": str(run_name),
        },
        model_cfg=model_cfg_dict,
    )
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("run/args", str(vars(args)), 0)
    writer.add_text("run/model_cfg", json_dumps_pretty(model_cfg_dict), 0)
    writer.add_text(
        "run/dims",
        f"obs={obs_dim}, state={state_dim}, lidar={lidar_dim}, act={act_dim}, grid={model_cfg.lidar_encoder.grid_shape}, encoder={model_cfg.lidar_encoder.type}",
        0,
    )
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
    states = sanitize_states(ensure_obs_shape(extract_policy_obs(raw_obs), num_envs, obs_dim), state_dim=state_dim, lidar_dim=lidar_dim)
    finite_check_interval = int(args.finite_check_interval)
    keep_nan_snapshot = finite_check_interval > 0
    last_good_snapshot = snapshot_models(models) if keep_nan_snapshot else {}
    enable_scalar_logging = int(args.tb_interval) > 0
    enable_hist_logging = int(args.dist_interval) > 0
    reward_weights = extract_reward_weights(base_env)
    reward_window = RewardBreakdownAccumulator() if enable_scalar_logging else None
    aux_window = TensorDictStats() if enable_scalar_logging else None
    termination_ratio_window = InfoTerminationRatioAccumulator()
    hist_logger = RollingHistogramLogger(
        obs_names=build_state_names(state_dim),
        action_names=build_action_names(act_dim),
        window=int(args.dist_window),
        max_samples=int(args.dist_max_samples),
    ) if enable_hist_logging else None
    latest_curriculum_log: Dict[str, float] = {}
    pbar = tqdm(
        range(int(args.timesteps)),
        ncols=110,
        disable=int(args.pbar_interval) <= 0,
        miniters=max(int(args.pbar_interval), 1),
        mininterval=1.0,
    )
    try:
        for t in pbar:
            global_step = t + 1
            agent.pre_interaction(timestep=t, timesteps=int(args.timesteps))
            with torch.no_grad():
                act_output = agent.act(states, timestep=t, timesteps=int(args.timesteps))
            actions = ensure_action_shape(extract_actions(act_output, act_dim), num_envs, act_dim).float()
            if not torch.isfinite(actions).all(): raise RuntimeError(f"Non-finite actions detected at t={t}")
            actions = sanitize_actions(actions)
            rollout_boundary = (global_step % int(args.rollouts) == 0)
            if keep_nan_snapshot and rollout_boundary:
                last_good_snapshot = snapshot_models(models)
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            next_states = sanitize_states(ensure_obs_shape(extract_policy_obs(next_obs), num_envs, obs_dim), state_dim=state_dim, lidar_dim=lidar_dim)
            rewards = ensure_vec_shape(torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0), num_envs, "rewards")
            terminated = ensure_vec_shape(torch.nan_to_num(terminated.float(), nan=0.0, posinf=0.0, neginf=0.0), num_envs, "terminated").bool()
            truncated = ensure_vec_shape(torch.nan_to_num(truncated.float(), nan=0.0, posinf=0.0, neginf=0.0), num_envs, "truncated").bool()
            train_rewards = scale_rewards(rewards, scale=args.reward_scale, clip=args.reward_clip)
            if reward_window is not None:
                raw_terms, weighted_terms, scaled_terms = build_reward_term_views(base_env, reward_weights=reward_weights, reward_scale=float(args.reward_scale), reward_clip=float(args.reward_clip))
                reward_window.update(raw_terms=raw_terms, weighted_terms=weighted_terms, scaled_terms=scaled_terms)
                if aux_window is not None:
                    aux_window.update(extract_tb_aux_terms(base_env))
                clear_tb_caches(base_env)
            done_count = int((terminated | truncated).sum().item())
            termination_ratio_window.update(
                build_termination_ratio_dict(
                    build_primary_termination_count_dict(base_env, terminated=terminated, truncated=truncated),
                    done_count=done_count,
                )
            )
            if done_count > 0:
                if len(curriculum_log_dict := extract_prefixed_log_scalars(infos, "Curriculum/")) > 0:
                    latest_curriculum_log.update(curriculum_log_dict)
            with torch.no_grad():
                agent.record_transition(states=states, actions=actions, rewards=train_rewards, next_states=next_states, terminated=terminated, truncated=truncated, infos=infos if args.keep_infos else {}, timestep=t, timesteps=int(args.timesteps))
            agent.post_interaction(timestep=t, timesteps=int(args.timesteps))
            if rollout_boundary: loss_mirror.flush(writer, global_step)
            if keep_nan_snapshot and rollout_boundary and finite_check_interval > 0 and ((global_step // int(args.rollouts)) % finite_check_interval == 0):
                if not models_are_finite(models):
                    debug_dir = exp_dir / "debug"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(last_good_snapshot, debug_dir / f"last_good_before_nan_t{t}.pt")
                    raise RuntimeError(f"Non-finite model parameters detected at t={t}")
            if not args.headless and int(args.render_interval) > 0 and (global_step % int(args.render_interval) == 0):
                try: env.render()
                except Exception: pass
            should_log_dist = enable_hist_logging and ((global_step % int(args.dist_interval) == 0) or (global_step == int(args.timesteps)))
            should_log_scalars = enable_scalar_logging and ((global_step % int(args.tb_interval) == 0) or (global_step == int(args.timesteps)))
            if should_log_dist and hist_logger is not None:
                hist_logger.update(states=states, actions=actions, state_dim=state_dim)
                hist_logger.flush(writer, global_step)
                if not should_log_scalars: writer.flush()
            if should_log_scalars:
                for key, value in sorted(latest_curriculum_log.items()): writer.add_scalar(sanitize_tb_tag(key), value, global_step)
                if reward_window is not None:
                    reward_window.flush(writer, global_step)
                if aux_window is not None:
                    aux_window.flush(writer, "Aux", global_step)
                termination_ratio_window.flush(writer, global_step)
                writer.flush()
                if reward_window is not None:
                    reward_window.reset()
                if aux_window is not None:
                    aux_window.reset()
                termination_ratio_window.reset()
            if int(args.grad_hist_interval) > 0 and rollout_boundary and (global_step // int(args.rollouts)) % int(args.grad_hist_interval) == 0:
                log_gradients(writer, models, global_step, int(args.grad_hist_samples))
                writer.flush()
            if int(args.checkpoint_interval) > 0 and (global_step % int(args.checkpoint_interval) == 0):
                ckpt_dir = exp_dir / "manual_checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({name: model.state_dict() for name, model in models.items()}, ckpt_dir / f"models_t{global_step}.pt")
            if int(args.cuda_clean_interval) > 0 and (global_step % int(args.cuda_clean_interval) == 0):
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            if int(args.pbar_interval) > 0 and (global_step % int(args.pbar_interval) == 0):
                pbar.set_description(f"t={t} envR={rewards.mean().item():+.3f} trainR={train_rewards.mean().item():+.3f} done={done_count}")
            states = next_states
    except KeyboardInterrupt:
        print("\n[WARN] KeyboardInterrupt: stopping training early", flush=True)
    finally:
        print("[INFO] Training loop exited", flush=True)
        try: writer.close()
        except: pass
        try: env.close()
        except: pass


if __name__ == "__main__":
    try: main()
    except Exception:
        print("\n[ERROR] Unhandled exception:\n", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
        print("[INFO] Simulation app closed", flush=True)
