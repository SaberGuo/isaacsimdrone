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
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument(
    "--num_obstacles",
    type=int,
    default=100,
    help="Maximum number of per-env obstacles to spawn.",
)
parser.add_argument("--timesteps", type=int, default=2_000_000)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--state_dim", type=int, default=17)
parser.add_argument("--lidar_dim", type=int, default=432)
parser.add_argument("--feat_dim", type=int, default=256)

parser.add_argument("--history_len", type=int, default=4,
                    help="Number of history frames K for causal transformer encoder")
parser.add_argument("--d_model", type=int, default=256,
                    help="Transformer model dimension")
parser.add_argument("--num_attn_heads", type=int, default=8,
                    help="Number of attention heads in transformer")
parser.add_argument("--dim_feedforward", type=int, default=512,
                    help="Feedforward dimension in transformer layers")
parser.add_argument("--num_transformer_layers", type=int, default=2,
                    help="Number of transformer encoder layers")

parser.add_argument("--rollouts", type=int, default=256)
parser.add_argument("--learning_epochs", type=int, default=8)
parser.add_argument("--mini_batches", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--_lambda", type=float, default=0.97)
parser.add_argument("--discount_factor", type=float, default=0.99)

parser.add_argument("--ratio_clip", type=float, default=0.2)
parser.add_argument("--value_clip", type=float, default=0.2)
parser.add_argument("--value_loss_scale", type=float, default=0.5)
parser.add_argument("--grad_norm_clip", type=float, default=1.0)
parser.add_argument("--entropy_coef", type=float, default=1e-2)
parser.add_argument("--kl_threshold", type=float, default=0.01)
parser.add_argument("--clip_predicted_values", action="store_true")
parser.add_argument("--no_clip_predicted_values", dest="clip_predicted_values", action="store_false")
parser.set_defaults(clip_predicted_values=True)

parser.add_argument("--reward_scale", type=float, default=0.1)
parser.add_argument("--reward_clip", type=float, default=100.0)

parser.add_argument("--tb_interval", type=int, default=500)
parser.add_argument("--dist_interval", type=int, default=500)
parser.add_argument("--dist_window", type=int, default=10)
parser.add_argument("--dist_max_samples", type=int, default=2048)
parser.add_argument("--checkpoint_interval", type=int, default=50000)
parser.add_argument("--cuda_clean_interval", type=int, default=2000)
parser.add_argument("--extra_tb_subdir", type=str, default="extra_tb")
parser.add_argument("--tb_suffix", type=str, default="",
                    help="Custom suffix appended to TensorBoard run name (e.g. 'transformer_K4')")

parser.add_argument("--keep_infos", action="store_true", default=False)
parser.add_argument("--grad_hist_interval", type=int, default=50)
parser.add_argument("--grad_hist_samples", type=int, default=65536)
parser.add_argument("--debug_act", action="store_true", default=False)

# Dijkstra navigation reward parameters
parser.add_argument("--use_dijkstra", action="store_true", default=True,
                    help="Enable Dijkstra-based navigation reward")
parser.add_argument("--no_dijkstra", dest="use_dijkstra", action="store_false",
                    help="Disable Dijkstra-based navigation reward")
parser.add_argument("--dijkstra_weight", type=float, default=20.0,
                    help="Weight for Dijkstra progress reward")
parser.add_argument("--dijkstra_grid_size", type=int, default=80,
                    help="Dijkstra grid resolution (grid_size x grid_size)")
parser.add_argument("--dijkstra_cell_size", type=float, default=2.0,
                    help="Grid cell size in meters")
parser.add_argument("--dijkstra_update_interval", type=int, default=10,
                    help="Recompute distance field every N steps")

# APF (Artificial Potential Field) reward parameters
parser.add_argument("--use_apf", action="store_true", default=False,
                    help="Enable APF potential field reward shaping")
parser.add_argument("--no_apf", dest="use_apf", action="store_false",
                    help="Disable APF potential field reward shaping")
parser.add_argument("--apf_att_weight", type=float, default=15.0,
                    help="Weight for APF attractive reward (positive)")
parser.add_argument("--apf_rep_weight", type=float, default=-150.0,
                    help="Weight for APF repulsive penalty (negative)")
parser.add_argument("--apf_d0", type=float, default=5.0,
                    help="APF repulsive influence radius in meters")
parser.add_argument("--apf_k_att", type=float, default=1.0,
                    help="APF attractive gain k_att")
parser.add_argument("--apf_k_rep", type=float, default=1.0,
                    help="APF repulsive gain k_rep")

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

from omniperception_isaacdrone.envs.test6_env import WallSpawner, setup_global_obstacles

STATE_OBS_NAMES_17 = [
    "root_pos_z", "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z",
    "root_lin_vel_x", "root_lin_vel_y", "root_lin_vel_z",
    "root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z",
    "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
    "goal_delta_x", "goal_delta_y", "goal_delta_z",
]
ACTION_NAMES_4 = ["vx_cmd", "vy_cmd", "vz_cmd", "yaw_rate_cmd"]

DEBUG_PRINT = False

from pxr import Usd, UsdGeom, UsdPhysics, Gf
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.bounds as bounds_utils


def scale_robot_visual_only(num_envs: int, visual_scale=(20.0, 20.0, 10.0)) -> None:
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

def format_array_preview(x: np.ndarray, max_items: int = 16) -> str:
    x = np.asarray(x).reshape(-1)
    if x.size <= max_items:
        return np.array2string(x, precision=3, separator=", ")
    return f"{np.array2string(x[:max_items], precision=3, separator=', ')} ..."

def print_space_bounds(name: str, space: gym.Space) -> None:
    print(f"\n[SPACE] {name}: type={type(space).__name__}", flush=True)

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
    if lidar_dim <= 0:
        return torch.clamp(states[:, :state_dim], -1.0, 1.0)
    single_dim = state_dim + lidar_dim
    K = max(1, states.shape[1] // single_dim)  # auto-detect number of history frames
    chunks = []
    for k in range(K):
        base = k * single_dim
        chunks.append(torch.cat([
            torch.clamp(states[:, base:base + state_dim], -1.0, 1.0),
            torch.clamp(states[:, base + state_dim:base + single_dim], 0.0, 1.0),
        ], dim=-1))
    return torch.cat(chunks, dim=-1)

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

def build_skrl_spaces(base_env: Any, state_dim: int, lidar_dim: int, K: int = 1) -> tuple[int, int, gym.spaces.Dict, Box]:
    num_envs = int(getattr(base_env, "num_envs", 1))
    act_space = getattr(base_env, "single_action_space", getattr(base_env, "action_space", None))
    act_dim = infer_single_dim_from_box(act_space, num_envs) if isinstance(act_space, gym.spaces.Box) else 4
    single_dim = state_dim + lidar_dim
    obs_dim = K * single_dim  # e.g. 4 * 449 = 1796
    obs_low  = np.full((obs_dim,), -1.0, dtype=np.float32)
    obs_high = np.full((obs_dim,),  1.0, dtype=np.float32)
    if lidar_dim > 0:
        for k in range(K):
            base = k * single_dim
            obs_low[base + state_dim : base + single_dim] = 0.0  # lidar ∈ [0, 1]
    return obs_dim, act_dim, gym.spaces.Dict({"policy": Box(low=obs_low, high=obs_high, dtype=np.float32)}), Box(low=-np.ones((act_dim,), dtype=np.float32), high=np.ones((act_dim,), dtype=np.float32), dtype=np.float32)

class SkrlSpaceAdapter(gym.Wrapper):
    def __init__(self, env: gym.Env, obs_space: gym.spaces.Dict, act_space: Box,
                 state_dim: int, lidar_dim: int, K: int = 1):
        super().__init__(env)
        self.state_dim  = int(state_dim)
        self.lidar_dim  = int(lidar_dim)
        self.K          = int(K)
        self.single_dim = self.state_dim + self.lidar_dim   # 449 per frame
        self.obs_dim    = self.K * self.single_dim           # 1796 total
        self.observation_space = self.single_observation_space = obs_space
        self.action_space = self.single_action_space = act_space
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.device = getattr(env, "device", None)
        # History buffer: (num_envs, K, single_dim); None until first reset()
        self._history: torch.Tensor | None = None

    def _sanitize_single_frame(self, raw: torch.Tensor) -> torch.Tensor:
        """Sanitize a single-frame obs tensor of shape (N, single_dim)."""
        raw = torch.nan_to_num(raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
        state = torch.clamp(raw[:, :self.state_dim], -1.0, 1.0)
        if self.lidar_dim > 0:
            lidar = torch.clamp(raw[:, self.state_dim:self.single_dim], 0.0, 1.0)
            return torch.cat([state, lidar], dim=-1)
        return state

    def _init_history(self, first_obs: torch.Tensor) -> None:
        """Fill all K history slots with the first observation (no zero-padding artifacts)."""
        # first_obs: (N, single_dim)
        self._history = (
            first_obs
            .unsqueeze(1)                   # (N, 1, single_dim)
            .expand(-1, self.K, -1)         # (N, K, single_dim)
            .clone()
        )

    def _update_history(self, next_obs: torch.Tensor, done_mask: torch.Tensor) -> None:
        """Shift history and append new frame; reset done envs BEFORE shifting.

        In IsaacLab, when terminated[i]=True the returned obs[i] is already the
        initial obs of the NEW episode. We therefore fill done envs' entire history
        with that new initial obs before shifting all envs.
        """
        # Step 1: reset history for finished episodes
        if done_mask.any():
            reset_obs = next_obs[done_mask]   # (D, single_dim)
            self._history[done_mask] = (
                reset_obs
                .unsqueeze(1)               # (D, 1, single_dim)
                .expand(-1, self.K, -1)     # (D, K, single_dim)
                .clone()
            )
        # Step 2: shift left by 1 and append new frame for all envs
        self._history = torch.cat(
            [self._history[:, 1:, :],       # (N, K-1, single_dim)
             next_obs.unsqueeze(1)],        # (N,   1, single_dim)
            dim=1                           # → (N, K, single_dim)
        )

    def _build_stacked_obs(self) -> dict[str, torch.Tensor]:
        """Flatten (N, K, single_dim) → (N, K*single_dim)."""
        return {"policy": self._history.reshape(self.num_envs, self.obs_dim)}

    def _get_raw_single_frame(self, raw_obs: Any) -> torch.Tensor:
        return ensure_obs_shape(extract_policy_obs(raw_obs), self.num_envs, self.single_dim)

    def reset(self, **kwargs):
        raw_obs, infos = self.env.reset(**kwargs)
        sanitized = self._sanitize_single_frame(self._get_raw_single_frame(raw_obs))
        self._init_history(sanitized)
        return self._build_stacked_obs(), infos

    def step(self, actions):
        if self._history is None:
            raise RuntimeError("SkrlSpaceAdapter.step() called before reset()")
        raw_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        sanitized = self._sanitize_single_frame(self._get_raw_single_frame(raw_obs))
        done_mask = (terminated | truncated).reshape(-1).bool()
        self._update_history(sanitized, done_mask)
        return self._build_stacked_obs(), rewards, terminated, truncated, infos

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

def extract_termination_count_dict(infos: Any) -> Dict[str, float]:
    return {k.split("/", 1)[1]: to_float(v) for k, v in extract_log_dict(infos).items() if k.startswith("Episode_Termination/")}

def extract_termination_ratio_dict(infos: Any, done_count: int) -> Dict[str, float]:
    if int(done_count) <= 0: return {}
    denom = max(float(done_count), 1.0)
    return {k: float(v) / denom for k, v in extract_termination_count_dict(infos).items()}

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

class TensorDictStats:
    def __init__(self): self.reset()
    def reset(self): self.window_steps, self.sum, self.min, self.max = 0, {}, {}, {}
    def update(self, values: Dict[str, torch.Tensor]):
        if len(values) == 0: return
        self.window_steps += 1
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.numel() == 0: continue
            v = torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            mean_v, min_v, max_v = float(v.mean().item()), float(v.min().item()), float(v.max().item())
            self.sum[name] = self.sum.get(name, 0.0) + mean_v
            self.min[name] = min_v if name not in self.min else min(self.min[name], min_v)
            self.max[name] = max_v if name not in self.max else max(self.max[name], max_v)
    def flush(self, writer: SummaryWriter, prefix: str, step: int):
        if self.window_steps <= 0: return
        for name in sorted(self.sum.keys()):
            tag = sanitize_tb_tag(name)
            writer.add_scalar(f"{prefix}/{tag}/mean", self.sum[name] / float(self.window_steps), step)
            writer.add_scalar(f"{prefix}/{tag}/min", self.min[name], step)
            writer.add_scalar(f"{prefix}/{tag}/max", self.max[name], step)

class RewardBreakdownAccumulator:
    def __init__(self): self.raw, self.weighted, self.scaled = TensorDictStats(), TensorDictStats(), TensorDictStats()
    def reset(self): self.raw.reset(); self.weighted.reset(); self.scaled.reset()
    def update(self, raw_terms, weighted_terms, scaled_terms): self.raw.update(raw_terms); self.weighted.update(weighted_terms); self.scaled.update(scaled_terms)
    def flush(self, writer, step): self.raw.flush(writer, "RewardRaw", step); self.weighted.flush(writer, "RewardWeighted", step); self.scaled.flush(writer, "RewardScaled", step)

class InfoTerminationRatioAccumulator:
    def __init__(self): self.reset()
    def reset(self): self.update_steps, self.sum_ratios, self.last_ratios = 0, {}, {}
    def update(self, infos, done_count):
        if int(done_count) <= 0 or len(ratio_dict := extract_termination_ratio_dict(infos, done_count)) == 0: return
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
    def __init__(self): self.reset()
    def reset(self): self.policy_losses, self.value_losses = [], []
    def bind(self, agent: PPO):
        def wrapped_track_data(tag, value):
            if "loss" in (low := str(tag).lower()) and "policy" in low: self.policy_losses.append(to_float(value))
            elif "loss" in low and "value" in low: self.value_losses.append(to_float(value))
        agent.track_data = wrapped_track_data
    def flush(self, writer, step):
        if len(self.policy_losses) > 0: writer.add_scalar("Loss/policy", float(np.mean(self.policy_losses)), step)
        if len(self.value_losses) > 0: writer.add_scalar("Loss/value", float(np.mean(self.value_losses)), step)
        self.reset()

def log_gradients(writer, models, step, max_samples):
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

class ModalTransformerEncoder(nn.Module):
    """Causal multi-modal transformer encoder for K-frame UAV observations.

    Paper: HuaJiarui UAVTT – adapted from SAC to PPO feature extractor.

    Token layout per frame k (3 tokens, interleaved):
        position 3k+0 : lidar  token  (432D → d_model)
        position 3k+1 : body   token  (14D  → d_model)
        position 3k+2 : goal   token  (3D   → d_model)

    Causal mask: token q (frame q//3) may attend to token k only if k//3 ≤ q//3.
    """

    def __init__(
        self,
        state_dim: int,
        lidar_dim: int,
        K: int = 4,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        num_layers: int = 2,
        d_feat: int = 256,
    ):
        super().__init__()
        self.state_dim  = int(state_dim)   # 17
        self.lidar_dim  = int(lidar_dim)   # 432
        self.K          = int(K)           # 4
        self.d_model    = int(d_model)
        self.T          = 3 * K            # 12 tokens total
        self.single_dim = state_dim + lidar_dim

        body_dim = state_dim - 3  # 14  (all state dims except last 3 = goal_delta)
        goal_dim = 3

        # Modality projections (shared across all frames)
        self.lidar_proj = nn.Linear(lidar_dim, d_model)
        self.body_proj  = nn.Linear(body_dim, d_model)
        self.goal_proj  = nn.Linear(goal_dim, d_model)

        # Learned positional embeddings
        self.frame_pe = nn.Embedding(K, d_model)   # which temporal frame
        self.token_pe = nn.Embedding(3, d_model)   # which modality (lidar/body/goal)

        # Pre-computed index tensors (registered as buffers → auto .to(device))
        token_pos = torch.arange(self.T)
        self.register_buffer("frame_ids",      token_pos // 3)  # [0,0,0,1,1,1,...,3,3,3]
        self.register_buffer("token_type_ids", token_pos % 3)   # [0,1,2,0,1,2,...,0,1,2]

        # Causal attention mask: (T, T) float, −∞ where key_frame > query_frame
        frame_idx   = token_pos // 3                               # (T,)
        blocked     = frame_idx.unsqueeze(0) > frame_idx.unsqueeze(1)  # (T,T) bool
        causal_mask = torch.zeros(self.T, self.T)
        causal_mask[blocked] = float("-inf")
        self.register_buffer("causal_mask", causal_mask)

        # Transformer encoder (Post-Norm = norm_first=False, no dropout for RL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # avoids warning with additive float mask
        )

        # Output head: global avg-pool → Linear → LayerNorm → LeakyReLU
        self.head = nn.Sequential(
            nn.Linear(d_model, d_feat),
            nn.LayerNorm(d_feat),
            nn.LeakyReLU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.lidar_proj, self.body_proj, self.goal_proj):
            nn.init.orthogonal_(proj.weight, gain=np.sqrt(2.0))
            nn.init.constant_(proj.bias, 0.0)
        nn.init.normal_(self.frame_pe.weight, std=0.02)
        nn.init.normal_(self.token_pe.weight, std=0.02)
        nn.init.orthogonal_(self.head[0].weight, gain=np.sqrt(2.0))
        nn.init.constant_(self.head[0].bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, K * single_dim)  e.g. (B, 1796)
        Returns:
            (B, d_feat)  e.g. (B, 256)
        """
        B  = obs.shape[0]
        K  = self.K
        sd = self.state_dim   # 17
        ld = self.lidar_dim   # 432
        dm = self.d_model

        # ── 1. Reshape and split ────────────────────────────────────────────
        frames    = obs.view(B, K, self.single_dim)      # (B, 4, 449)
        lidar_all = frames[:, :, sd:]                    # (B, 4, 432)
        body_all  = frames[:, :, :sd - 3]               # (B, 4, 14)
        goal_all  = frames[:, :, sd - 3:sd]             # (B, 4, 3)

        # ── 2. Project modalities (batch K frames together) ─────────────────
        lidar_tok = self.lidar_proj(lidar_all.reshape(B * K, ld)).view(B, K, dm)
        body_tok  = self.body_proj(body_all.reshape(B * K, sd - 3)).view(B, K, dm)
        goal_tok  = self.goal_proj(goal_all.reshape(B * K, 3)).view(B, K, dm)

        # ── 3. Interleave: [lidar_0, body_0, goal_0, lidar_1, ...] ─────────
        tokens = torch.stack([lidar_tok, body_tok, goal_tok], dim=2)  # (B, K, 3, dm)
        tokens = tokens.reshape(B, self.T, dm)                         # (B, 12, 256)

        # ── 4. Add positional embeddings ────────────────────────────────────
        tokens = (
            tokens
            + self.frame_pe(self.frame_ids).unsqueeze(0)       # (1, 12, 256)
            + self.token_pe(self.token_type_ids).unsqueeze(0)  # (1, 12, 256)
        )

        # ── 5. Causal transformer encoder ───────────────────────────────────
        encoded = self.transformer(tokens, mask=self.causal_mask)  # (B, 12, 256)

        # ── 6. Global average pooling + output head ─────────────────────────
        return self.head(encoded.mean(dim=1))                      # (B, d_feat)


class StructuredFeatureExtractor(nn.Module):
    """Kept for backward compatibility / K=1 fallback."""
    def __init__(self, state_dim, lidar_dim, feat_dim=256):
        super().__init__()
        self.state_dim, self.lidar_dim = int(state_dim), int(lidar_dim)
        self.state_ln, self.state_net = nn.LayerNorm(self.state_dim), nn.Sequential(nn.Linear(self.state_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        if self.lidar_dim > 0:
            self.lidar_ln, self.lidar_net = nn.LayerNorm(self.lidar_dim), nn.Sequential(nn.Linear(self.lidar_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh())
            fuse_in = 128 + 256
        else:
            self.lidar_ln, self.lidar_net, fuse_in = nn.Identity(), None, 128
        self.fuse_net = nn.Sequential(nn.Linear(fuse_in, feat_dim), nn.Tanh())
        self.apply(init_hidden)
    def forward(self, obs):
        state = self.state_net(self.state_ln(torch.clamp(obs[:, :self.state_dim], -1.0, 1.0)))
        if self.lidar_dim <= 0: return self.fuse_net(state)
        return self.fuse_net(torch.cat([state, self.lidar_net(self.lidar_ln(torch.clamp(obs[:, self.state_dim:self.state_dim + self.lidar_dim], 0.0, 1.0) * 2.0 - 1.0))], dim=-1))

def gaussian_mixin_kwargs():
    kwargs = {"clip_actions": True}
    if "clip_mean_actions" in inspect.signature(GaussianMixin.__init__).parameters: kwargs["clip_mean_actions"] = True
    return kwargs

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 state_dim, lidar_dim, feat_dim=256,
                 K=4, d_model=256, num_heads=8, dim_feedforward=512, num_layers=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **gaussian_mixin_kwargs())
        expected_obs = K * (state_dim + lidar_dim)
        if self.num_observations != expected_obs:
            raise RuntimeError(
                f"Policy obs dim mismatch: got {self.num_observations}, "
                f"expected {expected_obs} (K={K} × {state_dim + lidar_dim})"
            )
        self.fe = ModalTransformerEncoder(
            state_dim=state_dim, lidar_dim=lidar_dim, K=K,
            d_model=d_model, num_heads=num_heads,
            dim_feedforward=dim_feedforward, num_layers=num_layers,
            d_feat=feat_dim,
        )
        self.mean = nn.Linear(feat_dim, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -1.0))
        init_policy_head(self.mean)

    def compute(self, inputs, role):
        mean = torch.tanh(self.mean(self.fe(inputs["states"])))
        return mean, torch.clamp(self.log_std_parameter, min=-5.0, max=0.0).expand_as(mean), {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 state_dim, lidar_dim, feat_dim=256,
                 K=4, d_model=256, num_heads=8, dim_feedforward=512, num_layers=2):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        expected_obs = K * (state_dim + lidar_dim)
        if self.num_observations != expected_obs:
            raise RuntimeError(
                f"Value obs dim mismatch: got {self.num_observations}, "
                f"expected {expected_obs} (K={K} × {state_dim + lidar_dim})"
            )
        self.fe = ModalTransformerEncoder(
            state_dim=state_dim, lidar_dim=lidar_dim, K=K,
            d_model=d_model, num_heads=num_heads,
            dim_feedforward=dim_feedforward, num_layers=num_layers,
            d_feat=feat_dim,
        )
        self.value = nn.Linear(feat_dim, 1)
        init_value_head(self.value)

    def compute(self, inputs, role):
        return self.value(self.fe(inputs["states"])), {}

def build_state_names(state_dim: int) -> List[str]:
    return list(STATE_OBS_NAMES_17) if int(state_dim) == len(STATE_OBS_NAMES_17) else [f"state_{i}" for i in range(int(state_dim))]

def build_action_names(act_dim: int) -> List[str]:
    return list(ACTION_NAMES_4) if int(act_dim) == len(ACTION_NAMES_4) else [f"action_{i}" for i in range(int(act_dim))]


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

    # Configure Dijkstra navigation reward
    if args.use_dijkstra:
        print(f"[INFO] Dijkstra reward enabled: weight={args.dijkstra_weight}, "
              f"grid={args.dijkstra_grid_size}x{args.dijkstra_grid_size}, "
              f"cell={args.dijkstra_cell_size}m, update_interval={args.dijkstra_update_interval}")
        env_cfg.rewards.dijkstra_progress.weight = args.dijkstra_weight
        env_cfg.rewards.dijkstra_progress.params["grid_size"] = args.dijkstra_grid_size
        env_cfg.rewards.dijkstra_progress.params["cell_size"] = args.dijkstra_cell_size
        env_cfg.rewards.dijkstra_progress.params["update_interval"] = args.dijkstra_update_interval
    else:
        print("[INFO] Dijkstra reward disabled")
        env_cfg.rewards.dijkstra_progress.weight = 0.0

    # Configure APF potential field reward
    if args.use_apf:
        print(f"[INFO] APF reward enabled: att_weight={args.apf_att_weight}, "
              f"rep_weight={args.apf_rep_weight}, d0={args.apf_d0}m, "
              f"k_att={args.apf_k_att}, k_rep={args.apf_k_rep}")
        env_cfg.rewards.apf_attractive.weight = args.apf_att_weight
        env_cfg.rewards.apf_attractive.params["k_att"] = args.apf_k_att
        env_cfg.rewards.apf_repulsive.weight = args.apf_rep_weight
        env_cfg.rewards.apf_repulsive.params["d0"] = args.apf_d0
        env_cfg.rewards.apf_repulsive.params["k_rep"] = args.apf_k_rep
    else:
        print("[INFO] APF reward disabled")

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

    # 在 gym.make 之前，先将障碍物模板写入全局路径
    print("[INFO] Setting up global obstacles template...", flush=True)
    setup_global_obstacles(int(args.num_obstacles))

    print("[INFO] Creating env...", flush=True)
    base_env = gym.make(args.task, cfg=env_cfg).unwrapped
    scale_robot_visual_only(
        num_envs=base_env.num_envs,
        visual_scale=(20.0, 20.0, 10.0),
    )

    space = getattr(base_env, "single_observation_space", None)
    policy_space = space.spaces.get("policy", None) if isinstance(space, gym.spaces.Dict) else getattr(base_env, "observation_space", None)
    if policy_space is None:
        raise RuntimeError("policy observation space not found")

    obs_dim_raw = int(np.prod(policy_space.shape))
    state_dim, lidar_dim = get_state_lidar_dims(base_env, obs_dim_raw)
    K = int(args.history_len)
    obs_dim, act_dim, obs_space, act_space = build_skrl_spaces(base_env, state_dim, lidar_dim, K=K)

    adapted_env = SkrlSpaceAdapter(
        base_env, obs_space=obs_space, act_space=act_space,
        state_dim=state_dim, lidar_dim=lidar_dim, K=K,
    )
    env = wrap_env(adapted_env, wrapper="isaaclab")

    num_envs = int(getattr(env, "num_envs", args.num_envs))
    device = torch.device(getattr(env, "device", args.device))
    step_dt = get_env_step_dt(base_env)

    models = {
        "policy": Policy(
            obs_space, act_space, device,
            state_dim=state_dim, lidar_dim=lidar_dim, feat_dim=args.feat_dim,
            K=K, d_model=args.d_model, num_heads=args.num_attn_heads,
            dim_feedforward=args.dim_feedforward, num_layers=args.num_transformer_layers,
        ),
        "value": Value(
            obs_space, act_space, device,
            state_dim=state_dim, lidar_dim=lidar_dim, feat_dim=args.feat_dim,
            K=K, d_model=args.d_model, num_heads=args.num_attn_heads,
            dim_feedforward=args.dim_feedforward, num_layers=args.num_transformer_layers,
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

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    log_root = project_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.tb_suffix}" if args.tb_suffix else ""
    run_name = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f") + "_PPO" + suffix
    exp_dir = log_root / run_name
    tb_dir = exp_dir / args.extra_tb_subdir
    tb_dir.mkdir(parents=True, exist_ok=True)

    cfg["experiment"]["directory"] = str(log_root)
    cfg["experiment"]["experiment_name"] = run_name
    cfg["experiment"]["write_interval"] = int(args.tb_interval)
    cfg["experiment"]["checkpoint_interval"] = int(args.checkpoint_interval)

    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("run/args", str(vars(args)), 0)
    writer.add_text("run/dims", f"obs={obs_dim}, K={K}, state={state_dim}, lidar={lidar_dim}, act={act_dim}, d_model={args.d_model}", 0)
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
    last_good_snapshot = snapshot_models(models)

    reward_weights = extract_reward_weights(base_env)
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
            if not torch.isfinite(actions).all(): raise RuntimeError(f"Non-finite actions detected at t={t}")
            actions = sanitize_actions(actions)

            rollout_boundary = (global_step % int(args.rollouts) == 0)
            if rollout_boundary: last_good_snapshot = snapshot_models(models)

            next_obs, rewards, terminated, truncated, infos = env.step(actions)

            next_states = sanitize_states(ensure_obs_shape(extract_policy_obs(next_obs), num_envs, obs_dim), state_dim=state_dim, lidar_dim=lidar_dim)
            rewards = ensure_vec_shape(torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0), num_envs, "rewards")
            terminated = ensure_vec_shape(torch.nan_to_num(terminated.float(), nan=0.0, posinf=0.0, neginf=0.0), num_envs, "terminated").bool()
            truncated = ensure_vec_shape(torch.nan_to_num(truncated.float(), nan=0.0, posinf=0.0, neginf=0.0), num_envs, "truncated").bool()
            train_rewards = scale_rewards(rewards, scale=args.reward_scale, clip=args.reward_clip)

            raw_terms, weighted_terms, scaled_terms = build_reward_term_views(base_env, reward_weights=reward_weights, reward_scale=float(args.reward_scale), reward_clip=float(args.reward_clip))
            reward_window.update(raw_terms=raw_terms, weighted_terms=weighted_terms, scaled_terms=scaled_terms)
            clear_tb_caches(base_env)

            done_count = int((terminated | truncated).sum().item())
            termination_ratio_window.update(infos, done_count=done_count)

            if done_count > 0:
                if len(curriculum_log_dict := extract_prefixed_log_scalars(infos, "Curriculum/")) > 0:
                    latest_curriculum_log.update(curriculum_log_dict)

            with torch.no_grad():
                agent.record_transition(states=states, actions=actions, rewards=train_rewards, next_states=next_states, terminated=terminated, truncated=truncated, infos=infos if args.keep_infos else {}, timestep=t, timesteps=int(args.timesteps))

            agent.post_interaction(timestep=t, timesteps=int(args.timesteps))
            if rollout_boundary: loss_mirror.flush(writer, global_step)

            if rollout_boundary and not models_are_finite(models):
                debug_dir = exp_dir / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                torch.save(last_good_snapshot, debug_dir / f"last_good_before_nan_t{t}.pt")
                raise RuntimeError(f"Non-finite model parameters detected at t={t}")

            if not args.headless:
                try: env.render()
                except Exception: pass

            should_log_dist = int(args.dist_interval) > 0 and ((global_step % int(args.dist_interval) == 0) or (global_step == int(args.timesteps)))
            should_log_scalars = int(args.tb_interval) > 0 and ((global_step % int(args.tb_interval) == 0) or (global_step == int(args.timesteps)))

            if should_log_dist:
                hist_logger.update(states=states, actions=actions, state_dim=state_dim)
                hist_logger.flush(writer, global_step)
                if not should_log_scalars: writer.flush()

            if should_log_scalars:
                for key, value in sorted(latest_curriculum_log.items()): writer.add_scalar(sanitize_tb_tag(key), value, global_step)
                reward_window.flush(writer, global_step)
                termination_ratio_window.flush(writer, global_step)
                writer.flush()
                reward_window.reset()
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
