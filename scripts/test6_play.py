from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser("Play trained skrl PPO policy for IsaacLab drone lidar task")
parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-Lidar-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_obstacles", type=int, default=100)  # 修改为默认 100，与训练脚本对齐
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--state_dim", type=int, default=17)       # 修改为默认 17 维，与训练脚本对齐
parser.add_argument("--lidar_dim", type=int, default=432)
parser.add_argument("--feat_dim", type=int, default=256)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="",
    help="Checkpoint path. If empty, auto search the latest under logs/",
)
parser.add_argument(
    "--use_stochastic_policy",
    action="store_true",
    default=False,
    help="Use Gaussian sampled action instead of deterministic mean action.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=5000,
    help="Maximum play steps. <= 0 means loop until app closes.",
)
parser.add_argument(
    "--reset_on_done",
    action="store_true",
    default=True,
    help="Reset env when terminated/truncated.",
)
parser.add_argument(
    "--no_reset_on_done",
    dest="reset_on_done",
    action="store_false",
)

parser.add_argument("--print_every", type=int, default=50)
parser.add_argument("--show_obs_stats", action="store_true", default=False)

# Transformer architecture parameters (must match training script)
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

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Launch Isaac Sim first
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Runtime imports
# -----------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import omniperception_isaacdrone.tasks.test6_registry as _test6_registry  # noqa: F401
from gymnasium.spaces import Box
from isaaclab_tasks.utils import parse_env_cfg
import isaacsim.core.utils.prims as prim_utils
from pxr import UsdGeom, Gf

# 替换旧的 ObstacleSpawner，使用统一的全局障碍物生成函数
from omniperception_isaacdrone.envs.test6_env import WallSpawner, setup_global_obstacles


# -----------------------------------------------------------------------------
# Names
# -----------------------------------------------------------------------------
STATE_OBS_NAMES_17 = [
    "root_pos_z", "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z",
    "root_lin_vel_x", "root_lin_vel_y", "root_lin_vel_z",
    "root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z",
    "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
    "goal_delta_x", "goal_delta_y", "goal_delta_z",
]

ACTION_NAMES_4 = ["vx_cmd", "vy_cmd", "vz_cmd", "yaw_rate_cmd"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def scale_robot_visual_only(num_envs: int, visual_scale=(20.0, 20.0, 10.0)) -> None:
    """Scale only the visual subtree of the drone, without touching physics/collision."""
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
        return

    print(f"[SPACE] {name} = {space}", flush=True)


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


def to_float(x: Any) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, torch.Tensor):
        y = x.detach().float()
        if y.numel() == 0:
            return 0.0
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return float(y.mean().item())
    try:
        return float(x)
    except Exception:
        return 0.0


def infer_single_dim_from_box(space: Box) -> int:
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


def build_spaces(base_env: Any, state_dim: int, lidar_dim: int, K: int = 1) -> tuple[int, int, gym.spaces.Dict, Box]:
    act_space = getattr(base_env, "single_action_space", None)
    if not isinstance(act_space, gym.spaces.Box):
        act_space = getattr(base_env, "action_space", None)

    if not isinstance(act_space, gym.spaces.Box):
        raise RuntimeError("Action space is not a gym.spaces.Box")

    act_dim = infer_single_dim_from_box(act_space)
    single_dim = state_dim + lidar_dim
    obs_dim = K * single_dim

    obs_low = -np.ones((obs_dim,), dtype=np.float32)
    obs_high = np.ones((obs_dim,), dtype=np.float32)
    if lidar_dim > 0:
        for k in range(K):
            base = k * single_dim
            obs_low[base + state_dim:base + single_dim] = 0.0

    obs_box = Box(low=obs_low, high=obs_high, dtype=np.float32)
    act_box = Box(
        low=-np.ones((act_dim,), dtype=np.float32),
        high=np.ones((act_dim,), dtype=np.float32),
        dtype=np.float32,
    )
    obs_space = gym.spaces.Dict({"policy": obs_box})
    return obs_dim, act_dim, obs_space, act_box


class PlaySpaceAdapter(gym.Wrapper):
    """Expose stable single-env semantic spaces and sanitized observations.
    Supports K-frame history for transformer-based policies.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_space: gym.spaces.Dict,
        act_space: Box,
        state_dim: int,
        lidar_dim: int,
        K: int = 1,
    ):
        super().__init__(env)
        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)
        self.K = int(K)
        self.single_dim = state_dim + lidar_dim
        self.obs_dim = self.K * self.single_dim

        self.observation_space = obs_space
        self.single_observation_space = obs_space
        self.action_space = act_space
        self.single_action_space = act_space

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
        x = extract_policy_obs(raw_obs)
        x = ensure_obs_shape(x, self.num_envs, self.single_dim)
        return self._sanitize_single_frame(x)

    def reset(self, **kwargs):
        raw_obs, infos = self.env.reset(**kwargs)
        sanitized = self._get_raw_single_frame(raw_obs)
        if self.K > 1:
            self._init_history(sanitized)
            return self._build_stacked_obs(), infos
        else:
            return {"policy": sanitized}, infos

    def step(self, actions):
        raw_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        sanitized = self._get_raw_single_frame(raw_obs)
        if self.K > 1:
            if self._history is None:
                raise RuntimeError("PlaySpaceAdapter.step() called before reset()")
            done_mask = (terminated | truncated).reshape(-1).bool()
            self._update_history(sanitized, done_mask)
            return self._build_stacked_obs(), rewards, terminated, truncated, infos
        else:
            return {"policy": sanitized}, rewards, terminated, truncated, infos


def init_hidden(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def init_policy_head(m: nn.Linear) -> None:
    nn.init.orthogonal_(m.weight, gain=0.01)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.0)


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


class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, state_dim: int, lidar_dim: int,
                 feat_dim: int = 256, K: int = 1,
                 d_model: int = 256, num_heads: int = 8,
                 dim_feedforward: int = 512, num_layers: int = 2):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)
        self.K = int(K)
        self.single_dim = state_dim + lidar_dim
        self.expected_obs = K * self.single_dim

        # Use transformer for K > 1, legacy for K == 1
        if K > 1:
            self.fe = ModalTransformerEncoder(
                state_dim=state_dim, lidar_dim=lidar_dim, K=K,
                d_model=d_model, num_heads=num_heads,
                dim_feedforward=dim_feedforward, num_layers=num_layers,
                d_feat=feat_dim,
            )
        else:
            self.fe = StructuredFeatureExtractor(state_dim, lidar_dim, feat_dim)

        self.mean = nn.Linear(feat_dim, self.act_dim)
        self.log_std_parameter = nn.Parameter(torch.full((self.act_dim,), -1.0))
        init_policy_head(self.mean)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.fe(obs)
        mean = torch.tanh(self.mean(feat))
        log_std = torch.clamp(self.log_std_parameter, min=-5.0, max=0.0).expand_as(mean)
        return mean, log_std

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            action = mean
        else:
            std = torch.exp(log_std)
            dist = D.Normal(mean, std)
            action = dist.sample()
        return torch.clamp(action, -1.0, 1.0)


def build_state_names(state_dim: int) -> list[str]:
    if int(state_dim) == len(STATE_OBS_NAMES_17):
        return list(STATE_OBS_NAMES_17)
    return [f"state_{i}" for i in range(int(state_dim))]


def build_action_names(act_dim: int) -> list[str]:
    if int(act_dim) == len(ACTION_NAMES_4):
        return list(ACTION_NAMES_4)
    return [f"action_{i}" for i in range(int(act_dim))]


def print_obs_summary(obs: torch.Tensor, state_dim: int, lidar_dim: int, prefix: str = "[OBS]") -> None:
    obs0 = obs[0].detach().cpu()
    state = obs0[:state_dim]
    print(f"{prefix} state = {np.array2string(state.numpy(), precision=3, separator=', ')}", flush=True)
    if lidar_dim > 0:
        lidar = obs0[state_dim:state_dim + lidar_dim]
        nz = float((lidar > 1e-6).float().mean().item())
        print(
            f"{prefix} lidar: min={float(lidar.min().item()):.4f}, "
            f"max={float(lidar.max().item()):.4f}, "
            f"mean={float(lidar.mean().item()):.4f}, "
            f"nonzero_ratio={nz:.4f}",
            flush=True,
        )


def extract_model_state_dict(payload: Any, key_hint: str | None = None) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if key_hint is not None and key_hint in payload and isinstance(payload[key_hint], dict):
            return payload[key_hint]

        for k in ["policy", "policy_state_dict", "model", "state_dict"]:
            if k in payload and isinstance(payload[k], dict):
                return payload[k]

        if all(isinstance(k, str) for k in payload.keys()):
            if any(("weight" in k) or ("bias" in k) or ("log_std_parameter" in k) for k in payload.keys()):
                return payload

    raise RuntimeError("Could not extract policy state_dict from checkpoint payload")


def find_latest_checkpoint(log_root: Path) -> Path:
    candidates: list[Path] = []

    if not log_root.exists():
        raise FileNotFoundError(f"log root not found: {log_root}")

    for p in log_root.rglob("*.pt"):
        if "manual_checkpoints" in str(p):
            candidates.append(p)

    if len(candidates) == 0:
        for p in log_root.rglob("*.pth"):
            candidates.append(p)
    if len(candidates) == 0:
        for p in log_root.rglob("*.pt"):
            candidates.append(p)

    if len(candidates) == 0:
        raise FileNotFoundError(f"No checkpoint found under {log_root}")

    candidates = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def load_policy_checkpoint(policy: Policy, checkpoint_path: Path, device: torch.device) -> None:
    print(f"[INFO] Loading checkpoint: {checkpoint_path}", flush=True)
    payload = torch.load(checkpoint_path, map_location=device)

    try:
        state_dict = extract_model_state_dict(payload, key_hint="policy")
    except Exception:
        state_dict = extract_model_state_dict(payload, key_hint=None)

    missing, unexpected = policy.load_state_dict(state_dict, strict=False)

    print(f"[INFO] Policy checkpoint loaded", flush=True)
    if len(missing) > 0:
        print(f"[WARN] Missing keys: {missing}", flush=True)
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys: {unexpected}", flush=True)


def maybe_print_goal(base_env: Any, env_ids: list[int] | None = None) -> None:
    try:
        if not hasattr(base_env, "goal_pos_w"):
            return
        goal = base_env.goal_pos_w.detach().cpu().numpy()
        if env_ids is None:
            env_ids = [0]
        for eid in env_ids:
            if 0 <= int(eid) < goal.shape[0]:
                print(f"[INFO] goal_pos_w[{eid}]={goal[eid]}", flush=True)
    except Exception:
        pass


def main() -> None:
    print(
        f"[INFO] task={args.task}, num_envs={args.num_envs}, device={args.device}, headless={args.headless}",
        flush=True,
    )

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

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

    # ---------------------------------------------------------
    # 强制使能碰撞和物理同步，与训练脚本对齐
    # ---------------------------------------------------------
    env_cfg.scene.replicate_physics = True
    env_cfg.scene.filter_collisions = True

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

    # ---------------------------------------------------------
    # 调用统一的全局障碍物生成器
    # ---------------------------------------------------------
    print("[INFO] Setting up global obstacles template...", flush=True)
    setup_global_obstacles(int(args.num_obstacles))

    print("[INFO] Creating env...", flush=True)
    base_env = gym.make(args.task, cfg=env_cfg).unwrapped

    scale_robot_visual_only(
        num_envs=base_env.num_envs,
        visual_scale=(20.0, 20.0, 10.0),
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
    K = int(args.history_len)
    obs_dim, act_dim, obs_space, act_space = build_spaces(base_env, state_dim, lidar_dim, K=K)

    print_space_bounds("play_obs_space", obs_space)
    print_space_bounds("play_act_space", act_space)

    env = PlaySpaceAdapter(
        base_env,
        obs_space=obs_space,
        act_space=act_space,
        state_dim=state_dim,
        lidar_dim=lidar_dim,
        K=K,
    )

    num_envs = int(getattr(env, "num_envs", args.num_envs))
    device = torch.device(getattr(env, "device", args.device))

    print(
        f"[INFO] play spaces -> obs={obs_dim} (K={K} x state={state_dim} + lidar={lidar_dim}), act={act_dim}",
        flush=True,
    )
    print(f"[INFO] num_envs={num_envs}, device={device}", flush=True)

    policy = Policy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        state_dim=state_dim,
        lidar_dim=lidar_dim,
        feat_dim=args.feat_dim,
        K=K,
        d_model=args.d_model,
        num_heads=args.num_attn_heads,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_transformer_layers,
    ).to(device)
    policy.eval()

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    log_root = project_dir / "logs"

    if args.checkpoint.strip():
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    else:
        checkpoint_path = find_latest_checkpoint(log_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_policy_checkpoint(policy, checkpoint_path, device=device)

    obs, infos = env.reset()
    # obs["policy"] already has shape (num_envs, K*single_dim) from PlaySpaceAdapter
    states = extract_policy_obs(obs)

    maybe_print_goal(base_env)

    if args.show_obs_stats:
        print_obs_summary(states, state_dim=state_dim, lidar_dim=lidar_dim, prefix="[RESET]")

    print("[INFO] Start play loop...", flush=True)

    step = 0
    episode_idx = 0
    episode_reward = torch.zeros((num_envs,), device=device, dtype=torch.float32)

    try:
        while simulation_app.is_running():
            if args.steps > 0 and step >= int(args.steps):
                print("[INFO] Reached max play steps, exiting.", flush=True)
                break

            with torch.no_grad():
                actions = policy.act(
                    states,
                    deterministic=not bool(args.use_stochastic_policy),
                )

            actions = ensure_action_shape(actions, num_envs, act_dim)
            actions = sanitize_actions(actions)

            next_obs, rewards, terminated, truncated, infos = env.step(actions)

            # next_obs["policy"] already has shape (num_envs, K*single_dim) from PlaySpaceAdapter
            next_states = extract_policy_obs(next_obs)
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

            episode_reward += rewards.squeeze(-1)
            done = terminated | truncated
            done_mask = done.squeeze(-1) if done.dim() == 2 else done

            if (step % int(args.print_every) == 0) or done_mask.any().item():
                a0 = actions[0].detach().cpu().numpy()
                r_mean = float(rewards.mean().item())
                done_count = int(done.sum().item())

                print(
                    f"[PLAY] step={step:06d} "
                    f"reward_mean={r_mean:+.4f} "
                    f"done_count={done_count} "
                    f"action0={np.array2string(a0, precision=3, separator=', ')}",
                    flush=True,
                )

                if args.show_obs_stats:
                    print_obs_summary(next_states, state_dim=state_dim, lidar_dim=lidar_dim, prefix="[PLAY]")

            if not args.headless:
                try:
                    env.render()
                except Exception:
                    pass

            if done_mask.any().item():
                done_ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
                done_ids_list = done_ids.detach().cpu().tolist()
                if isinstance(done_ids_list, int):
                    done_ids_list = [done_ids_list]

                print(f"[PLAY] episode finished on env ids: {done_ids_list}", flush=True)

                for eid in done_ids_list:
                    print(
                        f"[PLAY] env{eid} episode_reward={float(episode_reward[eid].item()):+.4f}",
                        flush=True,
                    )
                    episode_reward[eid] = 0.0

                episode_idx += len(done_ids_list)

                # IsaacLab 的 ManagerBasedRLEnv 通常在 done 的时候会自动返回重置后的 obs
                # 这里我们仅对用户可能强制要求二次 reset 做个兜底。通常在 IsaacLab 里这段逻辑是不必要的
                if args.reset_on_done:
                    # 避免对整个环境重复 reset，只刷新状态
                    reset_obs, reset_infos = env.reset(env_ids=done_ids)
                    # reset_obs["policy"] already has shape (num_envs, K*single_dim)
                    reset_states = extract_policy_obs(reset_obs)

                    next_states[done_ids] = reset_states[done_ids]

                    maybe_print_goal(base_env, env_ids=done_ids_list)

                    if args.show_obs_stats:
                        print_obs_summary(next_states, state_dim=state_dim, lidar_dim=lidar_dim, prefix="[RESET_DONE]")

            states = next_states
            step += 1

    except KeyboardInterrupt:
        print("\n[WARN] KeyboardInterrupt: stopping play", flush=True)
    finally:
        print("[INFO] Play loop exited", flush=True)
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
