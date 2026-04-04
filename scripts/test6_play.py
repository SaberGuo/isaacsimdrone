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


def build_spaces(base_env: Any, state_dim: int, lidar_dim: int) -> tuple[int, int, gym.spaces.Dict, Box]:
    act_space = getattr(base_env, "single_action_space", None)
    if not isinstance(act_space, gym.spaces.Box):
        act_space = getattr(base_env, "action_space", None)

    if not isinstance(act_space, gym.spaces.Box):
        raise RuntimeError("Action space is not a gym.spaces.Box")

    act_dim = infer_single_dim_from_box(act_space)
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


class PlaySpaceAdapter(gym.Wrapper):
    """Expose stable single-env semantic spaces and sanitized observations."""

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


class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, state_dim: int, lidar_dim: int, feat_dim: int = 256):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)

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
    obs_dim, act_dim, obs_space, act_space = build_spaces(base_env, state_dim, lidar_dim)

    print_space_bounds("play_obs_space", obs_space)
    print_space_bounds("play_act_space", act_space)

    env = PlaySpaceAdapter(
        base_env,
        obs_space=obs_space,
        act_space=act_space,
        state_dim=state_dim,
        lidar_dim=lidar_dim,
    )

    num_envs = int(getattr(env, "num_envs", args.num_envs))
    device = torch.device(getattr(env, "device", args.device))

    print(
        f"[INFO] play spaces -> obs={obs_dim} (state={state_dim}, lidar={lidar_dim}), act={act_dim}",
        flush=True,
    )
    print(f"[INFO] num_envs={num_envs}, device={device}", flush=True)

    policy = Policy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        state_dim=state_dim,
        lidar_dim=lidar_dim,
        feat_dim=args.feat_dim,
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
    states = sanitize_states(
        ensure_obs_shape(extract_policy_obs(obs), num_envs, obs_dim),
        state_dim=state_dim,
        lidar_dim=lidar_dim,
    )

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
                    reset_states = sanitize_states(
                        ensure_obs_shape(extract_policy_obs(reset_obs), num_envs, obs_dim),
                        state_dim=state_dim,
                        lidar_dim=lidar_dim,
                    )

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
