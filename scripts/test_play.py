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
parser.add_argument("--num_obstacles", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--state_dim", type=int, default=18)
parser.add_argument("--lidar_dim", type=int, default=144)
parser.add_argument("--feat_dim", type=int, default=256)
parser.add_argument("--model_cfg_path", type=str, default="")
parser.add_argument("--model_cfg_json", type=str, default="")
parser.add_argument("--theta_min", type=float, default=30.0)
parser.add_argument("--theta_max", type=float, default=90.0)
parser.add_argument("--phi_min", type=float, default=0.0)
parser.add_argument("--phi_max", type=float, default=360.0)
parser.add_argument("--delta_theta", type=float, default=30.0)
parser.add_argument("--delta_phi", type=float, default=5.0)
parser.add_argument("--lidar_max_distance", type=float, default=50.0)
parser.add_argument("--lidar_min_range", type=float, default=0.2)
parser.add_argument("--lidar_surface_step", type=float, default=0.5)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="",
    help="Checkpoint path (.pt). If empty, auto-search the latest under logs/",
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

# ← 新增：play 模式下直接指定障碍物数量，不走课程学习
parser.add_argument(
    "--play_obstacle_count",
    type=int,
    default=-1,
    help=(
        "Override obstacle count for play mode, bypassing curriculum. "
        "Defaults to --num_obstacles if not set (i.e. -1)."
    ),
)

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
import omniperception_isaacdrone.tasks.test_registry as _test_registry  # noqa: F401
from gymnasium.spaces import Box
from isaaclab_tasks.utils import parse_env_cfg
import isaacsim.core.utils.prims as prim_utils
from pxr import UsdGeom, Gf

from omniperception_isaacdrone.envs.test_env import WallSpawner, setup_global_obstacles
from omniperception_isaacdrone.models import (
    Policy,
    find_config_snapshot_for_checkpoint,
    load_model_cfg_from_snapshot,
    model_cfg_to_dict,
    resolve_model_cfg,
)

# -----------------------------------------------------------------------------
# Names
# -----------------------------------------------------------------------------
STATE_OBS_NAMES_18 = [
    "root_pos_z", "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z",
    "root_lin_vel_x", "root_lin_vel_y", "root_lin_vel_z",
    "root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z",
    "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
    "goal_dir_x", "goal_dir_y", "goal_dir_z", "goal_dist",
]

ACTION_NAMES_4 = ["vx_cmd", "vy_cmd", "vz_cmd", "yaw_rate_cmd"]


# -----------------------------------------------------------------------------
# Helpers — visual / debug
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
        scale_ops = [
            op for op in xform.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeScale
        ]
        if len(scale_ops) > 0:
            scale_ops[0].Set(Gf.Vec3f(sx, sy, sz))
        else:
            xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))


def format_array_preview(x: np.ndarray, max_items: int = 16) -> str:
    x = np.asarray(x).reshape(-1)
    if x.size <= max_items:
        return np.array2string(x, precision=3, separator=", ")
    return f"{np.array2string(x[:max_items], precision=3, separator=', ')} ... (total={x.size})"


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


# -----------------------------------------------------------------------------
# Observation / action utilities  （与训练脚本保持完全一致）
# -----------------------------------------------------------------------------

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
    return x.view(num_envs, obs_dim)


def ensure_action_shape(x: torch.Tensor, num_envs: int, act_dim: int) -> torch.Tensor:
    if x.dim() == 1 and x.shape[0] == act_dim:
        x = x.unsqueeze(0).repeat(num_envs, 1)
    elif x.dim() == 2 and x.shape == (1, act_dim) and num_envs > 1:
        x = x.repeat(num_envs, 1)
    return x


def ensure_vec_shape(x: torch.Tensor, num_envs: int, name: str) -> torch.Tensor:
    if x.dim() == 1 and x.shape[0] == num_envs:
        return x.unsqueeze(-1)
    if x.dim() == 2 and x.shape[0] == num_envs:
        return x
    raise RuntimeError(f"Invalid {name} shape {tuple(x.shape)}")


def sanitize_states(
    states: torch.Tensor,
    state_dim: int,
    lidar_dim: int,
    lidar_max_distance: float = 50.0,
) -> torch.Tensor:
    states = torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    state = torch.clamp(states[:, :state_dim], -1.0, 1.0)
    if lidar_dim <= 0:
        return state
    max_distance = max(float(lidar_max_distance), 1.0e-6)
    lidar = torch.clamp(states[:, state_dim: state_dim + lidar_dim] / max_distance, 0.0, 1.0)
    return torch.cat([state, lidar], dim=-1)


def sanitize_actions(actions: torch.Tensor) -> torch.Tensor:
    return torch.clamp(
        torch.nan_to_num(actions.float(), nan=0.0, posinf=0.0, neginf=0.0),
        -1.0, 1.0,
    )


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


def apply_lidar_grid_cli_params(env_cfg: Any) -> None:
    params = {
        "theta_min": float(args.theta_min),
        "theta_max": float(args.theta_max),
        "phi_min": float(args.phi_min),
        "phi_max": float(args.phi_max),
        "delta_theta": float(args.delta_theta),
        "delta_phi": float(args.delta_phi),
        "min_range": float(args.lidar_min_range),
        "max_distance": float(args.lidar_max_distance),
        "obstacle_size_xy": 1.0,
        "obstacle_height": 10.0,
        "surface_step": float(args.lidar_surface_step),
    }
    try:
        env_cfg.observations.policy.lidar_grid.params = dict(params)
    except Exception:
        pass


def build_spaces(
    base_env: Any,
    state_dim: int,
    lidar_dim: int,
    lidar_max_distance: float,
) -> tuple[int, int, gym.spaces.Dict, Box]:
    act_space = getattr(base_env, "single_action_space", None)
    if not isinstance(act_space, gym.spaces.Box):
        act_space = getattr(base_env, "action_space", None)
    if not isinstance(act_space, gym.spaces.Box):
        raise RuntimeError("Action space is not a gym.spaces.Box")

    act_dim = infer_single_dim_from_box(act_space)
    obs_dim = state_dim + lidar_dim

    obs_low  = -np.ones((obs_dim,), dtype=np.float32)
    obs_high =  np.ones((obs_dim,), dtype=np.float32)
    if lidar_dim > 0:
        obs_low[state_dim:] = 0.0
        obs_high[state_dim:] = 1.0

    obs_space = gym.spaces.Dict({"policy": Box(low=obs_low, high=obs_high, dtype=np.float32)})
    act_box   = Box(
        low=-np.ones((act_dim,), dtype=np.float32),
        high= np.ones((act_dim,), dtype=np.float32),
        dtype=np.float32,
    )
    return obs_dim, act_dim, obs_space, act_box


# -----------------------------------------------------------------------------
# Env adapter  （与训练脚本 SkrlSpaceAdapter 逻辑完全一致）
# -----------------------------------------------------------------------------

class PlaySpaceAdapter(gym.Wrapper):
    """Expose stable single-env semantic spaces and sanitized observations."""

    def __init__(
        self,
        env: gym.Env,
        obs_space: gym.spaces.Dict,
        act_space: Box,
        state_dim: int,
        lidar_dim: int,
        lidar_max_distance: float = 50.0,
    ):
        super().__init__(env)
        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)
        self.obs_dim   = self.state_dim + self.lidar_dim
        self.lidar_max_distance = float(lidar_max_distance)

        self.observation_space        = obs_space
        self.single_observation_space = obs_space
        self.action_space             = act_space
        self.single_action_space      = act_space

        self.num_envs = int(getattr(env, "num_envs", 1))
        self.device   = getattr(env, "device", None)

    def _convert_obs(self, raw_obs: Any) -> dict[str, torch.Tensor]:
        return {
            "policy": sanitize_states(
                ensure_obs_shape(extract_policy_obs(raw_obs), self.num_envs, self.obs_dim),
                self.state_dim,
                self.lidar_dim,
                lidar_max_distance=self.lidar_max_distance,
            )
        }

    def reset(self, **kwargs):
        # IsaacLab ManagerBasedRLEnv.reset() 不接受 env_ids 关键字；
        # 直接忽略多余参数，整体 reset。
        raw_obs, infos = self.env.reset()
        return self._convert_obs(raw_obs), infos

    def step(self, actions):
        raw_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        return self._convert_obs(raw_obs), rewards, terminated, truncated, infos


# -----------------------------------------------------------------------------
# Space / name utilities
# -----------------------------------------------------------------------------

def build_state_names(state_dim: int) -> list[str]:
    return list(STATE_OBS_NAMES_18) if int(state_dim) == len(STATE_OBS_NAMES_18) else [f"state_{i}" for i in range(int(state_dim))]


def build_action_names(act_dim: int) -> list[str]:
    return list(ACTION_NAMES_4) if int(act_dim) == len(ACTION_NAMES_4) else [f"action_{i}" for i in range(int(act_dim))]


def print_obs_summary(obs: torch.Tensor, state_dim: int, lidar_dim: int, prefix: str = "[OBS]") -> None:
    obs0  = obs[0].detach().cpu()
    state = obs0[:state_dim]
    print(f"{prefix} state = {np.array2string(state.numpy(), precision=3, separator=', ')}", flush=True)
    if lidar_dim > 0:
        lidar = obs0[state_dim: state_dim + lidar_dim]
        nz    = float((lidar > 1e-6).float().mean().item())
        print(
            f"{prefix} lidar: min={float(lidar.min().item()):.4f}, "
            f"max={float(lidar.max().item()):.4f}, "
            f"mean={float(lidar.mean().item()):.4f}, "
            f"nonzero_ratio={nz:.4f}",
            flush=True,
        )


# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------

def find_latest_checkpoint(log_root: Path) -> Path:
    if not log_root.exists():
        raise FileNotFoundError(f"log root not found: {log_root}")

    # 优先搜索 manual_checkpoints 目录下的 .pt 文件
    candidates: list[Path] = sorted(
        log_root.rglob("*.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    # 按优先级：manual_checkpoints → 其余 .pt → .pth
    manual = [p for p in candidates if "manual_checkpoints" in str(p)]
    other  = [p for p in candidates if "manual_checkpoints" not in str(p)]
    pth    = sorted(log_root.rglob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)

    ordered = manual + other + pth
    if len(ordered) == 0:
        raise FileNotFoundError(f"No checkpoint (.pt / .pth) found under {log_root}")

    chosen = ordered[0]
    print(f"[INFO] Auto-selected checkpoint: {chosen}", flush=True)
    return chosen


def load_policy_checkpoint(
    policy: Policy,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """
    训练脚本保存格式（manual_checkpoints）：
        torch.save({name: model.state_dict() for name, model in models.items()}, ...)
    即 payload = {"policy": {...}, "value": {...}}

    skrl 内置保存格式（experiment checkpoints）：
        payload["policy"] 可能是完整 state_dict，也可能嵌套在其他键下。

    本函数统一处理以上两种格式。
    """
    print(f"[INFO] Loading checkpoint: {checkpoint_path}", flush=True)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict: dict | None = None

    # ① 训练脚本 manual_checkpoints 格式：{"policy": state_dict, "value": state_dict}
    if isinstance(payload, dict) and "policy" in payload and isinstance(payload["policy"], dict):
        candidate = payload["policy"]
        # 确认确实是 state_dict（含 weight/bias/log_std_parameter 等键）
        if any(("weight" in k or "bias" in k or "log_std_parameter" in k) for k in candidate.keys()):
            state_dict = candidate

    # ② 直接是 state_dict（顶层就是参数键）
    if state_dict is None and isinstance(payload, dict):
        if any(("weight" in k or "bias" in k or "log_std_parameter" in k) for k in payload.keys()):
            state_dict = payload

    # ③ skrl 内置 checkpoint 可能再嵌套一层
    if state_dict is None and isinstance(payload, dict):
        for k in ["policy_state_dict", "model", "state_dict", "actor", "network"]:
            if k in payload and isinstance(payload[k], dict):
                state_dict = payload[k]
                break

    if state_dict is None:
        raise RuntimeError(
            f"Cannot extract policy state_dict from checkpoint {checkpoint_path}. "
            f"Top-level keys: {list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
        )

    try:
        policy.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Policy checkpoint does not match the play network definition. "
            "This usually means test_play.py and the training-time feature extractor are not aligned."
        ) from exc
    print("[INFO] Policy checkpoint loaded successfully.", flush=True)


def resolve_play_model_cfg(
    *,
    checkpoint_path: Path,
    feat_dim: int,
    lidar_grid_shape: tuple[int, int] | None,
) -> tuple[Any, dict[str, Any], Path | None]:
    snapshot_path = find_config_snapshot_for_checkpoint(checkpoint_path)
    snapshot_model_cfg = None
    if snapshot_path is not None:
        snapshot_model_cfg = load_model_cfg_from_snapshot(snapshot_path)
        if snapshot_model_cfg is not None:
            print(f"[INFO] Loaded model_cfg snapshot: {snapshot_path}", flush=True)
    model_cfg = resolve_model_cfg(
        feat_dim=int(feat_dim),
        lidar_grid_shape=lidar_grid_shape,
        model_cfg_path=args.model_cfg_path or None,
        model_cfg_json=args.model_cfg_json or None,
        base_model_cfg=snapshot_model_cfg,
    )
    return model_cfg, model_cfg_to_dict(model_cfg), snapshot_path


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

def maybe_print_goal(base_env: Any, env_ids: list[int] | None = None) -> None:
    try:
        if not hasattr(base_env, "goal_pos_w"):
            return
        goal = base_env.goal_pos_w.detach().cpu().numpy()
        for eid in (env_ids or [0]):
            if 0 <= int(eid) < goal.shape[0]:
                print(f"[INFO] goal_pos_w[{eid}]={goal[eid]}", flush=True)
    except Exception:
        pass


# ← 新增：play 模式直接固定障碍物数量，绕过课程学习晋级逻辑
def freeze_obstacle_count_for_play(base_env: Any, obstacle_count: int) -> None:
    """
    在 play 模式下直接设置障碍物数量，并锁定课程学习使其不再晋级。

    做法：
    1. 强制写入 curr_obstacle_count，使 randomize_obstacles_on_reset
       读取到正确数量。
    2. 将 curr_level_idx 设为最大值，使晋级条件永远不满足。
    3. 同步初始化 curr_history / curr_levels 等字段，避免首次调用
       update_obstacle_curriculum 时报 AttributeError。
    """
    u = base_env.unwrapped if hasattr(base_env, "unwrapped") else base_env
    count = max(0, int(obstacle_count))

    # 直接覆盖/写入目标字段
    u.curr_obstacle_count   = count
    u.obstacle_level_changed = True  # 触发首次重排

    # 如果课程学习状态尚未初始化，构造一个"单级别"的假状态
    # 使后续任何 update_obstacle_curriculum 调用直接返回而不晋级
    if not hasattr(u, "curr_history"):
        device = torch.device(getattr(u, "device", "cpu"))
        num_envs = int(getattr(u, "num_envs", 1))
        # 单一等级列表，idx 永远停在 0，无法晋级
        u.curr_levels      = [count]
        u.curr_num_envs    = num_envs
        u.curr_k_roll      = 1
        u.curr_window_size = num_envs
        u.curr_history     = torch.full(
            (num_envs,), -1.0, dtype=torch.float32, device=device
        )
        u.curr_write_round = torch.zeros(num_envs, dtype=torch.long, device=device)
        u.curr_level_idx   = 0          # 已是末尾，不会晋级
        u.curr_device      = device
    else:
        # 课程学习状态已存在：把 level_idx 推到末尾使其无法晋级
        u.curr_levels      = [count]   # 重写为单级别
        u.curr_level_idx   = 0         # 已是末尾
        # 清空历史，防止残留脏数据触发晋级
        u.curr_history.fill_(-1.0)

    print(
        f"[PLAY] 障碍物数量已固定为 {count}，课程学习晋级已禁用。",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    # 确定实际障碍物数量：--play_obstacle_count 优先，否则回退到 --num_obstacles
    play_obstacle_count = (
        int(args.num_obstacles)
        if args.play_obstacle_count < 0
        else int(args.play_obstacle_count)
    )

    print(
        f"[INFO] task={args.task}, num_envs={args.num_envs}, "
        f"device={args.device}, headless={args.headless}, "
        f"play_obstacle_count={play_obstacle_count}",
        flush=True,
    )

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    # ------------------------------------------------------------------
    # 环境配置（与训练脚本保持一致）
    # ------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    apply_lidar_grid_cli_params(env_cfg)
    env_cfg.scene.replicate_physics = True
    env_cfg.scene.filter_collisions = True
    try:
        setattr(env_cfg, "seed", int(args.seed))
    except Exception:
        pass

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
    # ← 全局预生成的上限仍用 num_obstacles，保证模板池足够大
    setup_global_obstacles(int(args.num_obstacles))

    print("[INFO] Creating env...", flush=True)
    base_env = gym.make(args.task, cfg=env_cfg).unwrapped
    scale_robot_visual_only(num_envs=base_env.num_envs, visual_scale=(20.0, 20.0, 10.0))

    # ← 新增：环境创建后立即固定障碍物数量，绕过课程学习
    freeze_obstacle_count_for_play(base_env, play_obstacle_count)

    # ------------------------------------------------------------------
    # 推断观测 / 动作空间
    # ------------------------------------------------------------------
    space        = getattr(base_env, "single_observation_space", None)
    policy_space = (
        space.spaces.get("policy", None)
        if isinstance(space, gym.spaces.Dict)
        else getattr(base_env, "observation_space", None)
    )
    if policy_space is None:
        raise RuntimeError("policy observation space not found")

    obs_dim_raw          = int(np.prod(policy_space.shape))
    state_dim, lidar_dim = get_state_lidar_dims(base_env, obs_dim_raw)
    lidar_grid_shape     = infer_lidar_grid_shape(base_env, lidar_dim)
    obs_dim, act_dim, obs_space, act_space = build_spaces(
        base_env,
        state_dim,
        lidar_dim,
        lidar_max_distance=float(args.lidar_max_distance),
    )

    print_space_bounds("play_obs_space", obs_space)
    print_space_bounds("play_act_space", act_space)

    env = PlaySpaceAdapter(
        base_env,
        obs_space=obs_space,
        act_space=act_space,
        state_dim=state_dim,
        lidar_dim=lidar_dim,
        lidar_max_distance=float(args.lidar_max_distance),
    )

    num_envs = int(getattr(env, "num_envs", args.num_envs))
    device   = torch.device(getattr(env, "device", args.device))

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    log_root = project_dir / "logs"

    if args.checkpoint.strip():
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    else:
        checkpoint_path = find_latest_checkpoint(log_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_cfg, model_cfg_dict, snapshot_path = resolve_play_model_cfg(
        checkpoint_path=checkpoint_path,
        feat_dim=int(args.feat_dim),
        lidar_grid_shape=lidar_grid_shape,
    )
    if int(lidar_dim) > 0:
        model_cfg.lidar_encoder.input_clamp = [0.0, 1.0]
        model_cfg.lidar_encoder.input_rescale_to_neg_one_to_one = True
        model_cfg_dict = model_cfg_to_dict(model_cfg)

    print(
        f"[INFO] play spaces -> obs={obs_dim} (state={state_dim}, lidar={lidar_dim}), "
        f"act={act_dim}, grid={model_cfg.lidar_encoder.grid_shape}, encoder={model_cfg.lidar_encoder.type}",
        flush=True,
    )
    if snapshot_path is None:
        print("[INFO] No config snapshot found for checkpoint; using default/CLI model_cfg.", flush=True)
    print(f"[INFO] num_envs={num_envs}, device={device}", flush=True)
    print(f"[INFO] model_cfg={model_cfg_dict}", flush=True)

    # ------------------------------------------------------------------
    # 构建策略网络（与训练脚本 Policy 完全一致）
    # ------------------------------------------------------------------
    policy = Policy(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        state_dim=state_dim,
        lidar_dim=lidar_dim,
        feat_dim=args.feat_dim,
        lidar_grid_shape=lidar_grid_shape,
        model_cfg=model_cfg,
    )
    policy.to(device)
    policy.eval()

    load_policy_checkpoint(policy, checkpoint_path, device=device)

    # ------------------------------------------------------------------
    # 初始 reset
    # ------------------------------------------------------------------
    obs, infos = env.reset()
    states = sanitize_states(
        ensure_obs_shape(extract_policy_obs(obs), num_envs, obs_dim),
        state_dim=state_dim,
        lidar_dim=lidar_dim,
        lidar_max_distance=float(args.lidar_max_distance),
    )

    maybe_print_goal(base_env)
    if args.show_obs_stats:
        print_obs_summary(states, state_dim=state_dim, lidar_dim=lidar_dim, prefix="[RESET]")

    print("[INFO] Start play loop...", flush=True)

    step           = 0
    episode_idx    = 0
    episode_reward = torch.zeros((num_envs,), device=device, dtype=torch.float32)

    try:
        while simulation_app.is_running():
            if args.steps > 0 and step >= int(args.steps):
                print("[INFO] Reached max play steps, exiting.", flush=True)
                break

            # --------------------------------------------------------------
            # 推理
            # --------------------------------------------------------------
            with torch.no_grad():
                actions = policy.play_act(
                    states,
                    deterministic=not bool(args.use_stochastic_policy),
                )

            actions = ensure_action_shape(actions, num_envs, act_dim)
            actions = sanitize_actions(actions)

            # --------------------------------------------------------------
            # 环境交互
            # --------------------------------------------------------------
            next_obs, rewards, terminated, truncated, infos = env.step(actions)

            next_states = sanitize_states(
                ensure_obs_shape(extract_policy_obs(next_obs), num_envs, obs_dim),
                state_dim=state_dim,
                lidar_dim=lidar_dim,
                lidar_max_distance=float(args.lidar_max_distance),
            )
            rewards    = ensure_vec_shape(
                torch.nan_to_num(rewards.float(),    nan=0.0, posinf=0.0, neginf=0.0),
                num_envs, "rewards",
            )
            terminated = ensure_vec_shape(
                torch.nan_to_num(terminated.float(), nan=0.0, posinf=0.0, neginf=0.0),
                num_envs, "terminated",
            ).bool()
            truncated  = ensure_vec_shape(
                torch.nan_to_num(truncated.float(),  nan=0.0, posinf=0.0, neginf=0.0),
                num_envs, "truncated",
            ).bool()

            episode_reward += rewards.squeeze(-1)
            done      = terminated | truncated
            done_mask = done.squeeze(-1) if done.dim() == 2 else done

            # --------------------------------------------------------------
            # 打印
            # --------------------------------------------------------------
            if (step % int(args.print_every) == 0) or done_mask.any().item():
                a0         = actions[0].detach().cpu().numpy()
                r_mean     = float(rewards.mean().item())
                done_count = int(done.sum().item())
                print(
                    f"[PLAY] step={step:06d} "
                    f"reward_mean={r_mean:+.4f} "
                    f"done_count={done_count} "
                    f"action0={np.array2string(a0, precision=3, separator=', ')}",
                    flush=True,
                )
                if args.show_obs_stats:
                    print_obs_summary(
                        next_states, state_dim=state_dim, lidar_dim=lidar_dim, prefix="[PLAY]"
                    )

            if not args.headless:
                try:
                    env.render()
                except Exception:
                    pass

            # --------------------------------------------------------------
            # Episode 结束处理
            # IsaacLab ManagerBasedRLEnv 在 done 时已在内部自动重置对应子环境，
            # 并把重置后的 obs 作为 next_obs 返回，因此通常无需额外调用 reset()。
            # --reset_on_done 标志仅用于打印诊断信息并刷新状态缓存。
            # --------------------------------------------------------------
            if done_mask.any().item():
                done_ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
                done_ids_list: list[int] = done_ids.detach().cpu().tolist()
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
                maybe_print_goal(base_env, env_ids=done_ids_list)

                if args.show_obs_stats:
                    print_obs_summary(
                        next_states, state_dim=state_dim, lidar_dim=lidar_dim, prefix="[DONE]"
                    )

            states = next_states
            step  += 1

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
