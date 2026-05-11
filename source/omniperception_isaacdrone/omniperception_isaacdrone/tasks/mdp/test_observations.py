"""Observation terms and LiDAR grid preprocessing for Test6."""

from __future__ import annotations

import math
from typing import Any, Tuple

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_unique

from .test_lidar_data import get_exact_lidar_grid_cached


# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------

def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _clamp_m11(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -1.0, 1.0)


def _clamp_01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Environment config readers
# -----------------------------------------------------------------------------

def _get_cfg_obj(env: ManagerBasedRLEnv, names: Tuple[str, ...]) -> Any:
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return None
    for n in names:
        obj = getattr(cfg, n, None)
        if obj is not None:
            return obj
    return None


def _get_workspace_bounds(env: ManagerBasedRLEnv) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    norm = _get_cfg_obj(env, ("normalization", "obs_norm", "obs_normalization"))
    if norm is not None:
        xb = getattr(norm, "x_bounds", None) or getattr(norm, "workspace_x_bounds", None)
        yb = getattr(norm, "y_bounds", None) or getattr(norm, "workspace_y_bounds", None)
        zb = getattr(norm, "z_bounds", None) or getattr(norm, "workspace_z_bounds", None)
        if xb is not None and yb is not None and zb is not None:
            return (tuple(xb), tuple(yb), tuple(zb))
    return (-60.0, 60.0), (-60.0, 60.0), (0.0, 10.0)


def _get_vmax_wmax(env: ManagerBasedRLEnv) -> Tuple[float, float]:
    norm = _get_cfg_obj(env, ("normalization", "obs_norm", "obs_normalization"))
    if norm is not None:
        vmax = getattr(norm, "lin_vel_max", None)
        wmax = getattr(norm, "ang_vel_max", None)
        if vmax is not None and wmax is not None:
            return _safe_float(vmax, 5.0), _safe_float(wmax, 10.0)

    try:
        acfg = getattr(env.cfg, "actions", None)
        root_twist = getattr(acfg, "root_twist", None)
        params = getattr(root_twist, "params", None) or {}
        if isinstance(params, dict):
            vmax = _safe_float(params.get("vel_clip", 5.0), 5.0)
            wmax = _safe_float(params.get("obs_ang_vel_max", 10.0), 10.0)
            return vmax, wmax
    except Exception:
        pass
    return 5.0, 10.0


def _get_goal_distance_max(env: ManagerBasedRLEnv) -> float:
    norm = _get_cfg_obj(env, ("normalization", "obs_norm", "obs_normalization"))
    if norm is not None:
        goal_distance_max = getattr(norm, "goal_distance_max", None)
        if goal_distance_max is not None:
            return max(_safe_float(goal_distance_max, 80.0), 1e-6)

    xb, yb, zb = _get_workspace_bounds(env)
    dx = float(xb[1]) - float(xb[0])
    dy = float(yb[1]) - float(yb[0])
    dz = float(zb[1]) - float(zb[0])
    return max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-6)


def _get_quat_hemisphere(env: ManagerBasedRLEnv) -> bool:
    norm = _get_cfg_obj(env, ("normalization", "obs_norm", "obs_normalization"))
    if norm is None:
        return True
    try:
        return bool(getattr(norm, "quat_hemisphere", True))
    except Exception:
        return True


# -----------------------------------------------------------------------------
# Raw observation terms
# -----------------------------------------------------------------------------

def obs_goal_delta(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 获取目标位置与当前位置的误差，并转换到【机体坐标系 (Body Frame)】
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    goal = getattr(env, "goal_pos_w", None)

    if goal is None:
        return torch.zeros_like(pos)

    if goal.device != pos.device:
        goal = goal.to(pos.device)

    if goal.shape[0] == 1 and pos.shape[0] > 1:
        goal = goal.expand(pos.shape[0], 3)
    elif goal.shape[0] != pos.shape[0]:
        goal = goal[:1].expand(pos.shape[0], 3)

    # 世界坐标系下的距离差
    delta_w = goal - pos

    # 将世界系目标误差转换到机体坐标系
    quat_w = mdp.root_quat_w(env, asset_cfg=asset_cfg)
    return quat_apply_inverse(quat_w, delta_w)


# -----------------------------------------------------------------------------
# Normalized state observation terms
# -----------------------------------------------------------------------------

def obs_root_pos_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 根节点位置归一化
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg).to(torch.float32)
    xb, yb, zb = _get_workspace_bounds(env)

    mins = torch.tensor([float(xb[0]), float(yb[0]), float(zb[0])], device=pos.device, dtype=torch.float32)
    maxs = torch.tensor([float(xb[1]), float(yb[1]), float(zb[1])], device=pos.device, dtype=torch.float32)

    denom = torch.clamp(maxs - mins, min=1e-6)
    out = 2.0 * (pos - mins) / denom - 1.0

    return _clamp_m11(out)


def obs_root_pos_z_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 根节点 Z 轴位置归一化
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg).to(torch.float32)
    _, _, zb = _get_workspace_bounds(env)

    z = pos[:, 2:3]
    z_min, z_max = float(zb[0]), float(zb[1])
    denom = max(z_max - z_min, 1e-6)

    out = 2.0 * (z - z_min) / denom - 1.0
    return _clamp_m11(out)


def obs_root_quat_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 根节点姿态归一化：保留完整绝对姿态（包含 yaw）
    quat = mdp.root_quat_w(env, asset_cfg=asset_cfg).to(torch.float32)
    if _get_quat_hemisphere(env):
        quat = quat_unique(quat)
    return _clamp_m11(quat)


def obs_root_lin_vel_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 根节点线速度归一化 (使用机体系速度 base_lin_vel)
    v = mdp.base_lin_vel(env, asset_cfg=asset_cfg).to(torch.float32)
    vmax, _ = _get_vmax_wmax(env)
    vmax = max(float(vmax), 1e-6)
    return _clamp_m11(v / vmax)


def obs_root_ang_vel_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 根节点角速度归一化 (使用机体系角速度 base_ang_vel)
    w = mdp.base_ang_vel(env, asset_cfg=asset_cfg).to(torch.float32)
    _, wmax = _get_vmax_wmax(env)
    wmax = max(float(wmax), 1e-6)
    return _clamp_m11(w / wmax)


def obs_projected_gravity_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 投影重力归一化
    g = mdp.projected_gravity(env, asset_cfg=asset_cfg).to(torch.float32)
    return _clamp_m11(g)


def obs_goal_delta_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 目标误差归一化 (基于机体坐标系的值)
    delta_b = obs_goal_delta(env, asset_cfg=asset_cfg).to(torch.float32)
    xb, yb, zb = _get_workspace_bounds(env)

    # 为了保持机体坐标系下的旋转不变性 (各向同性)，取所有轴的最大范围作为统一缩放系数
    max_range = max(float(xb[1]) - float(xb[0]),
                    float(yb[1]) - float(yb[0]),
                    float(zb[1]) - float(zb[0]))
    max_range = max(max_range, 1e-6)

    out = delta_b / max_range

    return _clamp_m11(out)


def obs_goal_dir_dist_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # AirSim 风格目标观测：环境坐标系单位方向 + 归一化距离。
    # mdp.root_pos_w 返回 environment frame 位置；goal_pos_w 在 reset 中也按同一坐标写入。
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg).to(torch.float32)
    goal = getattr(env, "goal_pos_w", None)

    if goal is None:
        return torch.zeros((pos.shape[0], 4), device=pos.device, dtype=torch.float32)

    goal = goal.to(device=pos.device, dtype=torch.float32)
    if goal.shape[0] == 1 and pos.shape[0] > 1:
        goal = goal.expand(pos.shape[0], 3)
    elif goal.shape[0] != pos.shape[0]:
        goal = goal[:1].expand(pos.shape[0], 3)

    delta = goal - pos
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
    unit_dir = torch.where(dist > 1e-6, delta / torch.clamp(dist, min=1e-6), torch.zeros_like(delta))
    dist_norm = torch.clamp(dist / _get_goal_distance_max(env), 0.0, 1.0)

    return torch.cat([_clamp_m11(unit_dir), dist_norm], dim=-1)


def obs_prev_action_norm01(env: ManagerBasedRLEnv, action_name: str = "root_twist") -> torch.Tensor:
    try:
        term = env.action_manager.get_term(action_name)
        action = getattr(term, "raw_actions", None)
    except Exception:
        action = None

    if not isinstance(action, torch.Tensor):
        return torch.full((env.num_envs, 4), 0.5, device=env.device, dtype=torch.float32)

    action = torch.nan_to_num(action.to(device=env.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if action.dim() == 1:
        action = action.unsqueeze(0).expand(env.num_envs, -1)
    if action.shape[0] != env.num_envs:
        action = action[:1].expand(env.num_envs, -1)
    action = torch.clamp(action, -1.0, 1.0)
    return _clamp_01(0.5 * (action + 1.0))


def obs_state_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 拼接完整的无人机观测状态向量
    z = obs_root_pos_z_norm(env, asset_cfg)
    q = obs_root_quat_norm(env, asset_cfg)
    v = obs_root_lin_vel_norm(env, asset_cfg)
    w = obs_root_ang_vel_norm(env, asset_cfg)
    g = obs_projected_gravity_norm(env, asset_cfg)
    d = obs_goal_dir_dist_norm(env, asset_cfg)
    return torch.cat([z, q, v, w, g, d], dim=-1)


# -----------------------------------------------------------------------------
# LiDAR grid cache
# -----------------------------------------------------------------------------

def get_lidar_grid_cached(
    env: ManagerBasedRLEnv,
    lidar_name: str = "lidar",
    theta_min: float = 75.0,
    theta_max: float = 105.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 1.0,
    delta_phi: float = 15.0,
    empty_value: float = 0.0,
    max_vis_points: int | None = None,
    max_distance: float | None = None,
    min_range: float = 0.2,
    obstacle_size_xy: float = 1.0,
    obstacle_height: float = 10.0,
    surface_step: float = 0.5,
) -> torch.Tensor:
    """Return exact obstacle-geometry nearest-distance grid.

    The legacy function name is kept so rewards and configs can share the same
    cache interface, but the implementation no longer reads scan LiDAR data.
    ``lidar_name``, ``empty_value`` and ``max_vis_points`` are accepted only for
    backward-compatible call sites.
    """
    resolved_max_d = 10.0 if max_distance is None else float(max_distance)
    return get_exact_lidar_grid_cached(
        env,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
        delta_theta=delta_theta,
        delta_phi=delta_phi,
        min_range=min_range,
        max_distance=resolved_max_d,
        obstacle_size_xy=obstacle_size_xy,
        obstacle_height=obstacle_height,
        surface_step=surface_step,
    )


# -----------------------------------------------------------------------------
# LiDAR observation term
# -----------------------------------------------------------------------------

def obs_lidar_min_range_grid(
    env: ManagerBasedRLEnv,
    lidar_name: str = "lidar",
    theta_min: float = 75.0,
    theta_max: float = 105.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 1.0,
    delta_phi: float = 15.0,
    empty_value: float = 0.0,
    max_vis_points: int | None = None,
    max_distance: float | None = None,
    min_range: float = 0.2,
    obstacle_size_xy: float = 1.0,
    obstacle_height: float = 10.0,
    surface_step: float = 0.5,
) -> torch.Tensor:
    """Return raw nearest distances for the exact obstacle LiDAR grid."""
    return get_lidar_grid_cached(
        env, lidar_name,
        theta_min, theta_max, phi_min, phi_max,
        delta_theta, delta_phi, empty_value,
        max_vis_points, max_distance,
        min_range=min_range,
        obstacle_size_xy=obstacle_size_xy,
        obstacle_height=obstacle_height,
        surface_step=surface_step,
    )
