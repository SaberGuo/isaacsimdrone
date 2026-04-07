from __future__ import annotations

import math
from typing import Any, Tuple

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# =============================================================================
# 基础辅助函数
# =============================================================================

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


# =============================================================================
# 环境配置读取辅助函数
# =============================================================================

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


def _get_quat_hemisphere(env: ManagerBasedRLEnv) -> bool:
    norm = _get_cfg_obj(env, ("normalization", "obs_norm", "obs_normalization"))
    if norm is None:
        return True
    try:
        return bool(getattr(norm, "quat_hemisphere", True))
    except Exception:
        return True


# =============================================================================
# 激光雷达配置与预处理辅助函数
# =============================================================================

def _get_lidar_ranges(lidar, default_min: float = 0.2, default_max: float = 50.0) -> tuple[float, float]:
    min_r = float(default_min)
    max_r = float(default_max)
    try:
        if hasattr(lidar, "cfg"):
            if hasattr(lidar.cfg, "min_range"):
                min_r = float(lidar.cfg.min_range)
            if hasattr(lidar.cfg, "max_distance"):
                max_r = float(lidar.cfg.max_distance)
    except Exception:
        pass
    return min_r, max_r


def _get_downsampled_pc_torch(env, lidar, env_ids: torch.Tensor, max_pts: int | None):
    if lidar is None:
        return None, None

    pc = lidar.get_pointcloud(env_ids)
    if pc is None:
        return None, None

    if pc.dim() == 2:
        pc = pc.unsqueeze(0)

    E, P, _ = pc.shape
    num_raw = torch.full((E,), P, device=pc.device, dtype=torch.int32)
    
    finite_mask = torch.isfinite(pc).all(dim=-1)
    pc = pc.clone()
    pc[~finite_mask] = float("nan")

    return pc, num_raw


# =============================================================================
# 原始观测函数
# =============================================================================

def obs_goal_delta(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 获取目标位置与当前位置的误差
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

    return goal - pos


# =============================================================================
# 归一化观测函数（限制在 [-1, 1] 之间）
# =============================================================================

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
    # 根节点四元数归一化与半球校正
    quat = mdp.root_quat_w(env, asset_cfg=asset_cfg).to(torch.float32)

    if _get_quat_hemisphere(env):
        w = quat[:, 0:1]
        sign = torch.where(w < 0.0, torch.tensor(-1.0, device=quat.device), torch.tensor(1.0, device=quat.device))
        quat = quat * sign

    return _clamp_m11(quat)


def obs_root_lin_vel_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 根节点线速度归一化
    v = mdp.root_lin_vel_w(env, asset_cfg=asset_cfg).to(torch.float32)
    vmax, _ = _get_vmax_wmax(env)
    vmax = max(float(vmax), 1e-6)
    return _clamp_m11(v / vmax)


def obs_root_ang_vel_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 根节点角速度归一化
    w = mdp.root_ang_vel_w(env, asset_cfg=asset_cfg).to(torch.float32)
    _, wmax = _get_vmax_wmax(env)
    wmax = max(float(wmax), 1e-6)
    return _clamp_m11(w / wmax)


def obs_projected_gravity_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 投影重力归一化
    g = mdp.projected_gravity(env, asset_cfg=asset_cfg).to(torch.float32)
    return _clamp_m11(g)


def obs_goal_delta_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 目标误差归一化
    delta = obs_goal_delta(env, asset_cfg=asset_cfg).to(torch.float32)
    xb, yb, zb = _get_workspace_bounds(env)

    axis_range = torch.tensor(
        [float(xb[1]) - float(xb[0]), float(yb[1]) - float(yb[0]), float(zb[1]) - float(zb[0])],
        device=delta.device,
        dtype=torch.float32,
    )

    axis_range = torch.clamp(axis_range, min=1e-6)
    out = delta / axis_range

    return _clamp_m11(out)


def obs_state_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 拼接完整的无人机观测状态向量
    z = obs_root_pos_z_norm(env, asset_cfg)
    q = obs_root_quat_norm(env, asset_cfg)
    v = obs_root_lin_vel_norm(env, asset_cfg)
    w = obs_root_ang_vel_norm(env, asset_cfg)
    g = obs_projected_gravity_norm(env, asset_cfg)
    d = obs_goal_delta_norm(env, asset_cfg)
    return torch.cat([z, q, v, w, g, d], dim=-1)


# =============================================================================
# 激光雷达网格观测函数
# =============================================================================

def obs_lidar_min_range_grid(
    env: ManagerBasedRLEnv,
    lidar_name: str = "lidar",
    theta_min: float = 30.0,
    theta_max: float = 90.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 1.0,
    delta_phi: float = 5.0,
    empty_value: float = 0.0,
    max_vis_points: int | None = None,
    max_distance: float | None = None,
) -> torch.Tensor:
    # 计算激光雷达基于网格过滤的距离逼近度，限制在 [0, 1] 区间
    T = max(int((theta_max - theta_min) / delta_theta), 1)
    Pn = max(int((phi_max - phi_min) / delta_phi), 1)
    out_shape = (env.num_envs, T * Pn)

    if not hasattr(env, "scene"):
        return torch.zeros(out_shape, device=env.device, dtype=torch.float32)

    try:
        lidar = env.scene[lidar_name]
    except Exception:
        return torch.zeros(out_shape, device=env.device, dtype=torch.float32)

    min_r_cfg, max_r_cfg = _get_lidar_ranges(lidar, default_min=0.2, default_max=50.0)
    max_d = float(max_distance) if max_distance is not None else float(max_r_cfg)
    min_r = float(min_r_cfg)

    env_ids = torch.arange(env.num_envs, device=env.device)
    pc, _ = _get_downsampled_pc_torch(env, lidar, env_ids, max_pts=max_vis_points)
    
    if pc is None:
        return torch.zeros(out_shape, device=env.device, dtype=torch.float32)

    x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]
    valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z)
    r = torch.sqrt(x * x + y * y + z * z + 1e-12)

    valid = valid & (r > (min_r + 1e-3)) & (r <= (max_d + 1e-3))

    cos_theta = torch.clamp(z / r, -1.0, 1.0)
    theta = torch.rad2deg(torch.acos(cos_theta))
    phi = torch.remainder(torch.rad2deg(torch.atan2(y, x)), 360.0)

    in_theta = (theta >= theta_min) & (theta < theta_max)
    in_phi = (phi >= phi_min) & (phi < phi_max)
    m = valid & in_theta & in_phi

    num_bins = T * Pn
    min_dist = torch.full((env.num_envs, num_bins), float("inf"), device=env.device, dtype=torch.float32)

    if m.any():
        t_idx = torch.clamp(torch.floor((theta - theta_min) / delta_theta).to(torch.long), 0, T - 1)
        p_idx = torch.clamp(torch.floor((phi - phi_min) / delta_phi).to(torch.long), 0, Pn - 1)
        lin_idx = t_idx * Pn + p_idx

        for e in range(env.num_envs):
            me = m[e]
            if me.any():
                idx_e = lin_idx[e, me]
                r_e = r[e, me].to(torch.float32)
                min_dist[e].scatter_reduce_(0, idx_e, r_e, reduce="amin", include_self=True)

    max_d_t = torch.tensor(max_d, device=env.device, dtype=torch.float32)
    min_dist = torch.where(torch.isfinite(min_dist), min_dist, max_d_t)
    min_dist = torch.clamp(min_dist, 0.0, max_d_t)

    closeness = 1.0 - torch.clamp(min_dist / max_d_t, 0.0, 1.0)
    return _clamp_01(closeness.to(torch.float32))
