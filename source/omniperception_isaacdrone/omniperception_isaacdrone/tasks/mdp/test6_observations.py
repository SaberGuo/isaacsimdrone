# omniperception_isaacdrone/tasks/mdp/test6_observations.py

from __future__ import annotations

import math
from typing import Any, Tuple

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# =============================================================================
# Helpers: read normalization hyper-parameters from env.cfg (no circular imports)
# =============================================================================

def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _get_cfg_obj(env: ManagerBasedRLEnv, names: Tuple[str, ...]) -> Any:
    """Try get env.cfg.<name> in order. Return None if not found."""
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return None
    for n in names:
        obj = getattr(cfg, n, None)
        if obj is not None:
            return obj
    return None


def _get_workspace_bounds(env: ManagerBasedRLEnv) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Return (x_bounds, y_bounds, z_bounds) from env.cfg.normalization if possible."""
    norm = _get_cfg_obj(env, ("normalization", "obs_norm", "obs_normalization"))
    if norm is not None:
        xb = getattr(norm, "x_bounds", None) or getattr(norm, "workspace_x_bounds", None)
        yb = getattr(norm, "y_bounds", None) or getattr(norm, "workspace_y_bounds", None)
        zb = getattr(norm, "z_bounds", None) or getattr(norm, "workspace_z_bounds", None)
        if xb is not None and yb is not None and zb is not None:
            return (tuple(xb), tuple(yb), tuple(zb))

    # Fallback (keep same defaults as your termination_out_of_workspace)
    return (-60.0, 60.0), (-60.0, 60.0), (0.0, 10.0)


def _get_diag_scale(env: ManagerBasedRLEnv) -> float:
    norm = _get_cfg_obj(env, ("normalization", "obs_norm", "obs_normalization"))
    if norm is None:
        return 1.1
    return _safe_float(getattr(norm, "diag_scale", 1.1), 1.1)


def _get_vmax_wmax(env: ManagerBasedRLEnv) -> Tuple[float, float]:
    """Read vmax/wmax from env.cfg.normalization, fallback to action params, then defaults."""
    norm = _get_cfg_obj(env, ("normalization", "obs_norm", "obs_normalization"))
    if norm is not None:
        vmax = getattr(norm, "lin_vel_max", None)
        wmax = getattr(norm, "ang_vel_max", None)
        if vmax is not None and wmax is not None:
            return _safe_float(vmax, 5.0), _safe_float(wmax, 10.0)

    # fallback: try action term params (if exists)
    try:
        acfg = getattr(env.cfg, "actions", None)
        root_twist = getattr(acfg, "root_twist", None)
        params = getattr(root_twist, "params", None) or {}
        if isinstance(params, dict):
            # Use vel_clip as a reasonable vmax; yaw_rate_clip only controls yaw command, not full body ang vel
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


def _diag_full(xb: Tuple[float, float], yb: Tuple[float, float], zb: Tuple[float, float]) -> float:
    """Full diagonal length across the workspace bounding box."""
    dx = abs(float(xb[1]) - float(xb[0]))
    dy = abs(float(yb[1]) - float(yb[0]))
    dz = abs(float(zb[1]) - float(zb[0]))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _clamp_m11(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -1.0, 1.0)


def _clamp_01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


# =============================================================================
# 1) goal delta (raw, unchanged)
# =============================================================================
def obs_goal_delta(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """goal_state_delta = goal_pos_w - uav_root_pos_w (raw in meters)."""
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)  # (N, 3)
    goal = getattr(env, "goal_pos_w", None)
    if goal is None:
        return torch.zeros_like(pos)

    if goal.device != pos.device:
        goal = goal.to(pos.device)

    # broadcast safety
    if goal.shape[0] == 1 and pos.shape[0] > 1:
        goal = goal.expand(pos.shape[0], 3)
    elif goal.shape[0] != pos.shape[0]:
        goal = goal[:1].expand(pos.shape[0], 3)

    return goal - pos


# =============================================================================
# 2) normalized state terms (strict bounds)
# =============================================================================

def obs_root_pos_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Normalize root position (world) per-axis using workspace bounds into [-1,1].

    For each axis:
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
        z_norm = 2 * (z - z_min) / (z_max - z_min) - 1
    """
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg).to(torch.float32)  # (N, 3)
    xb, yb, zb = _get_workspace_bounds(env)

    mins = torch.tensor(
        [float(xb[0]), float(yb[0]), float(zb[0])],
        device=pos.device,
        dtype=torch.float32,
    )
    maxs = torch.tensor(
        [float(xb[1]), float(yb[1]), float(zb[1])],
        device=pos.device,
        dtype=torch.float32,
    )

    denom = torch.clamp(maxs - mins, min=1e-6)
    out = 2.0 * (pos - mins) / denom - 1.0

    return _clamp_m11(out)

def obs_root_pos_z_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Only keep normalized root z position in [-1, 1]. Shape: (N, 1)."""
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg).to(torch.float32)  # (N, 3)
    _, _, zb = _get_workspace_bounds(env)

    z = pos[:, 2:3]
    z_min = float(zb[0])
    z_max = float(zb[1])
    denom = max(z_max - z_min, 1e-6)

    out = 2.0 * (z - z_min) / denom - 1.0
    return _clamp_m11(out)



def obs_root_quat_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Quaternion is already in [-1,1], but we:
    - ensure float32
    - optionally force hemisphere (w >= 0) to avoid sign-flip discontinuity
    - clamp to [-1,1]
    """
    quat = mdp.root_quat_w(env, asset_cfg=asset_cfg).to(torch.float32)  # (N,4) wxyz

    if _get_quat_hemisphere(env):
        w = quat[:, 0:1]
        sign = torch.where(w < 0.0, torch.tensor(-1.0, device=quat.device), torch.tensor(1.0, device=quat.device))
        quat = quat * sign

    return _clamp_m11(quat)


def obs_root_lin_vel_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Normalize root linear velocity (world) into [-1,1] by vmax."""
    v = mdp.root_lin_vel_w(env, asset_cfg=asset_cfg).to(torch.float32)  # (N,3)
    vmax, _ = _get_vmax_wmax(env)
    vmax = max(float(vmax), 1e-6)
    return _clamp_m11(v / vmax)


def obs_root_ang_vel_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Normalize root angular velocity (world) into [-1,1] by wmax."""
    w = mdp.root_ang_vel_w(env, asset_cfg=asset_cfg).to(torch.float32)  # (N,3)
    _, wmax = _get_vmax_wmax(env)
    wmax = max(float(wmax), 1e-6)
    return _clamp_m11(w / wmax)


def obs_projected_gravity_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Projected gravity is unit-ish already. Clamp to [-1,1] for strictness."""
    g = mdp.projected_gravity(env, asset_cfg=asset_cfg).to(torch.float32)  # (N,3)
    return _clamp_m11(g)


def obs_goal_delta_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Normalize goal delta (goal - pos) per-axis into [-1, 1].

    For each axis:
        dx_norm = (goal_x - x) / (x_max - x_min)
        dy_norm = (goal_y - y) / (y_max - y_min)
        dz_norm = (goal_z - z) / (z_max - z_min)
    """
    delta = obs_goal_delta(env, asset_cfg=asset_cfg).to(torch.float32)  # (N, 3)
    xb, yb, zb = _get_workspace_bounds(env)

    axis_range = torch.tensor(
        [
            float(xb[1]) - float(xb[0]),
            float(yb[1]) - float(yb[0]),
            float(zb[1]) - float(zb[0]),
        ],
        device=delta.device,
        dtype=torch.float32,
    )

    axis_range = torch.clamp(axis_range, min=1e-6)
    out = delta / axis_range

    return _clamp_m11(out)


def obs_state_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Produce the 16D normalized state vector excluding root_pos_z:
      [root_quat(4), root_lin_vel(3), root_ang_vel(3), projected_gravity(3), goal_delta(3)]
    """

    z = obs_root_pos_z_norm(env, asset_cfg)
    q = obs_root_quat_norm(env, asset_cfg)
    v = obs_root_lin_vel_norm(env, asset_cfg)
    w = obs_root_ang_vel_norm(env, asset_cfg)
    g = obs_projected_gravity_norm(env, asset_cfg)
    d = obs_goal_delta_norm(env, asset_cfg)
    return torch.cat([z, q, v, w, g, d], dim=-1)  # (N,17)



# =============================================================================
# 3) LiDAR grid closeness (already [0,1], keep strict clamp)
# =============================================================================
def _get_lidar_ranges(lidar, default_min: float = 0.2, default_max: float = 50.0) -> tuple[float, float]:
    """Best-effort read of lidar min/max range from cfg."""
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
    """Read lidar pointcloud (torch) and mask invalid points with NaN."""
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


def obs_lidar_min_range_grid(
    env: "ManagerBasedRLEnv",
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
    """Return flattened closeness grid in [0,1]. Strictly clamped to [0,1]."""

    # bin counts
    T = int((theta_max - theta_min) / delta_theta)
    Pn = int((phi_max - phi_min) / delta_phi)
    T = max(T, 1)
    Pn = max(Pn, 1)
    out_shape = (env.num_envs, T * Pn)

    if not hasattr(env, "scene"):
        return torch.zeros(out_shape, device=env.device, dtype=torch.float32)

    try:
        lidar = env.scene[lidar_name]
    except Exception:
        return torch.zeros(out_shape, device=env.device, dtype=torch.float32)

    min_r_cfg, max_r_cfg = _get_lidar_ranges(lidar, default_min=0.2, default_max=50.0)
    if max_distance is None:
        max_distance = float(max_r_cfg)
    max_d = float(max_distance)
    min_r = float(min_r_cfg)

    env_ids = torch.arange(env.num_envs, device=env.device)
    pc, _ = _get_downsampled_pc_torch(env, lidar, env_ids, max_pts=max_vis_points)
    if pc is None:
        return torch.zeros(out_shape, device=env.device, dtype=torch.float32)

    x = pc[..., 0]
    y = pc[..., 1]
    z = pc[..., 2]

    valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z)

    r = torch.sqrt(x * x + y * y + z * z + 1e-12)

    # reject near-zero / out-of-range points
    valid = valid & (r > (min_r + 1e-3)) & (r <= (max_d + 1e-3))

    cos_theta = torch.clamp(z / r, -1.0, 1.0)
    theta = torch.rad2deg(torch.acos(cos_theta))

    phi = torch.rad2deg(torch.atan2(y, x))
    phi = torch.remainder(phi, 360.0)

    in_theta = (theta >= theta_min) & (theta < theta_max)
    in_phi = (phi >= phi_min) & (phi < phi_max)
    m = valid & in_theta & in_phi

    num_bins = T * Pn

    min_dist = torch.full(
        (env.num_envs, num_bins),
        float("inf"),
        device=env.device,
        dtype=torch.float32,
    )

    if m.any():
        t_idx = torch.floor((theta - theta_min) / delta_theta).to(torch.long)
        p_idx = torch.floor((phi - phi_min) / delta_phi).to(torch.long)

        t_idx = torch.clamp(t_idx, 0, T - 1)
        p_idx = torch.clamp(p_idx, 0, Pn - 1)

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
    closeness = closeness.to(torch.float32)

    # strict clamp
    return _clamp_01(closeness)
