# omniperception_isaacdrone/tasks/mdp/test6_observations.py

from __future__ import annotations

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# -----------------------------------------------------------------------------
# goal_state: goal delta in world frame
# -----------------------------------------------------------------------------
def obs_goal_delta(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """goal_state_delta = goal_pos_w - uav_root_pos_w"""
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
    """Read lidar pointcloud (torch) and mask invalid points with NaN.
    Return:
      pc: (E, P, 3) containing NaN for invalid points
      num_raw: (E,) raw point count (before any optional downsample)
    """
    if lidar is None:
        return None, None

    pc = lidar.get_pointcloud(env_ids)
    if pc is None:
        return None, None

    if pc.dim() == 2:
        pc = pc.unsqueeze(0)

    E, P, _ = pc.shape
    num_raw = torch.full((E,), P, device=pc.device, dtype=torch.int32)

    # Optional downsample (disabled by default to match your current behavior)
    # if max_pts is not None and max_pts > 0 and P > max_pts:
    #     step = max(1, P // max_pts)
    #     pc = pc[:, ::step, :]
    #     num_raw = torch.full((E,), pc.shape[1], device=pc.device, dtype=torch.int32)

    finite_mask = torch.isfinite(pc).all(dim=-1)
    pc = pc.clone()
    pc[~finite_mask] = float("nan")

    return pc, num_raw


# -----------------------------------------------------------------------------
# lidar_state: polar grid minimum range (flattened as closeness)
# -----------------------------------------------------------------------------
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
    """lidar_state_grid: (num_envs, T*P) flattened *closeness* in each bin.

    theta: polar angle from +Z axis in degrees [0, 180]
    phi: azimuth angle in degrees [0, 360)
    Each bin stores min range -> converted to closeness:
        closeness = 1 - clamp(min_dist / max_distance, 0, 1)
    Empty bin is treated as min_dist = max_distance -> closeness = 0.

    IMPORTANT robustness fixes:
      - ignore non-finite points
      - ignore points with r <= min_range (e.g. zeros or self hits)
      - clamp normalization by max_distance
    """

    # bin counts
    T = int((theta_max - theta_min) / delta_theta)
    Pn = int((phi_max - phi_min) / delta_phi)
    T = max(T, 1)
    Pn = max(Pn, 1)
    out_shape = (env.num_envs, T * Pn)

    # If no scene/lidar -> output zeros (closeness=0)
    if not hasattr(env, "scene"):
        return torch.zeros(out_shape, device=env.device, dtype=torch.float32)

    try:
        lidar = env.scene[lidar_name]
    except Exception:
        return torch.zeros(out_shape, device=env.device, dtype=torch.float32)

    # ranges
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

    # Store min distance per bin (init inf)
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

        # NOTE: loop over envs for correctness & simplicity (num_envs is small)
        for e in range(env.num_envs):
            me = m[e]
            if me.any():
                idx_e = lin_idx[e, me]
                r_e = r[e, me].to(torch.float32)
                min_dist[e].scatter_reduce_(0, idx_e, r_e, reduce="amin", include_self=True)

    # Empty bin -> treat as max_distance (so closeness=0)
    max_d_t = torch.tensor(max_d, device=env.device, dtype=torch.float32)

    min_dist = torch.where(torch.isfinite(min_dist), min_dist, max_d_t)
    min_dist = torch.clamp(min_dist, 0.0, max_d_t)

    # closeness = 1 - clamp(min_dist / max_distance, 0, 1)
    closeness = 1.0 - torch.clamp(min_dist / max_d_t, 0.0, 1.0)

    return closeness.to(torch.float32)
