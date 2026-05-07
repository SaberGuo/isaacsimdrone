"""Exact scene-geometry LiDAR state for the drone task.

This module intentionally does not read ``LidarSensor.get_pointcloud()``.  It
uses the current obstacle/workspace geometry and the robot pose to build a
body-frame range grid directly from scene geometry.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse


def _get_workspace_bounds(
    env: ManagerBasedRLEnv,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    norm = getattr(getattr(env, "cfg", None), "normalization", None)
    if norm is not None:
        try:
            return (
                tuple(float(v) for v in getattr(norm, "x_bounds")),
                tuple(float(v) for v in getattr(norm, "y_bounds")),
                tuple(float(v) for v in getattr(norm, "z_bounds")),
            )
        except Exception:
            pass
    return (-80.0, 80.0), (-80.0, 80.0), (0.0, 10.0)


def _lidar_bin_center_dirs_body(
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    delta_theta: float,
    delta_phi: float,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    theta_bins = max(int((float(theta_max) - float(theta_min)) / float(delta_theta)), 1)
    phi_bins = max(int((float(phi_max) - float(phi_min)) / float(delta_phi)), 1)

    theta_idx = torch.arange(theta_bins, device=device, dtype=torch.float32)
    phi_idx = torch.arange(phi_bins, device=device, dtype=torch.float32)
    theta = torch.deg2rad(float(theta_min) + (theta_idx + 0.5) * float(delta_theta))
    phi = torch.deg2rad(float(phi_min) + (phi_idx + 0.5) * float(delta_phi))
    tt, pp = torch.meshgrid(theta, phi, indexing="ij")
    dirs = torch.stack(
        [torch.sin(tt) * torch.cos(pp), torch.sin(tt) * torch.sin(pp), torch.cos(tt)],
        dim=-1,
    ).reshape(-1, 3)
    return dirs, theta_bins, phi_bins


def _workspace_boundary_distance_grid(
    env: ManagerBasedRLEnv,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    delta_theta: float,
    delta_phi: float,
    min_range: float,
    max_distance: float,
) -> torch.Tensor:
    """Distance from lidar rays to ground and workspace boundary walls."""
    robot = env.scene["robot"]
    origin_w = robot.data.root_pos_w.to(torch.float32)
    quat_w = robot.data.root_quat_w.to(torch.float32)
    dirs_b, _, _ = _lidar_bin_center_dirs_body(
        theta_min, theta_max, phi_min, phi_max, delta_theta, delta_phi, env.device
    )
    num_bins = int(dirs_b.shape[0])
    out = torch.full((env.num_envs, num_bins), float(max_distance), device=env.device, dtype=torch.float32)

    dirs_expand = dirs_b.unsqueeze(0).expand(env.num_envs, -1, -1)
    quat_expand = quat_w.unsqueeze(1).expand(-1, num_bins, -1)
    dirs_w = quat_apply(quat_expand.reshape(-1, 4), dirs_expand.reshape(-1, 3)).view(env.num_envs, num_bins, 3)
    xb, yb, zb = _get_workspace_bounds(env)
    bounds = (xb, yb, zb)
    eps = 1.0e-6
    tol = 1.0e-4

    for axis, axis_bounds in enumerate(bounds):
        other_axes = [i for i in range(3) if i != axis]
        for plane_value in axis_bounds:
            denom = dirs_w[:, :, axis]
            valid = denom.abs() > eps
            t = (float(plane_value) - origin_w[:, axis].unsqueeze(1)) / torch.where(
                valid, denom, torch.ones_like(denom)
            )
            valid &= (t >= float(min_range)) & (t <= float(max_distance))
            hit = origin_w.unsqueeze(1) + dirs_w * t.unsqueeze(-1)
            for other_axis in other_axes:
                lo, hi = bounds[other_axis]
                valid &= (hit[:, :, other_axis] >= float(lo) - tol) & (hit[:, :, other_axis] <= float(hi) + tol)
            out = torch.minimum(out, torch.where(valid, t.to(torch.float32), out))
    return out


def _make_workspace_surface_points(
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    surface_step: float,
    device: torch.device,
) -> torch.Tensor:
    xb, yb, zb = bounds
    step = max(float(surface_step), 0.05)
    xs = torch.arange(float(xb[0]), float(xb[1]) + 0.5 * step, step, device=device, dtype=torch.float32)
    ys = torch.arange(float(yb[0]), float(yb[1]) + 0.5 * step, step, device=device, dtype=torch.float32)
    zs = torch.arange(float(zb[0]), float(zb[1]) + 0.5 * step, step, device=device, dtype=torch.float32)

    surfaces = []
    yy, zz = torch.meshgrid(ys, zs, indexing="ij")
    surfaces.append(torch.stack([torch.full_like(yy, float(xb[0])), yy, zz], dim=-1).reshape(-1, 3))
    surfaces.append(torch.stack([torch.full_like(yy, float(xb[1])), yy, zz], dim=-1).reshape(-1, 3))

    xx, zz = torch.meshgrid(xs, zs, indexing="ij")
    surfaces.append(torch.stack([xx, torch.full_like(xx, float(yb[0])), zz], dim=-1).reshape(-1, 3))
    surfaces.append(torch.stack([xx, torch.full_like(xx, float(yb[1])), zz], dim=-1).reshape(-1, 3))

    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    surfaces.append(torch.stack([xx, yy, torch.full_like(xx, float(zb[0]))], dim=-1).reshape(-1, 3))
    surfaces.append(torch.stack([xx, yy, torch.full_like(xx, float(zb[1]))], dim=-1).reshape(-1, 3))
    return torch.cat(surfaces, dim=0)


def _get_workspace_surface_points_cached(
    env: ManagerBasedRLEnv,
    surface_step: float,
) -> torch.Tensor:
    bounds = _get_workspace_bounds(env)
    key = (
        tuple(round(float(v), 6) for pair in bounds for v in pair),
        round(float(surface_step), 6),
        str(env.device),
    )
    cache = getattr(env, "_exact_lidar_workspace_surface_cache", None)
    if isinstance(cache, dict) and cache.get("key") == key and isinstance(cache.get("points"), torch.Tensor):
        return cache["points"]
    points = _make_workspace_surface_points(bounds, surface_step=surface_step, device=env.device)
    env._exact_lidar_workspace_surface_cache = {"key": key, "points": points}
    return points


def _workspace_boundary_pointcloud_body(
    env: ManagerBasedRLEnv,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    delta_theta: float,
    delta_phi: float,
    min_range: float,
    max_distance: float,
    surface_step: float = 0.5,
) -> list[torch.Tensor]:
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w.to(torch.float32)
    robot_quat_w = robot.data.root_quat_w.to(torch.float32)
    surface_w = _get_workspace_surface_points_cached(env, surface_step=surface_step)
    clouds: list[torch.Tensor] = []
    for env_id in range(int(env.num_envs)):
        rel_w = surface_w - robot_pos_w[env_id].unsqueeze(0)
        quat = robot_quat_w[env_id].unsqueeze(0).expand(rel_w.shape[0], -1)
        rel_b = quat_apply_inverse(quat, rel_w)
        ranges = torch.linalg.norm(rel_b, dim=-1)
        safe_r = torch.clamp(ranges, min=1.0e-12)
        theta = torch.rad2deg(torch.acos(torch.clamp(rel_b[:, 2] / safe_r, -1.0, 1.0)))
        phi = torch.remainder(torch.rad2deg(torch.atan2(rel_b[:, 1], rel_b[:, 0])), 360.0)
        valid = torch.isfinite(rel_b).all(dim=-1)
        valid &= (ranges >= float(min_range)) & (ranges <= float(max_distance))
        valid &= (theta >= float(theta_min)) & (theta < float(theta_max))
        valid &= (phi >= float(phi_min)) & (phi < float(phi_max))
        clouds.append(rel_b[valid].contiguous())
    return clouds


def _make_cuboid_surface_points(
    size_xy: float,
    height: float,
    surface_step: float,
    device: torch.device,
) -> torch.Tensor:
    half_x = 0.5 * float(size_xy)
    half_y = 0.5 * float(size_xy)
    half_z = 0.5 * float(height)
    step = max(float(surface_step), 0.02)

    xs = torch.arange(-half_x, half_x + 0.5 * step, step, device=device, dtype=torch.float32)
    ys = torch.arange(-half_y, half_y + 0.5 * step, step, device=device, dtype=torch.float32)
    zs = torch.arange(-half_z, half_z + 0.5 * step, step, device=device, dtype=torch.float32)

    faces = []
    yy, zz = torch.meshgrid(ys, zs, indexing="ij")
    faces.append(torch.stack([torch.full_like(yy, half_x), yy, zz], dim=-1).reshape(-1, 3))
    faces.append(torch.stack([torch.full_like(yy, -half_x), yy, zz], dim=-1).reshape(-1, 3))

    xx, zz = torch.meshgrid(xs, zs, indexing="ij")
    faces.append(torch.stack([xx, torch.full_like(xx, half_y), zz], dim=-1).reshape(-1, 3))
    faces.append(torch.stack([xx, torch.full_like(xx, -half_y), zz], dim=-1).reshape(-1, 3))

    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    faces.append(torch.stack([xx, yy, torch.full_like(xx, half_z)], dim=-1).reshape(-1, 3))
    faces.append(torch.stack([xx, yy, torch.full_like(xx, -half_z)], dim=-1).reshape(-1, 3))
    return torch.cat(faces, dim=0)


def _get_surface_points_cached(
    env: ManagerBasedRLEnv,
    obstacle_size_xy: float,
    obstacle_height: float,
    surface_step: float,
) -> torch.Tensor:
    key = (
        round(float(obstacle_size_xy), 6),
        round(float(obstacle_height), 6),
        round(float(surface_step), 6),
        str(env.device),
    )
    cache = getattr(env, "_exact_lidar_surface_cache", None)
    if isinstance(cache, dict) and cache.get("key") == key and isinstance(cache.get("points"), torch.Tensor):
        return cache["points"]
    points = _make_cuboid_surface_points(
        size_xy=obstacle_size_xy,
        height=obstacle_height,
        surface_step=surface_step,
        device=env.device,
    )
    env._exact_lidar_surface_cache = {"key": key, "points": points}
    return points


def _active_obstacle_pose(env: ManagerBasedRLEnv, hidden_abs_threshold: float = 500.0) -> tuple[torch.Tensor, torch.Tensor]:
    obstacles = env.scene["obstacles"]
    pos = obstacles.data.root_pos_w.to(torch.float32)
    quat = obstacles.data.root_quat_w.to(torch.float32)
    if pos.dim() == 3:
        pos = pos.reshape(-1, 3)
        quat = quat.reshape(-1, 4)
    active = torch.isfinite(pos).all(dim=-1) & (torch.max(torch.abs(pos), dim=-1).values < float(hidden_abs_threshold))
    return pos[active], quat[active]


def exact_obstacle_pointcloud_body(
    env: ManagerBasedRLEnv,
    obstacle_size_xy: float = 1.0,
    obstacle_height: float = 10.0,
    surface_step: float = 0.5,
    min_range: float = 0.2,
    max_distance: float = 50.0,
    include_workspace: bool = True,
    theta_min: float = 75.0,
    theta_max: float = 105.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 30.0,
    delta_phi: float = 15.0,
) -> list[torch.Tensor]:
    """Return per-env scene points in the robot body frame."""
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w.to(torch.float32)
    robot_quat_w = robot.data.root_quat_w.to(torch.float32)

    workspace_clouds = (
        _workspace_boundary_pointcloud_body(
            env,
            theta_min=theta_min,
            theta_max=theta_max,
            phi_min=phi_min,
            phi_max=phi_max,
            delta_theta=delta_theta,
            delta_phi=delta_phi,
            min_range=min_range,
            max_distance=max_distance,
            surface_step=surface_step,
        )
        if bool(include_workspace)
        else None
    )

    obstacle_pos_w, obstacle_quat_w = _active_obstacle_pose(env)
    if obstacle_pos_w.numel() == 0:
        if workspace_clouds is not None:
            return workspace_clouds
        return [torch.empty((0, 3), device=env.device, dtype=torch.float32) for _ in range(env.num_envs)]

    surface = _get_surface_points_cached(env, obstacle_size_xy, obstacle_height, surface_step)
    num_obs = int(obstacle_pos_w.shape[0])
    surface_expanded = surface.unsqueeze(0).expand(num_obs, -1, -1)
    quat_expanded = obstacle_quat_w.unsqueeze(1).expand(-1, surface.shape[0], -1)
    points_w = quat_apply(quat_expanded.reshape(-1, 4), surface_expanded.reshape(-1, 3)).view(num_obs, -1, 3)
    points_w = (points_w + obstacle_pos_w.unsqueeze(1)).reshape(-1, 3)

    clouds: list[torch.Tensor] = []
    for env_id in range(int(env.num_envs)):
        rel_w = points_w - robot_pos_w[env_id].unsqueeze(0)
        quat = robot_quat_w[env_id].unsqueeze(0).expand(rel_w.shape[0], -1)
        rel_b = quat_apply_inverse(quat, rel_w)
        ranges = torch.linalg.norm(rel_b, dim=-1)
        valid = (ranges >= float(min_range)) & (ranges <= float(max_distance))
        obstacle_cloud = rel_b[valid].contiguous()
        if workspace_clouds is not None and workspace_clouds[env_id].numel() > 0:
            clouds.append(torch.cat([obstacle_cloud, workspace_clouds[env_id]], dim=0))
        else:
            clouds.append(obstacle_cloud)
    return clouds


def exact_lidar_distance_grid(
    env: ManagerBasedRLEnv,
    theta_min: float = 75.0,
    theta_max: float = 105.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 30.0,
    delta_phi: float = 15.0,
    min_range: float = 0.2,
    max_distance: float = 50.0,
    obstacle_size_xy: float = 1.0,
    obstacle_height: float = 10.0,
    surface_step: float = 0.5,
) -> torch.Tensor:
    """Return an ``(num_envs, theta_bins * phi_bins)`` nearest-distance grid."""
    theta_bins = max(int((float(theta_max) - float(theta_min)) / float(delta_theta)), 1)
    phi_bins = max(int((float(phi_max) - float(phi_min)) / float(delta_phi)), 1)
    num_bins = theta_bins * phi_bins
    out = torch.full((env.num_envs, num_bins), float(max_distance), device=env.device, dtype=torch.float32)

    clouds = exact_obstacle_pointcloud_body(
        env,
        obstacle_size_xy=obstacle_size_xy,
        obstacle_height=obstacle_height,
        surface_step=surface_step,
        min_range=min_range,
        max_distance=max_distance,
        include_workspace=False,
    )
    for env_id, points in enumerate(clouds):
        if points.numel() == 0:
            continue
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = torch.linalg.norm(points, dim=-1)
        valid = torch.isfinite(points).all(dim=-1) & (r >= float(min_range)) & (r <= float(max_distance))
        safe_r = torch.clamp(r, min=1.0e-12)
        theta = torch.rad2deg(torch.acos(torch.clamp(z / safe_r, -1.0, 1.0)))
        phi = torch.remainder(torch.rad2deg(torch.atan2(y, x)), 360.0)
        valid &= (theta >= float(theta_min)) & (theta < float(theta_max))
        valid &= (phi >= float(phi_min)) & (phi < float(phi_max))
        if not valid.any():
            continue
        t_idx = torch.clamp(
            torch.floor((theta[valid] - float(theta_min)) / float(delta_theta)).to(torch.long),
            0,
            theta_bins - 1,
        )
        p_idx = torch.clamp(
            torch.floor((phi[valid] - float(phi_min)) / float(delta_phi)).to(torch.long),
            0,
            phi_bins - 1,
        )
        lin_idx = t_idx * phi_bins + p_idx
        out[env_id].scatter_reduce_(0, lin_idx, r[valid].to(torch.float32), reduce="amin", include_self=True)
    workspace_grid = _workspace_boundary_distance_grid(
        env,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
        delta_theta=delta_theta,
        delta_phi=delta_phi,
        min_range=min_range,
        max_distance=max_distance,
    )
    out = torch.minimum(out, workspace_grid)
    return out


def get_exact_lidar_grid_cached(
    env: ManagerBasedRLEnv,
    theta_min: float = 75.0,
    theta_max: float = 105.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 30.0,
    delta_phi: float = 15.0,
    min_range: float = 0.2,
    max_distance: float = 50.0,
    obstacle_size_xy: float = 1.0,
    obstacle_height: float = 10.0,
    surface_step: float = 0.5,
) -> torch.Tensor:
    current_step = getattr(env, "common_step_counter", -1)
    params = (
        theta_min,
        theta_max,
        phi_min,
        phi_max,
        delta_theta,
        delta_phi,
        min_range,
        max_distance,
        obstacle_size_xy,
        obstacle_height,
        surface_step,
    )
    cache = getattr(env, "_lidar_grid_cache", None)
    if isinstance(cache, dict) and cache.get("step") == current_step and cache.get("params") == params:
        data = cache.get("data")
        if isinstance(data, torch.Tensor):
            return data
    grid = exact_lidar_distance_grid(
        env,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
        delta_theta=delta_theta,
        delta_phi=delta_phi,
        min_range=min_range,
        max_distance=max_distance,
        obstacle_size_xy=obstacle_size_xy,
        obstacle_height=obstacle_height,
        surface_step=surface_step,
    )
    env._lidar_grid_cache = {"step": current_step, "params": params, "data": grid}
    return grid
