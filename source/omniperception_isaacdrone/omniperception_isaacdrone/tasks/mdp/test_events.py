"""Reset events for drone start states, goals, and global obstacles."""

from __future__ import annotations

import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# -----------------------------------------------------------------------------
# Obstacle sampling helpers
# -----------------------------------------------------------------------------
def _sample_obstacle_positions(
    device: torch.device,
    num_active: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_height: float,
    edge_margin: float,
    spawn_half_size: float,
    spawn_edge_clearance: float,
    min_separation: float,
    max_sample_tries: int,
) -> torch.Tensor:
    """Sample obstacle centers with soft spacing constraints.

    The sampler prefers layouts that are not clustered and stay away from the
    boundary spawn/goal belt. If the requested density is too high, it
    gradually relaxes the separation constraint instead of failing.
    """

    if num_active <= 0:
        return torch.zeros((0, 3), device=device, dtype=torch.float32)

    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])
    edge_margin = max(float(edge_margin), 0.0)

    inner_x_min = min(x_max, x_min + edge_margin)
    inner_x_max = max(inner_x_min, x_max - edge_margin)
    inner_y_min = min(y_max, y_min + edge_margin)
    inner_y_max = max(inner_y_min, y_max - edge_margin)

    positions_xy = torch.empty((num_active, 2), device=device, dtype=torch.float32)
    placed = 0
    tries = 0
    total_try_budget = max(int(max_sample_tries), num_active * 32)
    base_min_separation = max(float(min_separation), 0.0)
    spawn_half_size = abs(float(spawn_half_size))
    spawn_edge_clearance = max(float(spawn_edge_clearance), 0.0)

    def _near_spawn_edge(candidate_xy: torch.Tensor) -> bool:
        if spawn_half_size <= 0.0 or spawn_edge_clearance <= 0.0:
            return False

        x = float(candidate_xy[0].item())
        y = float(candidate_xy[1].item())
        inside_spawn_y_span = abs(y) <= spawn_half_size + spawn_edge_clearance
        inside_spawn_x_span = abs(x) <= spawn_half_size + spawn_edge_clearance
        near_vertical_spawn_edge = abs(abs(x) - spawn_half_size) < spawn_edge_clearance and inside_spawn_y_span
        near_horizontal_spawn_edge = abs(abs(y) - spawn_half_size) < spawn_edge_clearance and inside_spawn_x_span
        return bool(near_vertical_spawn_edge or near_horizontal_spawn_edge)

    while placed < num_active and tries < total_try_budget:
        candidate = torch.tensor(
            [
                torch.empty((), device=device).uniform_(inner_x_min, inner_x_max).item(),
                torch.empty((), device=device).uniform_(inner_y_min, inner_y_max).item(),
            ],
            device=device,
            dtype=torch.float32,
        )

        tries += 1
        if _near_spawn_edge(candidate):
            continue

        if placed == 0 or base_min_separation <= 0.0:
            positions_xy[placed] = candidate
            placed += 1
            continue

        progress = tries / float(total_try_budget)
        relaxed_min_separation = base_min_separation * max(0.35, 1.0 - 0.7 * progress)
        dist = torch.norm(positions_xy[:placed] - candidate.unsqueeze(0), dim=1)
        if bool((dist >= relaxed_min_separation).all()):
            positions_xy[placed] = candidate
            placed += 1

    if placed < num_active:
        remaining = num_active - placed
        fallback_xy = torch.empty((remaining, 2), device=device, dtype=torch.float32)
        filled = 0
        fallback_tries = 0
        fallback_budget = max(remaining * 128, 1024)
        while filled < remaining and fallback_tries < fallback_budget:
            candidate = torch.tensor(
                [
                    torch.empty((), device=device).uniform_(inner_x_min, inner_x_max).item(),
                    torch.empty((), device=device).uniform_(inner_y_min, inner_y_max).item(),
                ],
                device=device,
                dtype=torch.float32,
            )
            fallback_tries += 1
            if _near_spawn_edge(candidate):
                continue
            fallback_xy[filled] = candidate
            filled += 1
        if filled < remaining:
            xs = torch.linspace(inner_x_min, inner_x_max, steps=24, device=device)
            ys = torch.linspace(inner_y_min, inner_y_max, steps=24, device=device)
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
            safe_grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
            keep = torch.tensor(
                [not _near_spawn_edge(point) for point in safe_grid],
                device=device,
                dtype=torch.bool,
            )
            safe_grid = safe_grid[keep]
            if safe_grid.numel() == 0:
                safe_grid = torch.zeros((1, 2), device=device, dtype=torch.float32)
            need = remaining - filled
            repeat = (need + safe_grid.shape[0] - 1) // safe_grid.shape[0]
            fallback_xy[filled:] = safe_grid.repeat((repeat, 1))[:need]
        positions_xy[placed:] = fallback_xy

    positions = torch.zeros((num_active, 3), device=device, dtype=torch.float32)
    positions[:, :2] = positions_xy
    positions[:, 2] = float(z_height) / 2.0
    return positions


# -----------------------------------------------------------------------------
# Occupancy-map helpers
# -----------------------------------------------------------------------------
def _as_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(bounds[0]), float(bounds[1])
    return (lo, hi) if lo <= hi else (hi, lo)


def _resolve_workspace_bounds(
    env: ManagerBasedRLEnv,
    workspace_x_bounds: tuple[float, float] | None = None,
    workspace_y_bounds: tuple[float, float] | None = None,
    workspace_z_bounds: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Resolve workspace bounds from event params, then env normalization cfg."""
    if workspace_x_bounds is not None and workspace_y_bounds is not None and workspace_z_bounds is not None:
        return (_as_bounds(workspace_x_bounds), _as_bounds(workspace_y_bounds), _as_bounds(workspace_z_bounds))

    norm_cfg = getattr(getattr(env, "cfg", None), "normalization", None)
    cfg_x = getattr(norm_cfg, "x_bounds", None) if norm_cfg is not None else None
    cfg_y = getattr(norm_cfg, "y_bounds", None) if norm_cfg is not None else None
    cfg_z = getattr(norm_cfg, "z_bounds", None) if norm_cfg is not None else None
    xb = workspace_x_bounds if workspace_x_bounds is not None else (cfg_x or (-80.0, 80.0))
    yb = workspace_y_bounds if workspace_y_bounds is not None else (cfg_y or (-80.0, 80.0))
    zb = workspace_z_bounds if workspace_z_bounds is not None else (cfg_z or (0.0, 10.0))
    return (_as_bounds(xb), _as_bounds(yb), _as_bounds(zb))


def _make_empty_occupancy_map(
    device: torch.device,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    resolution: float,
) -> torch.Tensor:
    resolution = max(float(resolution), 1.0e-6)
    shape = tuple(max(int(math.ceil((axis[1] - axis[0]) / resolution)), 1) for axis in bounds)
    return torch.zeros(shape, dtype=torch.bool, device=device)


def _axis_index_range(
    lower: float,
    upper: float,
    axis_bounds: tuple[float, float],
    resolution: float,
    axis_size: int,
) -> tuple[int, int] | None:
    bound_min, bound_max = float(axis_bounds[0]), float(axis_bounds[1])
    lower = max(float(lower), bound_min)
    upper = min(float(upper), bound_max)
    if upper < bound_min or lower > bound_max or upper < lower:
        return None
    i0 = int(math.floor((lower - bound_min) / resolution))
    i1 = int(math.ceil((upper - bound_min) / resolution)) - 1
    i0 = max(0, min(axis_size - 1, i0))
    i1 = max(0, min(axis_size - 1, i1))
    if i1 < i0:
        return None
    return i0, i1


def _build_obstacle_occupancy_map(
    env: ManagerBasedRLEnv,
    obstacle_positions: torch.Tensor,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    resolution: float,
    safe_clearance: float,
    obstacle_size_xy: float,
    obstacle_height: float,
) -> torch.Tensor:
    """Build a 3D occupancy grid inflated by safe_clearance around active obstacles."""
    resolution = max(float(resolution), 1.0e-6)
    safe_clearance = max(float(safe_clearance), 0.0)
    half_xy = max(float(obstacle_size_xy), 0.0) * 0.5 + safe_clearance
    half_z = max(float(obstacle_height), 0.0) * 0.5 + safe_clearance
    occupancy = _make_empty_occupancy_map(env.device, bounds, resolution)

    if obstacle_positions.numel() == 0:
        return occupancy

    centers = obstacle_positions.detach().to(device=env.device, dtype=torch.float32)
    sx, sy, sz = occupancy.shape
    for center in centers:
        cx, cy, cz = float(center[0].item()), float(center[1].item()), float(center[2].item())
        ix = _axis_index_range(cx - half_xy, cx + half_xy, bounds[0], resolution, sx)
        iy = _axis_index_range(cy - half_xy, cy + half_xy, bounds[1], resolution, sy)
        iz = _axis_index_range(cz - half_z, cz + half_z, bounds[2], resolution, sz)
        if ix is None or iy is None or iz is None:
            continue
        occupancy[ix[0] : ix[1] + 1, iy[0] : iy[1] + 1, iz[0] : iz[1] + 1] = True
    return occupancy


def _store_obstacle_occupancy_cache(
    env: ManagerBasedRLEnv,
    obstacle_positions: torch.Tensor,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    resolution: float,
    safe_clearance: float,
    obstacle_size_xy: float,
    obstacle_height: float,
) -> None:
    u = env.unwrapped
    occupancy = _build_obstacle_occupancy_map(
        env=env,
        obstacle_positions=obstacle_positions,
        bounds=bounds,
        resolution=resolution,
        safe_clearance=safe_clearance,
        obstacle_size_xy=obstacle_size_xy,
        obstacle_height=obstacle_height,
    )
    u._obstacle_occupancy_map = occupancy
    u._obstacle_occupancy_bounds = bounds
    u._obstacle_occupancy_resolution = float(resolution)
    u._obstacle_occupancy_safe_clearance = float(safe_clearance)
    u._obstacle_occupancy_obstacle_size_xy = float(obstacle_size_xy)
    u._obstacle_occupancy_obstacle_height = float(obstacle_height)
    u._obstacle_occupancy_active_count = int(obstacle_positions.shape[0])
    u._obstacle_centers_w = obstacle_positions.detach().clone()
    u._obstacle_occupancy_version = int(getattr(u, "_obstacle_occupancy_version", 0)) + 1
    u._obstacle_occupancy_free_ratio = float((~occupancy).float().mean().item())


def _points_occupied_by_map(env: ManagerBasedRLEnv, points_w: torch.Tensor) -> torch.Tensor:
    """Return True for points inside inflated obstacle cells or outside the workspace."""
    points_w = points_w.reshape(-1, 3)
    if points_w.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=env.device)

    u = env.unwrapped
    occupancy = getattr(u, "_obstacle_occupancy_map", None)
    if not isinstance(occupancy, torch.Tensor):
        return torch.zeros((points_w.shape[0],), dtype=torch.bool, device=points_w.device)

    bounds = getattr(u, "_obstacle_occupancy_bounds", None)
    if bounds is None:
        return torch.zeros((points_w.shape[0],), dtype=torch.bool, device=points_w.device)

    resolution = float(getattr(u, "_obstacle_occupancy_resolution", 0.5))
    mins = torch.tensor([bounds[0][0], bounds[1][0], bounds[2][0]], device=points_w.device, dtype=points_w.dtype)
    maxs = torch.tensor([bounds[0][1], bounds[1][1], bounds[2][1]], device=points_w.device, dtype=points_w.dtype)
    inside = ((points_w >= mins) & (points_w < maxs)).all(dim=1)

    occupied = torch.ones((points_w.shape[0],), dtype=torch.bool, device=points_w.device)
    if bool(inside.any()):
        occ = occupancy.to(points_w.device) if occupancy.device != points_w.device else occupancy
        idx = torch.floor((points_w[inside] - mins) / resolution).to(dtype=torch.long)
        max_idx = torch.tensor(occ.shape, dtype=torch.long, device=points_w.device) - 1
        idx = torch.maximum(torch.zeros_like(idx), torch.minimum(idx, max_idx))
        occupied[inside] = occ[idx[:, 0], idx[:, 1], idx[:, 2]]
    return occupied


def _nearest_free_point_from_map(
    env: ManagerBasedRLEnv,
    desired_w: torch.Tensor,
    edge_half_size: float | None = None,
) -> tuple[torch.Tensor, bool]:
    """Find the nearest free grid-cell center at desired z; returns failed=True if none exists."""
    u = env.unwrapped
    occupancy = getattr(u, "_obstacle_occupancy_map", None)
    bounds = getattr(u, "_obstacle_occupancy_bounds", None)
    if not isinstance(occupancy, torch.Tensor) or bounds is None:
        return desired_w.detach().clone(), False

    occ = occupancy.to(desired_w.device) if occupancy.device != desired_w.device else occupancy
    resolution = float(getattr(u, "_obstacle_occupancy_resolution", 0.5))
    z_min, z_max = float(bounds[2][0]), float(bounds[2][1])
    z = min(max(float(desired_w[2].item()), z_min + 0.5 * resolution), z_max - 0.5 * resolution)
    z_idx = int(math.floor((z - z_min) / resolution))
    z_idx = max(0, min(occ.shape[2] - 1, z_idx))

    free_xy = ~occ[:, :, z_idx]
    if edge_half_size is not None:
        xs = (
            torch.arange(occ.shape[0], device=desired_w.device, dtype=torch.float32) + 0.5
        ) * resolution + float(bounds[0][0])
        ys = (
            torch.arange(occ.shape[1], device=desired_w.device, dtype=torch.float32) + 0.5
        ) * resolution + float(bounds[1][0])
        half = abs(float(edge_half_size))
        band = 0.5 * resolution + 1.0e-6
        x_span = torch.abs(xs) <= half + band
        y_span = torch.abs(ys) <= half + band
        x_edge = torch.abs(torch.abs(xs) - half) <= band
        y_edge = torch.abs(torch.abs(ys) - half) <= band
        edge_mask = (x_edge[:, None] & y_span[None, :]) | (x_span[:, None] & y_edge[None, :])
        edge_free_xy = free_xy & edge_mask
        if bool(edge_free_xy.any()):
            free_xy = edge_free_xy

    free_idx = torch.nonzero(free_xy, as_tuple=False)
    if free_idx.numel() == 0:
        return desired_w.detach().clone(), True

    centers_x = (free_idx[:, 0].to(torch.float32) + 0.5) * resolution + float(bounds[0][0])
    centers_y = (free_idx[:, 1].to(torch.float32) + 0.5) * resolution + float(bounds[1][0])
    desired_xy = desired_w[:2].to(dtype=torch.float32)
    dist2 = (centers_x - desired_xy[0]).square() + (centers_y - desired_xy[1]).square()
    best = int(torch.argmin(dist2).item())
    point = torch.tensor(
        [float(centers_x[best].item()), float(centers_y[best].item()), z],
        device=desired_w.device,
        dtype=torch.float32,
    )
    return point, False


def _sample_square_edge_position(
    device: torch.device,
    square_half_size: float,
    z_range: tuple[float, float],
) -> torch.Tensor:
    square_half_size = abs(float(square_half_size))
    z_min, z_max = _as_bounds(z_range)
    edge = int(torch.randint(0, 4, (1,), device=device).item())
    edge_pos = float((torch.rand((), device=device) * 2.0 - 1.0).item()) * square_half_size
    z = float((torch.rand((), device=device) * (z_max - z_min) + z_min).item())

    pos = torch.zeros((3,), device=device, dtype=torch.float32)
    if edge == 0:
        pos[0], pos[1] = -square_half_size, edge_pos
    elif edge == 1:
        pos[0], pos[1] = square_half_size, edge_pos
    elif edge == 2:
        pos[0], pos[1] = edge_pos, -square_half_size
    else:
        pos[0], pos[1] = edge_pos, square_half_size
    pos[2] = z
    return pos


def _sample_safe_square_edge_position(
    env: ManagerBasedRLEnv,
    square_half_size: float,
    z_range: tuple[float, float],
    max_tries: int,
) -> tuple[torch.Tensor, int, bool]:
    max_tries = max(int(max_tries), 1)
    fallback = torch.zeros((3,), device=env.device, dtype=torch.float32)
    for attempt in range(max_tries):
        candidate = _sample_square_edge_position(env.device, square_half_size, z_range)
        fallback = candidate
        if not bool(_points_occupied_by_map(env, candidate.unsqueeze(0))[0].item()):
            return candidate, attempt, False
    nearest, failed = _nearest_free_point_from_map(env, fallback, edge_half_size=square_half_size)
    return nearest, max_tries, failed


def _sample_safe_goal_position(
    env: ManagerBasedRLEnv,
    start_position: torch.Tensor,
    goal_z: float,
    goal_noise_xy: float,
    max_tries: int,
) -> tuple[torch.Tensor, int, bool]:
    max_tries = max(int(max_tries), 1)
    goal_noise_xy = max(float(goal_noise_xy), 0.0)
    target = torch.tensor(
        [-float(start_position[0].item()), -float(start_position[1].item()), float(goal_z)],
        device=env.device,
        dtype=torch.float32,
    )

    fallback = target.clone()
    for attempt in range(max_tries):
        candidate = target.clone()
        if goal_noise_xy > 0.0:
            noise = (torch.rand((2,), device=env.device) * 2.0 - 1.0) * goal_noise_xy
            candidate[:2] += noise
        fallback = candidate
        if not bool(_points_occupied_by_map(env, candidate.unsqueeze(0))[0].item()):
            return candidate, attempt, False
    nearest, failed = _nearest_free_point_from_map(env, fallback, edge_half_size=None)
    return nearest, max_tries, failed


# -----------------------------------------------------------------------------
# Robot and goal reset event
# -----------------------------------------------------------------------------
def reset_root_state_on_square_edge(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    square_half_size: float = 35.0,
    z_range: tuple = (3.0, 7.0),
    goal_z: float = 5.0,
    goal_noise_xy: float = 5.0,
    max_reset_sample_tries: int = 128,
):
    """将无人机 root pose 初始化到 square 边界，并避开膨胀后的障碍物占用图。"""
    asset = env.scene[asset_cfg.name]
    num_resets = len(env_ids)

    positions = torch.zeros((num_resets, 3), device=env.device)
    start_tries = torch.zeros((num_resets,), dtype=torch.float32, device=env.device)
    start_fallbacks = torch.zeros((num_resets,), dtype=torch.float32, device=env.device)

    for i in range(num_resets):
        pos, tries, used_fallback = _sample_safe_square_edge_position(
            env=env,
            square_half_size=square_half_size,
            z_range=z_range,
            max_tries=max_reset_sample_tries,
        )
        positions[i] = pos
        start_tries[i] = float(tries)
        start_fallbacks[i] = float(used_fallback)

    orientations = torch.zeros((num_resets, 4), device=env.device)
    orientations[:, 0] = 1.0  # w

    root_states = torch.cat([positions, orientations], dim=1)
    asset.write_root_pose_to_sim(root_states, env_ids=env_ids)

    velocities = torch.zeros((num_resets, 6), device=env.device)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    if hasattr(env.unwrapped, "goal_pos_w"):
        goals = torch.zeros((num_resets, 3), dtype=torch.float32, device=env.device)
        goal_tries = torch.zeros((num_resets,), dtype=torch.float32, device=env.device)
        goal_fallbacks = torch.zeros((num_resets,), dtype=torch.float32, device=env.device)
        for i in range(num_resets):
            goal, tries, used_fallback = _sample_safe_goal_position(
                env=env,
                start_position=positions[i],
                goal_z=goal_z,
                goal_noise_xy=goal_noise_xy,
                max_tries=max_reset_sample_tries,
            )
            goals[i] = goal
            goal_tries[i] = float(tries)
            goal_fallbacks[i] = float(used_fallback)

        env.unwrapped.goal_pos_w[env_ids] = goals
        env.unwrapped._reset_occupancy_start_tries = start_tries.detach()
        env.unwrapped._reset_occupancy_start_fallbacks = start_fallbacks.detach()
        env.unwrapped._reset_occupancy_goal_tries = goal_tries.detach()
        env.unwrapped._reset_occupancy_goal_fallbacks = goal_fallbacks.detach()

        if hasattr(env.unwrapped, "_update_goal_visualizers"):
            env.unwrapped._update_goal_visualizers(env_ids)


# -----------------------------------------------------------------------------
# Global obstacle layout event
# -----------------------------------------------------------------------------
def randomize_obstacles_on_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    x_range: tuple = (-33.0, 33.0),
    y_range: tuple = (-33.0, 33.0),
    workspace_x_bounds: tuple = (-80.0, 80.0),
    workspace_y_bounds: tuple = (-80.0, 80.0),
    workspace_z_bounds: tuple = (0.0, 10.0),
    z_height: float = 10.0,
    obstacle_size_xy: float = 1.0,
    edge_margin: float = 8.0,
    spawn_half_size: float = 35.0,
    spawn_edge_clearance: float = 6.0,
    min_separation: float = 3.5,
    max_sample_tries: int = 4000,
    occupancy_resolution: float = 0.5,
    safe_clearance: float = 2.0,
):
    """
    每次 level 变化时，根据当前 Curriculum 等级重新随机放置全局障碍物。

    设计原则
    --------
    * 障碍物是全局共享资产（/World/Obstacles/obj_*），与 env 无关，
      因此每次只要有任意 env reset，就整体重新布置一次即可。
    * 只有在 obstacle_level_changed=True（晋级）、显式刷新标志或障碍物数量变化时，
      才真正重排障碍物，其余 reset 继续沿用当前布置。
    * 读取障碍物数量统一使用 curr_obstacle_count（由 Curriculum 维护）。
    """
    u = env.unwrapped

    # ------------------------------------------------------------------ #
    # 1. 初始化保护：首次运行时确保状态字段存在                            #
    # ------------------------------------------------------------------ #
    if not hasattr(u, "obstacle_level_changed"):
        u.obstacle_level_changed = True          # 触发首次布置

    if not hasattr(u, "curr_obstacle_count"):
        # Curriculum 尚未初始化（极早期 reset），从 cfg 读取 level 0 数量
        try:
            levels = env.cfg.curriculum.obstacle_count.params.get("levels", (0,))
            u.curr_obstacle_count = int(levels[0])
        except Exception:
            u.curr_obstacle_count = 0

    # ------------------------------------------------------------------ #
    # 2. 解析全局资产的 max_obstacles                                      #
    # ------------------------------------------------------------------ #
    asset = env.scene[asset_cfg.name]
    state_shape = asset.data.default_root_state.shape
    # torch.Size 没有 .ndim，统一用 len() 判断维度数
    ndim = len(state_shape)
    if ndim == 2:
        max_obstacles = int(state_shape[0])
    elif ndim == 3:
        max_obstacles = int(state_shape[1])
    else:
        print(
            f"[WARN] randomize_obstacles_on_reset: unexpected state shape {state_shape}",
            flush=True,
        )
        return

    # ------------------------------------------------------------------ #
    # 3. 确定本次应激活的障碍物数量                                        #
    # ------------------------------------------------------------------ #
    num_requested = int(u.curr_obstacle_count)
    num_active = min(num_requested, max_obstacles)
    workspace_bounds = _resolve_workspace_bounds(
        env=env,
        workspace_x_bounds=workspace_x_bounds,
        workspace_y_bounds=workspace_y_bounds,
        workspace_z_bounds=workspace_z_bounds,
    )

    refresh_required = bool(getattr(u, "obstacle_layout_refresh_required", False))
    level_changed = bool(getattr(u, "obstacle_level_changed", False))
    last_applied_count = getattr(u, "_last_applied_obstacle_count", None)
    count_changed = last_applied_count is None or int(last_applied_count) != num_active
    map_missing = not isinstance(getattr(u, "_obstacle_occupancy_map", None), torch.Tensor)

    # 只在首次、课程升级、障碍物数量变化或占用图缺失时重排；其余 reset 沿用当前布局。
    if not (refresh_required or level_changed or count_changed or map_missing):
        return

    # 消耗标志位（同一 step 内多个 env 同时 reset 也只执行一次）
    u.obstacle_level_changed = False
    u.obstacle_layout_refresh_required = False

    print(
        f"[OBSTACLE] 重排障碍物: active={num_active}/{max_obstacles} "
        f"(curr_obstacle_count={u.curr_obstacle_count}, "
        f"occupancy={occupancy_resolution:.2f}m, clearance={safe_clearance:.2f}m)",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # 5. 构造全量位姿张量                                                   #
    # ------------------------------------------------------------------ #
    # 默认全部隐藏到地下
    positions = torch.zeros((max_obstacles, 3), device=env.device)
    positions[:, 0] = 1000.0
    positions[:, 1] = 1000.0
    positions[:, 2] = -1000.0

    # 激活部分随机分布在 workspace 中
    if num_active > 0:
        sampled_positions = _sample_obstacle_positions(
            device=env.device,
            num_active=num_active,
            x_range=x_range,
            y_range=y_range,
            z_height=z_height,
            edge_margin=edge_margin,
            spawn_half_size=spawn_half_size,
            spawn_edge_clearance=spawn_edge_clearance,
            min_separation=min_separation,
            max_sample_tries=max_sample_tries,
        )
        positions[:num_active] = sampled_positions

    orientations = torch.zeros((max_obstacles, 4), device=env.device)
    orientations[:, 0] = 1.0  # w = 1（单位四元数）

    velocities = torch.zeros((max_obstacles, 6), device=env.device)

    # ------------------------------------------------------------------ #
    # 6. 写入仿真                                                          #
    # 全局资产没有 per-env 概念，env_ids 对应障碍物实例索引                 #
    # 传入所有障碍物的索引以整体更新                                         #
    # ------------------------------------------------------------------ #
    all_obstacle_ids = torch.arange(max_obstacles, dtype=torch.long, device=env.device)
    root_pose = torch.cat([positions, orientations], dim=-1)   # (max_obstacles, 7)

    asset.write_root_pose_to_sim(root_pose, env_ids=all_obstacle_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=all_obstacle_ids)
    _store_obstacle_occupancy_cache(
        env=env,
        obstacle_positions=positions[:num_active],
        bounds=workspace_bounds,
        resolution=occupancy_resolution,
        safe_clearance=safe_clearance,
        obstacle_size_xy=obstacle_size_xy,
        obstacle_height=z_height,
    )
    u._last_applied_obstacle_count = int(num_active)
