# omniperception_isaacdrone/tasks/mdp/test6_rewards.py

from __future__ import annotations

import math
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .test6_observations import obs_lidar_min_range_grid
from .test6_terminations import (
    termination_collision,
    termination_out_of_workspace,
    termination_reached_goal,
)
from .test6_dijkstra_utils import DijkstraNavigator, batched_dijkstra_reward


# -----------------------------------------------------------------------------
# TensorBoard / debug cache helpers (store per-step tensors on env)
# -----------------------------------------------------------------------------
def _tb_get_dict(env: ManagerBasedRLEnv, attr: str) -> dict:
    d = getattr(env, attr, None)
    if d is None or (not isinstance(d, dict)):
        d = {}
        setattr(env, attr, d)
    return d


def _tb_store_reward(env: ManagerBasedRLEnv, name: str, value: torch.Tensor):
    """Store raw reward term output (shape: [N]) for this step."""
    try:
        d = _tb_get_dict(env, "_tb_reward_terms")
        if isinstance(value, torch.Tensor):
            d[name] = value.detach()
    except Exception:
        pass


def _tb_store_aux(env: ManagerBasedRLEnv, name: str, value: torch.Tensor):
    """Store auxiliary debug metrics (shape: [N]) for this step."""
    try:
        d = _tb_get_dict(env, "_tb_aux_terms")
        if isinstance(value, torch.Tensor):
            d[name] = value.detach()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------
def _safe_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=-1) + eps)


def _broadcast_goal(goal: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Make sure goal has shape (N, 3) on the same device as pos."""
    if goal.device != pos.device:
        goal = goal.to(pos.device)

    if goal.shape[0] == 1 and pos.shape[0] > 1:
        goal = goal.expand(pos.shape[0], 3)
    elif goal.shape[0] != pos.shape[0]:
        goal = goal[:1].expand(pos.shape[0], 3)

    return goal


def _get_goal_pos(env: ManagerBasedRLEnv, pos: torch.Tensor) -> torch.Tensor:
    goal = getattr(env, "goal_pos_w", None)
    if goal is None:
        return torch.zeros_like(pos)
    return _broadcast_goal(goal, pos)


def _get_step_dt(env: ManagerBasedRLEnv) -> float:
    if hasattr(env, "step_dt"):
        try:
            return float(env.step_dt)
        except Exception:
            pass
    try:
        return float(env.cfg.sim.dt) * float(env.cfg.decimation)
    except Exception:
        return 1.0 / 60.0


def _get_lidar_max_distance(lidar) -> float:
    try:
        if hasattr(lidar, "cfg") and hasattr(lidar.cfg, "max_distance"):
            return float(lidar.cfg.max_distance)
    except Exception:
        pass
    return 50.0


# -----------------------------------------------------------------------------
# ① goal distance reward (bounded & stable)
# -----------------------------------------------------------------------------
def reward_distance_to_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 6.0) -> torch.Tensor:
    """Bounded goal proximity reward: exp(-0.5*(d/std)^2) in (0, 1]."""
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    goal = _get_goal_pos(env, pos)

    d = _safe_norm(goal - pos)
    std = max(float(std), 1e-6)

    r = d / std
    r2 = torch.clamp(r * r, 0.0, 400.0)
    out = torch.exp(-0.5 * r2)

    _tb_store_reward(env, "dist_to_goal", out)
    _tb_store_aux(env, "goal_distance", d)
    return out


# -----------------------------------------------------------------------------
# dense progress reward (approach goal => positive, go away => negative)
# -----------------------------------------------------------------------------
def reward_progress_to_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    speed_ref: float = 3.0,
    clip: float = 1.0,
) -> torch.Tensor:
    """Dense shaping reward based on distance decrease to the goal.

    delta_d = prev_dist - current_dist
    towards_speed = delta_d / step_dt
    out = clamp(towards_speed / speed_ref, -clip, clip)
    """
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    goal = _get_goal_pos(env, pos)
    d = _safe_norm(goal - pos)

    prev = getattr(env, "_progress_prev_goal_dist", None)
    if prev is None or (not isinstance(prev, torch.Tensor)) or prev.shape != d.shape:
        setattr(env, "_progress_prev_goal_dist", d.detach().clone())
        towards_speed = torch.zeros_like(d)
        out = torch.zeros_like(d)
    else:
        dt = max(_get_step_dt(env), 1e-6)
        delta = prev - d
        towards_speed = delta / dt
        denom = max(float(speed_ref), 1e-6)
        out = torch.clamp(towards_speed / denom, min=-float(clip), max=float(clip))
        env._progress_prev_goal_dist = d.detach().clone()

    out = out.to(torch.float32)
    _tb_store_reward(env, "progress_to_goal", out)
    _tb_store_aux(env, "goal_progress_norm", out)
    _tb_store_aux(env, "goal_progress_speed", towards_speed.to(torch.float32))
    return out


# -----------------------------------------------------------------------------
# height tracking
# -----------------------------------------------------------------------------
def reward_height_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_z: float = 5.0,
    std: float = 2.0,
) -> torch.Tensor:
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    dz = pos[:, 2] - float(target_z)
    std = max(float(std), 1e-6)
    r2 = torch.clamp((dz / std) ** 2, 0.0, 400.0)
    out = torch.exp(-0.5 * r2)

    _tb_store_reward(env, "height", out)
    _tb_store_aux(env, "height_error", dz)
    return out


# -----------------------------------------------------------------------------
# stability reward
# -----------------------------------------------------------------------------
def reward_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_std: float = 2.0,
    ang_std: float = 6.0,
) -> torch.Tensor:
    lin = mdp.base_lin_vel(env, asset_cfg=asset_cfg)
    ang = mdp.base_ang_vel(env, asset_cfg=asset_cfg)

    lin2 = torch.sum(lin * lin, dim=-1)
    ang2 = torch.sum(ang * ang, dim=-1)

    lin_std = max(float(lin_std), 1e-6)
    ang_std = max(float(ang_std), 1e-6)

    rl = torch.exp(-0.5 * torch.clamp(lin2 / (lin_std * lin_std), 0.0, 400.0))
    ra = torch.exp(-0.5 * torch.clamp(ang2 / (ang_std * ang_std), 0.0, 400.0))
    out = rl * ra

    _tb_store_reward(env, "stability", out)
    _tb_store_aux(env, "lin_speed", _safe_norm(lin))
    _tb_store_aux(env, "ang_speed", _safe_norm(ang))
    return out


# -----------------------------------------------------------------------------
# ④ velocity direction reward (towards goal)
# -----------------------------------------------------------------------------
def reward_velocity_towards_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_speed: float = 0.2,
    speed_ref: float = 3.0,
    use_relu: bool = True,
) -> torch.Tensor:
    """Reward the velocity component projected onto the goal direction.

    Compared with the old cos(theta) * speed-factor form, this version directly uses
    the signed projected speed, so the magnitude is easier to interpret and matches
    the physical meaning of "moving towards the goal" more closely.
    """
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    goal = _get_goal_pos(env, pos)

    v = mdp.root_lin_vel_w(env, asset_cfg=asset_cfg)
    speed = _safe_norm(v)

    dir_vec = goal - pos
    dir_norm = _safe_norm(dir_vec)
    dir_unit = dir_vec / (dir_norm.unsqueeze(-1) + 1e-6)

    projected_speed = torch.sum(v * dir_unit, dim=-1)
    cos = torch.clamp(projected_speed / (speed + 1e-6), -1.0, 1.0)

    if use_relu:
        projected_speed = torch.clamp(projected_speed, min=0.0)

    min_speed = float(min_speed)
    if min_speed > 0.0:
        projected_speed = torch.where(speed >= min_speed, projected_speed, torch.zeros_like(projected_speed))

    denom = max(float(speed_ref), 1e-6)
    if use_relu:
        out = torch.clamp(projected_speed / denom, 0.0, 1.0)
    else:
        out = torch.clamp(projected_speed / denom, -1.0, 1.0)

    out = out.to(torch.float32)
    _tb_store_reward(env, "vel_towards_goal", out)
    _tb_store_aux(env, "speed", speed)
    _tb_store_aux(env, "goal_direction_cos", cos)
    _tb_store_aux(env, "towards_speed", projected_speed.to(torch.float32))
    return out


# -----------------------------------------------------------------------------
# ② LiDAR safety penalty
# -----------------------------------------------------------------------------
def penalty_lidar_threat(
    env: ManagerBasedRLEnv,
    lidar_name: str = "lidar",
    safe_dist: float | None = None,
    safe_dist_ratio: float = 0.1,
    exp_scale: float = 1.0,
    cap: float = 5.0,
    use_grid: bool = True,
    threshold: float | None = None,
    theta_min: float = 30.0,
    theta_max: float = 90.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 1.0,
    delta_phi: float = 5.0,
    max_vis_points: int | None = None,
) -> torch.Tensor:
    exp_scale = max(float(exp_scale), 1e-6)
    cap = float(cap)

    out0 = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)

    try:
        lidar = env.scene[lidar_name]
    except Exception:
        _tb_store_reward(env, "lidar_threat", out0)
        return out0

    env_ids = torch.arange(env.num_envs, device=env.device)
    max_d = _get_lidar_max_distance(lidar)

    if safe_dist is None and threshold is not None:
        safe_dist = float(threshold)
    if safe_dist is None:
        safe_dist = float(safe_dist_ratio) * float(max_d)
    safe_dist = float(safe_dist)

    if bool(use_grid):
        grid = obs_lidar_min_range_grid(
            env,
            lidar_name=lidar_name,
            theta_min=theta_min,
            theta_max=theta_max,
            phi_min=phi_min,
            phi_max=phi_max,
            delta_theta=delta_theta,
            delta_phi=delta_phi,
            empty_value=0.0,
            max_vis_points=max_vis_points,
            max_distance=max_d,
        )
        max_close = grid.max(dim=1).values
        min_dist = float(max_d) * (1.0 - max_close)
    else:
        dist = lidar.get_distances(env_ids)
        if dist is None:
            _tb_store_reward(env, "lidar_threat", out0)
            return out0
        if dist.dim() == 1:
            dist = dist.unsqueeze(0)
        dist = dist.to(dtype=torch.float32)
        dist = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, max_d))
        dist = torch.where(dist > 0.0, dist, torch.full_like(dist, max_d))
        min_dist = dist.min(dim=1).values

    delta = safe_dist - min_dist
    x = torch.clamp(delta / exp_scale, min=0.0)
    pen = torch.expm1(x).to(torch.float32)

    if cap > 0.0:
        pen = torch.clamp(pen, 0.0, cap)

    _tb_store_reward(env, "lidar_threat", pen)
    _tb_store_aux(env, "lidar_min_dist", min_dist)
    _tb_store_aux(env, "lidar_safe_dist", torch.full_like(min_dist, safe_dist))
    return pen


# -----------------------------------------------------------------------------
# ③ energy penalty
# -----------------------------------------------------------------------------
def penalty_energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_vel_scale: float = 6.0,
    ang_vel_scale: float = 10.0,
    lin_acc_scale: float = 50.0,
    ang_acc_scale: float = 80.0,
    include_acc: bool = True,
    acc_weight: float = 0.2,
    max_penalty: float = 10.0,
) -> torch.Tensor:
    lin_vel_scale = max(float(lin_vel_scale), 1e-6)
    ang_vel_scale = max(float(ang_vel_scale), 1e-6)
    lin_acc_scale = max(float(lin_acc_scale), 1e-6)
    ang_acc_scale = max(float(ang_acc_scale), 1e-6)
    acc_weight = max(float(acc_weight), 0.0)
    max_penalty = float(max_penalty)

    v = mdp.root_lin_vel_w(env, asset_cfg=asset_cfg)
    w = mdp.root_ang_vel_w(env, asset_cfg=asset_cfg)

    v_norm = _safe_norm(v)
    w_norm = _safe_norm(w)

    pv = (v_norm / lin_vel_scale) ** 2
    pw = (w_norm / ang_vel_scale) ** 2

    pa = torch.zeros_like(pv)
    palpha = torch.zeros_like(pv)

    a_norm = torch.zeros_like(pv)
    alpha_norm = torch.zeros_like(pv)

    if bool(include_acc):
        dt = max(_get_step_dt(env), 1e-6)

        prev_v = getattr(env, "_energy_prev_lin_vel_w", None)
        prev_w = getattr(env, "_energy_prev_ang_vel_w", None)

        if (
            prev_v is None
            or prev_w is None
            or (not isinstance(prev_v, torch.Tensor))
            or (not isinstance(prev_w, torch.Tensor))
            or prev_v.shape != v.shape
            or prev_w.shape != w.shape
        ):
            setattr(env, "_energy_prev_lin_vel_w", v.detach().clone())
            setattr(env, "_energy_prev_ang_vel_w", w.detach().clone())
        else:
            dv = torch.clamp(v - prev_v, min=-10.0 * dt, max=10.0 * dt)
            dw = torch.clamp(w - prev_w, min=-20.0 * dt, max=20.0 * dt)

            a = dv / dt
            alpha = dw / dt

            a_norm = _safe_norm(a)
            alpha_norm = _safe_norm(alpha)

            pa = (a_norm / lin_acc_scale) ** 2
            palpha = (alpha_norm / ang_acc_scale) ** 2

            env._energy_prev_lin_vel_w = v.detach().clone()
            env._energy_prev_ang_vel_w = w.detach().clone()

    pen = pv + pw + acc_weight * (pa + palpha)

    if max_penalty > 0.0:
        pen = torch.clamp(pen, 0.0, max_penalty)

    pen = pen.to(torch.float32)
    _tb_store_reward(env, "energy", pen)
    _tb_store_aux(env, "energy_lin_speed", v_norm)
    _tb_store_aux(env, "energy_ang_speed", w_norm)
    _tb_store_aux(env, "energy_lin_acc", a_norm)
    _tb_store_aux(env, "energy_ang_acc", alpha_norm)
    return pen


# -----------------------------------------------------------------------------
# ⑤ termination-related rewards / penalties
# -----------------------------------------------------------------------------
def reward_goal_reached(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    out = termination_reached_goal(env, asset_cfg=asset_cfg, threshold=threshold).to(torch.float32)
    _tb_store_reward(env, "success_bonus", out)
    return out


def penalty_out_of_workspace(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    x_bounds: tuple[float, float] = (-60.0, 60.0),
    y_bounds: tuple[float, float] = (-60.0, 60.0),
    z_bounds: tuple[float, float] = (0.0, 10.0),
) -> torch.Tensor:
    out = termination_out_of_workspace(
        env,
        asset_cfg=asset_cfg,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
    ).to(torch.float32)
    _tb_store_reward(env, "oob_penalty", out)
    return out


def penalty_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    out = mdp.time_out(env).to(torch.float32)
    _tb_store_reward(env, "timeout_penalty", out)
    return out


def penalty_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    out = termination_collision(env, sensor_cfg=sensor_cfg, threshold=threshold).to(torch.float32)
    _tb_store_reward(env, "collision_penalty", out)
    return out


# -----------------------------------------------------------------------------
# Action regularization
# -----------------------------------------------------------------------------
def reward_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on raw actions to encourage smooth control."""
    term = env.action_manager.get_term("root_twist")
    a = term.raw_actions
    return (a * a).sum(dim=-1)


# -----------------------------------------------------------------------------
# Dijkstra-based navigation reward (geodesic distance progress)
# -----------------------------------------------------------------------------
def reward_dijkstra_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    grid_size: int = 160,
    cell_size: float = 1.0,
    update_interval: int = 5,
    speed_ref: float = 4.0,
    clip: float = 1.0,
) -> torch.Tensor:
    """Reward based on progress along Dijkstra-computed optimal path.

    Uses geodesic distance (path around obstacles) instead of Euclidean distance.
    This provides consistent reward signals when navigating around obstacles,
    avoiding local optima where the drone gets "stuck" behind obstacles.

    Args:
        env: Environment instance.
        asset_cfg: Asset configuration for the robot.
        grid_size: Size of the occupancy grid (grid_size x grid_size).
        cell_size: Size of each grid cell in meters.
        update_interval: Recompute distance field every N steps.
        speed_ref: Reference speed for reward normalization.
        clip: Clip reward to [-clip, clip].

    Returns:
        (N,) tensor of rewards.
    """
    # Get current position
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)

    # Get goal position
    goal = _get_goal_pos(env, pos)

    # Validate goal shape
    if goal.shape[0] != env.num_envs:
        # If goal is singleton, broadcast it
        if goal.shape[0] == 1:
            goal = goal.expand(env.num_envs, -1)
        else:
            # Fallback: create zeros
            goal = torch.zeros((env.num_envs, 3), device=env.device, dtype=pos.dtype)

    # Initialize navigator if not exists
    if getattr(env, "_dijkstra_navigator", None) is None:
        # Get workspace bounds from config
        x_bounds = (-80.0, 80.0)
        y_bounds = (-80.0, 80.0)
        try:
            norm_cfg = getattr(env.cfg, "normalization", None)
            if norm_cfg:
                x_bounds = getattr(norm_cfg, "x_bounds", x_bounds)
                y_bounds = getattr(norm_cfg, "y_bounds", y_bounds)
        except Exception:
            pass

        env._dijkstra_navigator = DijkstraNavigator(
            grid_size=grid_size,
            cell_size=cell_size,
            workspace_origin=(float(x_bounds[0]), float(y_bounds[0])),
            max_distance=300.0,
        )
        env._dijkstra_distance_fields = None
        env._dijkstra_prev_positions = pos.detach().clone()
        env._dijkstra_prev_distances = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )
        env._dijkstra_update_counter = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.int32
        )

    navigator: DijkstraNavigator = env._dijkstra_navigator

    # Ensure all buffers are initialized (safety check)
    if env._dijkstra_update_counter is None:
        env._dijkstra_update_counter = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.int32
        )
    if env._dijkstra_prev_distances is None:
        env._dijkstra_prev_distances = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )

    # Check if we need to update distance fields
    env_ids_to_update = []
    for i in range(env.num_envs):
        env._dijkstra_update_counter[i] += 1
        if env._dijkstra_update_counter[i] >= update_interval:
            env._dijkstra_update_counter[i] = 0
            env_ids_to_update.append(i)

    # Initialize distance fields if needed
    if env._dijkstra_distance_fields is None:
        env._dijkstra_distance_fields = torch.full(
            (env.num_envs, grid_size, grid_size),
            navigator.max_distance,
            device=env.device,
            dtype=torch.float32,
        )
        env_ids_to_update = list(range(env.num_envs))

    # Update distance fields for selected environments
    if env_ids_to_update:
        # Get obstacle positions from scene
        try:
            with torch.no_grad():  # Reduce memory usage during Dijkstra computation
                obstacles = env.scene["obstacles"]
                obstacle_positions = obstacles.data.root_pos_w  # (max_obstacles, 3)

                # Build occupancy grid once (shared across all environments)
                # Cache it to avoid rebuilding every time
                cache_key = "_dijkstra_occupancy_grid"

                # Check if obstacles changed (compare with cached positions)
                prev_positions = getattr(env, "_dijkstra_cached_obstacle_positions", None)
                occupancy_grid = None
                need_rebuild = False

                if prev_positions is None:
                    need_rebuild = True
                elif prev_positions.shape != obstacle_positions.shape:
                    need_rebuild = True
                else:
                    # Check if any positions changed (with small tolerance)
                    need_rebuild = not torch.allclose(obstacle_positions, prev_positions, atol=1e-3)

                if need_rebuild:
                    # Rebuild occupancy grid
                    occupancy_grid = navigator.build_occupancy_grid_from_obstacles(
                        obstacle_positions, obstacle_size=1.5, device=env.device
                    )
                    setattr(env, cache_key, occupancy_grid)
                    # Store reference (not clone) to save memory - obstacles rarely change
                    setattr(env, "_dijkstra_cached_obstacle_positions", obstacle_positions.detach())
                else:
                    occupancy_grid = getattr(env, cache_key)

            # Update distance fields for each environment (different goals)
            # Ensure goal is properly broadcast before indexing
            if goal.shape[0] == 1 and len(env_ids_to_update) > 1:
                goal = goal.expand(env.num_envs, -1)

            for idx in env_ids_to_update:
                # Safety check: ensure index is valid
                if idx >= goal.shape[0] or idx >= pos.shape[0]:
                    continue

                # Compute distance field from goal (fast GPU implementation)
                try:
                    distance_field = navigator.compute_distance_field(
                        occupancy_grid, goal[idx]
                    )
                except Exception as field_e:
                    # Skip this environment if distance field computation fails
                    continue

                env._dijkstra_distance_fields[idx] = distance_field

                # Update previous distance for this environment
                if idx < pos.shape[0]:
                    env._dijkstra_prev_distances[idx] = navigator.get_geodesic_distance(
                        pos[idx : idx + 1], distance_field
                    )[0]

        except Exception as e:
            # Fallback: use Euclidean distance if Dijkstra fails
            # Only warn once to avoid spam
            if not getattr(env, "_dijkstra_warned", False):
                import warnings
                warnings.warn(f"Dijkstra distance field computation failed: {e}")
                env._dijkstra_warned = True
            pass

    # Compute current geodesic distances
    d_current = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if env._dijkstra_distance_fields is not None:
        for i in range(env.num_envs):
            d_current[i] = navigator.get_geodesic_distance(
                pos[i : i + 1], env._dijkstra_distance_fields[i]
            )[0]
    else:
        # Fallback: use Euclidean distance
        # Ensure goal is broadcast to match pos shape
        if goal.shape[0] == 1 and pos.shape[0] > 1:
            goal = goal.expand(pos.shape[0], -1)
        d_current = torch.norm(pos[:, :2] - goal[:, :2], dim=1)

    # Get previous geodesic distances (use current if not initialized)
    if env._dijkstra_prev_distances is not None:
        d_prev = env._dijkstra_prev_distances.clone()
    else:
        d_prev = d_current.clone()

    # Compute progress reward
    dt = max(_get_step_dt(env), 1e-6)
    delta = d_prev - d_current
    towards_speed = delta / dt
    denom = max(float(speed_ref), 1e-6)
    reward = torch.clamp(towards_speed / denom, -float(clip), float(clip))

    # Update previous distances for next step
    env._dijkstra_prev_distances = d_current.detach().clone()
    env._dijkstra_prev_positions = pos.detach().clone()

    # Periodic cleanup to prevent memory leak (every 1000 steps)
    step_counter = getattr(env, "_dijkstra_step_counter", 0)
    step_counter += 1
    if step_counter % 1000 == 0:
        # Clear cached obstacle positions reference (will be recreated if needed)
        if hasattr(env, "_dijkstra_cached_obstacle_positions"):
            delattr(env, "_dijkstra_cached_obstacle_positions")
        # Force garbage collection periodically
        import gc
        gc.collect()
        step_counter = 0
    env._dijkstra_step_counter = step_counter

    # Store for TensorBoard
    _tb_store_reward(env, "dijkstra_progress", reward)
    _tb_store_aux(env, "geodesic_distance", d_current)
    _tb_store_aux(env, "geodesic_progress_norm", delta / dt)

    return reward.to(torch.float32)


# -----------------------------------------------------------------------------
# APF — Attractive Potential Field reward
# -----------------------------------------------------------------------------
def reward_apf_attractive(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    k_att: float = 1.0,
    speed_ref: float = 4.0,
    clip: float = 1.0,
) -> torch.Tensor:
    """APF attractive potential shaping reward.

    U_att = 0.5 * k_att * d_goal²
    R = -(U_att(t) - U_att(t-1)) / dt / speed_ref
      = k_att * d_prev * (d_prev - d_curr) / dt / speed_ref

    Differs from reward_progress_to_goal in that the gradient is weighted by
    distance: the drone is pulled more strongly when it is far from the goal,
    providing a natural curriculum-like attractive force.
    """
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    goal = _get_goal_pos(env, pos)
    d = _safe_norm(goal - pos)

    prev = getattr(env, "_apf_prev_goal_dist", None)
    if prev is None or not isinstance(prev, torch.Tensor) or prev.shape != d.shape:
        setattr(env, "_apf_prev_goal_dist", d.detach().clone())
        out = torch.zeros_like(d)
    else:
        dt = max(_get_step_dt(env), 1e-6)
        delta_d = prev - d  # positive when approaching
        # APF gradient: d(0.5*k*d²)/dt = k * d * (d_prev - d_curr) / dt
        raw = float(k_att) * prev * delta_d / dt
        denom = max(float(speed_ref), 1e-6)
        out = torch.clamp(raw / denom, min=-float(clip), max=float(clip))
        env._apf_prev_goal_dist = d.detach().clone()

    out = out.to(torch.float32)
    _tb_store_reward(env, "apf_attractive", out)
    _tb_store_aux(env, "apf_goal_dist", d)
    return out


# -----------------------------------------------------------------------------
# APF — Repulsive Potential Field penalty
# -----------------------------------------------------------------------------
def penalty_apf_repulsive(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    d0: float = 5.0,
    k_rep: float = 1.0,
    use_2d: bool = True,
    cap: float = 5.0,
) -> torch.Tensor:
    """APF repulsive potential penalty from discrete obstacle positions.

    For each active obstacle i with distance d_i < d0:
        U_rep_i = 0.5 * k_rep * (1/d_i - 1/d0)²
    Penalty = sum_i(U_rep_i)  [positive; apply negative weight in cfg]

    Obstacles parked at (1000, 1000, -1000) (inactive) are filtered out.
    XY-only distance is used because obstacles are 10 m-tall pillars.
    """
    out0 = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    try:
        obstacles = env.scene["obstacles"]
        obs_pos = obstacles.data.root_pos_w  # (N_obs, 3)
    except Exception:
        _tb_store_reward(env, "apf_repulsive", out0)
        return out0

    # Filter inactive obstacles (parked far away, x > 500)
    active_mask = obs_pos[:, 0].abs() < 500.0
    obs_pos_active = obs_pos[active_mask]  # (N_active, 3)

    if obs_pos_active.shape[0] == 0:
        _tb_store_reward(env, "apf_repulsive", out0)
        return out0

    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)  # (N_envs, 3)

    # Compute distances: (N_envs, N_active)
    if use_2d:
        # XY plane only; obstacles are tall pillars
        dx = pos[:, 0:1] - obs_pos_active[:, 0].unsqueeze(0)  # (N_envs, N_active)
        dy = pos[:, 1:2] - obs_pos_active[:, 1].unsqueeze(0)
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
    else:
        diff = pos.unsqueeze(1) - obs_pos_active.unsqueeze(0)  # (N_envs, N_active, 3)
        dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-6)  # (N_envs, N_active)

    d0_t = float(d0)
    k_rep_t = float(k_rep)
    d_min_clip = 0.1  # avoid singularity

    # Clamp distances from below
    dist_safe = torch.clamp(dist, min=d_min_clip)

    # Repulsive potential: 0.5 * k * (1/d - 1/d0)^2 when d < d0, else 0
    in_range = dist_safe < d0_t  # (N_envs, N_active)
    inv_d = 1.0 / dist_safe
    inv_d0 = 1.0 / d0_t
    u_rep_i = 0.5 * k_rep_t * ((inv_d - inv_d0) ** 2)  # (N_envs, N_active)
    u_rep_i = torch.where(in_range, u_rep_i, torch.zeros_like(u_rep_i))

    penalty = u_rep_i.sum(dim=1)  # (N_envs,)
    if cap > 0.0:
        penalty = torch.clamp(penalty, 0.0, float(cap))

    penalty = penalty.to(torch.float32)
    _tb_store_reward(env, "apf_repulsive", penalty)
    _tb_store_aux(env, "apf_min_obs_dist", dist.min(dim=1).values)
    return penalty


# -----------------------------------------------------------------------------
# Attitude tilt penalty (penalize large roll/pitch angles)
# -----------------------------------------------------------------------------
def penalty_attitude_tilt(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_tilt_deg: float = 20.0,
    std_deg: float = 20.0,
) -> torch.Tensor:
    """Penalize tilt angle (combined roll/pitch) exceeding max_tilt_deg.

    Uses an exponential-shaped penalty that starts at 0 and grows rapidly:
        excess = max(tilt_rad - max_tilt_rad, 0)
        penalty = expm1( (excess / std_rad)^2 )   [= exp(...) - 1, so zero at threshold]

    cos_tilt = 1 - 2*(qx^2 + qy^2) gives the world-Z component of the body-Z axis,
    which is equivalent to the cosine of the combined roll/pitch tilt angle.

    Args:
        max_tilt_deg: Free zone upper limit in degrees. No penalty below this.
        std_deg: Penalty growth rate beyond the threshold (degrees).

    Returns:
        (N,) penalty tensor (positive values; apply with negative weight in cfg).
    """
    quat = mdp.root_quat_w(env, asset_cfg=asset_cfg)  # (N, 4) [w, x, y, z]
    qx = quat[:, 1]
    qy = quat[:, 2]
    cos_tilt = torch.clamp(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)
    tilt_rad = torch.acos(cos_tilt)  # (N,) in [0, pi]

    max_tilt_rad = math.radians(float(max_tilt_deg))
    std_rad = max(math.radians(float(std_deg)), 1e-6)

    excess = torch.clamp(tilt_rad - max_tilt_rad, min=0.0)
    pen = torch.expm1((excess / std_rad) ** 2)

    pen = pen.to(torch.float32)
    _tb_store_reward(env, "attitude_tilt", pen)
    _tb_store_aux(env, "tilt_angle_deg", torch.rad2deg(tilt_rad))
    return pen

