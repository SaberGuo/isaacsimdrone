"""Reset events for drone start states, goals, and global obstacles."""

from __future__ import annotations

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

    while placed < num_active and tries < total_try_budget:
        candidate = torch.tensor(
            [
                torch.empty((), device=device).uniform_(inner_x_min, inner_x_max).item(),
                torch.empty((), device=device).uniform_(inner_y_min, inner_y_max).item(),
            ],
            device=device,
            dtype=torch.float32,
        )

        if placed == 0 or base_min_separation <= 0.0:
            positions_xy[placed] = candidate
            placed += 1
            tries += 1
            continue

        progress = tries / float(total_try_budget)
        relaxed_min_separation = base_min_separation * max(0.35, 1.0 - 0.7 * progress)
        dist = torch.norm(positions_xy[:placed] - candidate.unsqueeze(0), dim=1)
        if bool((dist >= relaxed_min_separation).all()):
            positions_xy[placed] = candidate
            placed += 1

        tries += 1

    if placed < num_active:
        remaining = num_active - placed
        fallback_xy = torch.empty((remaining, 2), device=device, dtype=torch.float32)
        fallback_xy[:, 0].uniform_(inner_x_min, inner_x_max)
        fallback_xy[:, 1].uniform_(inner_y_min, inner_y_max)
        positions_xy[placed:] = fallback_xy

    positions = torch.zeros((num_active, 3), device=device, dtype=torch.float32)
    positions[:, :2] = positions_xy
    positions[:, 2] = float(z_height) / 2.0
    return positions


# -----------------------------------------------------------------------------
# Robot and goal reset event
# -----------------------------------------------------------------------------
def reset_root_state_on_square_edge(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    square_half_size: float = 35.0,
    z_range: tuple = (3.0, 7.0),
):
    """将无人机 root pose 随机初始化到 square 边界上，并在其对角生成带扰动的 Goal。"""
    asset = env.scene[asset_cfg.name]
    num_resets = len(env_ids)

    edges = torch.randint(0, 4, (num_resets,), device=env.device)
    positions = torch.zeros((num_resets, 3), device=env.device)
    edge_positions = torch.rand(num_resets, device=env.device) * 2 * square_half_size - square_half_size

    left_mask   = edges == 0
    right_mask  = edges == 1
    bottom_mask = edges == 2
    top_mask    = edges == 3

    positions[left_mask,   0] = -square_half_size
    positions[left_mask,   1] = edge_positions[left_mask]
    positions[right_mask,  0] =  square_half_size
    positions[right_mask,  1] = edge_positions[right_mask]
    positions[bottom_mask, 0] = edge_positions[bottom_mask]
    positions[bottom_mask, 1] = -square_half_size
    positions[top_mask,    0] = edge_positions[top_mask]
    positions[top_mask,    1] =  square_half_size

    positions[:, 2] = (
        torch.rand(num_resets, device=env.device) * (z_range[1] - z_range[0]) + z_range[0]
    )

    orientations = torch.zeros((num_resets, 4), device=env.device)
    orientations[:, 0] = 1.0  # w

    root_states = torch.cat([positions, orientations], dim=1)
    asset.write_root_pose_to_sim(root_states, env_ids=env_ids)

    velocities = torch.zeros((num_resets, 6), device=env.device)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    if hasattr(env.unwrapped, "goal_pos_w"):
        target_x = -positions[:, 0]
        target_y = -positions[:, 1]

        noise_x = (torch.rand_like(target_x) * 2.0 - 1.0) * 5.0
        noise_y = (torch.rand_like(target_y) * 2.0 - 1.0) * 5.0

        env.unwrapped.goal_pos_w[env_ids, 0] = target_x + noise_x
        env.unwrapped.goal_pos_w[env_ids, 1] = target_y + noise_y
        env.unwrapped.goal_pos_w[env_ids, 2] = 5.0

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
    z_height: float = 10.0,
    edge_margin: float = 8.0,
    min_separation: float = 3.5,
    max_sample_tries: int = 4000,
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

    refresh_required = bool(getattr(u, "obstacle_layout_refresh_required", False))
    level_changed = bool(getattr(u, "obstacle_level_changed", False))
    last_applied_count = getattr(u, "_last_applied_obstacle_count", None)
    count_changed = last_applied_count is None or int(last_applied_count) != num_active

    # 只在首次、课程升级或障碍物数量发生变化时重排；其余 reset 沿用当前布局。
    if not (refresh_required or level_changed or count_changed):
        return

    # 消耗标志位（同一 step 内多个 env 同时 reset 也只执行一次）
    u.obstacle_level_changed = False
    u.obstacle_layout_refresh_required = False

    print(
        f"[OBSTACLE] 重排障碍物: active={num_active}/{max_obstacles} "
        f"(curr_obstacle_count={u.curr_obstacle_count})",
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
    u._last_applied_obstacle_count = int(num_active)
