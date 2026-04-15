from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


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


def randomize_obstacles_on_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    x_range: tuple = (-33.0, 33.0),
    y_range: tuple = (-33.0, 33.0),
    z_height: float = 10.0,
):
    """
    每次 level 变化时，根据当前 Curriculum 等级重新随机放置全局障碍物。

    设计原则
    --------
    * 障碍物是全局共享资产（/World/Obstacles/obj_*），与 env 无关，
      因此每次只要有任意 env reset，就整体重新布置一次即可。
    * 只有在 obstacle_level_changed=True（晋级）或首次运行时，
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
    # 2. 只在 level 发生变化时重排；其余 reset 直接跳过                    #
    # ------------------------------------------------------------------ #
    if not u.obstacle_level_changed:
        return

    # 消耗标志位（同一 step 内多个 env 同时 reset 也只执行一次）
    u.obstacle_level_changed = False

    # ------------------------------------------------------------------ #
    # 3. 确定本次应激活的障碍物数量                                        #
    # ------------------------------------------------------------------ #
    num_active = int(u.curr_obstacle_count)

    # ------------------------------------------------------------------ #
    # 4. 解析全局资产的 max_obstacles                                      #
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

    num_active = min(num_active, max_obstacles)

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
        positions[:num_active, 0] = (
            torch.rand(num_active, device=env.device)
            * (x_range[1] - x_range[0]) + x_range[0]
        )
        positions[:num_active, 1] = (
            torch.rand(num_active, device=env.device)
            * (y_range[1] - y_range[0]) + y_range[0]
        )
        positions[:num_active, 2] = z_height / 2.0

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
