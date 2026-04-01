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

    left_mask = edges == 0
    right_mask = edges == 1
    bottom_mask = edges == 2
    top_mask = edges == 3

    positions[left_mask, 0] = -square_half_size
    positions[left_mask, 1] = edge_positions[left_mask]
    positions[right_mask, 0] = square_half_size
    positions[right_mask, 1] = edge_positions[right_mask]
    positions[bottom_mask, 0] = edge_positions[bottom_mask]
    positions[bottom_mask, 1] = -square_half_size
    positions[top_mask, 0] = edge_positions[top_mask]
    positions[top_mask, 1] = square_half_size

    positions[:, 2] = torch.rand(num_resets, device=env.device) * (z_range[1] - z_range[0]) + z_range[0]

    orientations = torch.zeros((num_resets, 4), device=env.device)
    orientations[:, 0] = 1.0  # w

    # 写入机器人位姿和速度
    root_states = torch.cat([positions, orientations], dim=1)
    asset.write_root_pose_to_sim(root_states, env_ids=env_ids)

    velocities = torch.zeros((num_resets, 6), device=env.device)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    # ---------------------------------------------------------
    # 新增：直接利用刚生成的无人机 position，将 Goal 设置在对面
    # ---------------------------------------------------------
    if hasattr(env.unwrapped, "goal_pos_w"):
        # 基础坐标取反：(-x, -y)
        target_x = -positions[:, 0]
        target_y = -positions[:, 1]
        
        # 加上小范围的随机扰动 [-5.0, 5.0]
        noise_x = (torch.rand_like(target_x) * 2.0 - 1.0) * 5.0
        noise_y = (torch.rand_like(target_y) * 2.0 - 1.0) * 5.0
        
        env.unwrapped.goal_pos_w[env_ids, 0] = target_x + noise_x
        env.unwrapped.goal_pos_w[env_ids, 1] = target_y + noise_y
        env.unwrapped.goal_pos_w[env_ids, 2] = 5.0
        
        # 更新红色的 Goal 可视化小球
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
    """根据当前 Curriculum 等级，在需要改变时随机放置全局障碍物"""
    
    # 【修复1】初次运行兜底处理：确保程序启动 step=0 时能刷出一波障碍物
    if not hasattr(env.unwrapped, "obstacle_level_changed"):
        env.unwrapped.obstacle_level_changed = True
        
    # 如果难度没变，直接返回，不打乱其他正常飞行的无人机环境
    if not getattr(env.unwrapped, "obstacle_level_changed", False):
        return
        
    env.unwrapped.obstacle_level_changed = False
    
    asset = env.scene[asset_cfg.name]
    
    # 【修复2】全局资产 shape 通常是 [max_obstacles, state_dim]，即 2维
    state_shape = asset.data.default_root_state.shape
    if len(state_shape) == 2:
        max_obstacles = state_shape[0]
    elif len(state_shape) == 3:
        max_obstacles = state_shape[1]
    else:
        return
        
    # 【修复3】安全获取当前应激活的数量 (兼容初始化时 Curriculum 还没赋值的情况)
    num_active = getattr(env.unwrapped, "current_obstacle_count", -1)
    if num_active == -1:
        try:
            # 从 cfg 中读取第一关的数量
            levels = env.cfg.curriculum.obstacle_count.params.get("levels", (30,))
            num_active = levels[0]
            env.unwrapped.current_obstacle_count = num_active
        except Exception:
            num_active = 30
            
    num_active = min(num_active, max_obstacles)

    # 1. 默认把所有障碍物扔到地下 (Z = -1000)
    positions = torch.zeros((max_obstacles, 3), device=env.device)
    positions[:, 0] = 1000.0
    positions[:, 1] = 1000.0
    positions[:, 2] = -1000.0

    # 2. 对需要激活的障碍物，随机分布在 workspace 中
    if num_active > 0:
        x_pos = torch.rand((num_active,), device=env.device) * (x_range[1] - x_range[0]) + x_range[0]
        y_pos = torch.rand((num_active,), device=env.device) * (y_range[1] - y_range[0]) + y_range[0]
        z_pos = torch.full((num_active,), z_height / 2.0, device=env.device)
        
        positions[:num_active, 0] = x_pos
        positions[:num_active, 1] = y_pos
        positions[:num_active, 2] = z_pos

    orientations = torch.zeros((max_obstacles, 4), device=env.device)
    orientations[:, 0] = 1.0  # w=1
    
    # 手动构造速度张量为 0
    velocities = torch.zeros((max_obstacles, 6), device=env.device)

    # 如果系统依然将其识别为 [1, max_obstacles, 13] 的 3维结构，增加一维以对齐
    if len(state_shape) == 3:
        positions = positions.unsqueeze(0)
        orientations = orientations.unsqueeze(0)
        velocities = velocities.unsqueeze(0)

    root_states = torch.cat([positions, orientations], dim=-1)

    # 【修复4】将全局物体的位姿和速度写入
    # 对于全局资产，直接使用 write_root_pose_to_sim 且 env_ids 缺省，这样会覆写它拥有的所有实例
    asset.write_root_pose_to_sim(root_states)
    asset.write_root_velocity_to_sim(velocities)
