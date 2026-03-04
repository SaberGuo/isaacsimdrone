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
    """将无人机 root pose 随机初始化到 square 边界上，速度清零。"""
    asset = env.scene[asset_cfg.name]
    # print("root mass:", asset.data.default_mass)              # 或 asset.data.mass / root_mass
    # print("inertia:", asset.data.default_inertia)           # 或 body_inertia
    # print("com:", asset.data.body_com_pos_w)           # 或 asset.data.com_pos_w / root_mass
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

    root_states = torch.cat([positions, orientations], dim=1)
    asset.write_root_pose_to_sim(root_states, env_ids=env_ids)

    velocities = torch.zeros((num_resets, 6), device=env.device)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
