# ~/hjr_isaacdrone_ws/omniperception_isaacdrone/source/omniperception_isaacdrone/omniperception_isaacdrone/envs/test6_env.py

"""
职责：
  1. MyDroneRLEnv  — 带 goal_pos_w 的自定义 ManagerBasedRLEnv
  2. ObstacleSpawner — 共享障碍物生成工具（训练/评估脚本调用）

新增：
  - 为能量惩罚提供跨步速度缓存：_energy_prev_lin_vel_w / _energy_prev_ang_vel_w
    并在 reset_idx 时刷新，避免跨 episode 的“假加速度尖峰”
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# =============================================================================
# 共享障碍物生成
# =============================================================================
class ObstacleSpawner:
    def __init__(
        self,
        num_obstacles: int = 50,
        x_range: tuple = (-33.0, 33.0),
        y_range: tuple = (-33.0, 33.0),
        xy_size_range: tuple = (0.5, 1.5),
        z_height: float = 10.0,
        seed: int = 42,
    ):
        self.num_obstacles = num_obstacles
        self.x_range = x_range
        self.y_range = y_range
        self.xy_size_range = xy_size_range
        self.z_height = z_height
        if seed is not None:
            np.random.seed(seed)

    def spawn_obstacles(self):
        import isaacsim.core.utils.prims as prim_utils

        prim_utils.create_prim("/World/Obstacles", "Xform")
        print(f"\n[INFO]: 正在生成 {self.num_obstacles} 个共享障碍物(静态/kinematic)...")
        for i in range(self.num_obstacles):
            x_pos = np.random.uniform(*self.x_range)
            y_pos = np.random.uniform(*self.y_range)
            z_pos = self.z_height / 2.0

            x_size = np.random.uniform(*self.xy_size_range)
            y_size = np.random.uniform(*self.xy_size_range)
            z_size = self.z_height

            color = (
                np.random.uniform(0.3, 0.7),
                np.random.uniform(0.3, 0.7),
                np.random.uniform(0.3, 0.7),
            )

            cfg_obstacle = sim_utils.CuboidCfg(
                size=(x_size, y_size, z_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=True,
                    kinematic_enabled=True,  # kinematic -> 不受物理驱动但保留碰撞
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

            obstacle_path = f"/World/Obstacles/Obstacle_{i:04d}"
            cfg_obstacle.func(obstacle_path, cfg_obstacle, translation=(x_pos, y_pos, z_pos))

            if (i + 1) % 10 == 0:
                print(f"[INFO]: 已生成 {i + 1}/{self.num_obstacles} 个障碍物")

        print("[INFO]: 共享障碍物生成完成（静态/kinematic）！")


# =============================================================================
# Env：带 goal buffer 的自定义 ManagerBasedRLEnv
# =============================================================================
class MyDroneRLEnv(ManagerBasedRLEnv):
    """
    Custom env that adds:
      - per-env goal buffer (goal_pos_w)
      - per-env prev velocity buffers for energy penalty
    Compatible with gym.make() kwargs used by IsaacLab registry.
    """

    def __init__(self, cfg=None, **kwargs):
        # Pop registry-specific kwargs that gym.make passes but we don't need
        kwargs.pop("env_cfg_entry_point", None)
        kwargs.pop("rl_games_cfg_entry_point", None)
        kwargs.pop("rsl_rl_cfg_entry_point", None)
        kwargs.pop("skrl_cfg_entry_point", None)
        kwargs.pop("sb3_cfg_entry_point", None)

        if cfg is None:
            cfg = kwargs.pop("env_cfg", None)

        # Temp placeholders (resized after super().__init__)
        self.goal_pos_w = torch.zeros((1, 3), dtype=torch.float32)

        # For energy penalty (finite-difference acceleration)
        self._energy_prev_lin_vel_w = torch.zeros((1, 3), dtype=torch.float32)
        self._energy_prev_ang_vel_w = torch.zeros((1, 3), dtype=torch.float32)

        super().__init__(cfg=cfg)

        # Now num_envs and device are known
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._energy_prev_lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._energy_prev_ang_vel_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        self._sample_goals(torch.arange(self.num_envs, device=self.device))

        # init velocity caches from current sim state
        self._refresh_energy_prev_buffers(torch.arange(self.num_envs, device=self.device))

    # ----- goal sampling -----
    def _sample_goals(self, env_ids: torch.Tensor):
        square_half_size = 35.0
        goal_z_min = 3.0
        goal_z_max = 8.0

        n = env_ids.numel()
        gx = (torch.rand(n, device=self.device) * 2 - 1) * square_half_size
        gy = (torch.rand(n, device=self.device) * 2 - 1) * square_half_size
        gz = torch.rand(n, device=self.device) * (goal_z_max - goal_z_min) + goal_z_min

        self.goal_pos_w[env_ids, 0] = gx
        self.goal_pos_w[env_ids, 1] = gy
        self.goal_pos_w[env_ids, 2] = gz

    def _refresh_energy_prev_buffers(self, env_ids: torch.Tensor):
        """Refresh prev velocity buffers to avoid cross-episode acceleration spikes."""
        try:
            v = mdp.root_lin_vel_w(self, asset_cfg=SceneEntityCfg("robot"))
            w = mdp.root_ang_vel_w(self, asset_cfg=SceneEntityCfg("robot"))

            self._energy_prev_lin_vel_w[env_ids] = v[env_ids].detach()
            self._energy_prev_ang_vel_w[env_ids] = w[env_ids].detach()
        except Exception:
            # fallback: zeros
            self._energy_prev_lin_vel_w[env_ids] = 0.0
            self._energy_prev_ang_vel_w[env_ids] = 0.0

    # ----- reset with re-sampled goals -----
    def reset_idx(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # resample goal
        self._sample_goals(env_ids)

        obs, info = super().reset_idx(env_ids)

        # refresh energy caches AFTER reset has written velocities to sim
        self._refresh_energy_prev_buffers(env_ids)

        # optional info
        try:
            pos = mdp.root_pos_w(self, asset_cfg=SceneEntityCfg("robot"))
            info["goal_state_delta"] = (self.goal_pos_w - pos).detach()
            info["goal_pos_w"] = self.goal_pos_w.detach()
        except Exception:
            pass

        return obs, info
