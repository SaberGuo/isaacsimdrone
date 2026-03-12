from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except Exception:  # pragma: no cover
    gym = None
    Box = None


# =============================================================================
# Shared obstacle spawning
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
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

            obstacle_path = f"/World/Obstacles/Obstacle_{i:04d}"
            cfg_obstacle.func(obstacle_path, cfg_obstacle, translation=(x_pos, y_pos, z_pos))

            if (i + 1) % 10 == 0:
                print(f"[INFO]: 已生成 {i + 1}/{self.num_obstacles} 个障碍物")

        print("[INFO]: 共享障碍物生成完成（静态/kinematic）！")


class WallSpawner:
    """Spawn workspace boundary walls under /World/Wall.

    Walls included:
      - 4 side walls on x/y boundaries
      - 1 ceiling wall on z upper boundary
      - ground is still provided by /World/ground, so no bottom wall
    """

    def __init__(
        self,
        x_bounds: tuple = (-60.0, 60.0),
        y_bounds: tuple = (-60.0, 60.0),
        z_bounds: tuple = (0.0, 10.0),
        wall_thickness: float = 0.5,
        color: tuple = (0.7, 0.7, 0.2),
    ):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.wall_thickness = float(wall_thickness)
        self.color = color

    def _make_wall_cfg(self):
        return sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),  # placeholder; actual size passed at spawn-time
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.color),
        )

    def spawn_walls(self):
        import isaacsim.core.utils.prims as prim_utils

        prim_utils.create_prim("/World/Wall", "Xform")

        x_min, x_max = float(self.x_bounds[0]), float(self.x_bounds[1])
        y_min, y_max = float(self.y_bounds[0]), float(self.y_bounds[1])
        z_min, z_max = float(self.z_bounds[0]), float(self.z_bounds[1])

        t = self.wall_thickness

        x_len = x_max - x_min
        y_len = y_max - y_min
        z_len = z_max - z_min

        z_center = 0.5 * (z_min + z_max)
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)

        print(
            f"\n[INFO]: 正在生成工作空间围墙 /World/Wall "
            f"(x={self.x_bounds}, y={self.y_bounds}, z={self.z_bounds}, thickness={t})..."
        )

        # ------------------------------------------------------------------
        # Four side walls
        # Put wall centers OUTSIDE the workspace so the inner face aligns exactly
        # with the workspace boundary.
        # ------------------------------------------------------------------
        walls = [
            # left wall: inner face at x = x_min
            dict(
                name="Wall_XMin",
                size=(t, y_len, z_len),
                translation=(x_min - t / 2.0, y_center, z_center),
            ),
            # right wall: inner face at x = x_max
            dict(
                name="Wall_XMax",
                size=(t, y_len, z_len),
                translation=(x_max + t / 2.0, y_center, z_center),
            ),
            # bottom-y wall: inner face at y = y_min
            dict(
                name="Wall_YMin",
                size=(x_len, t, z_len),
                translation=(x_center, y_min - t / 2.0, z_center),
            ),
            # top-y wall: inner face at y = y_max
            dict(
                name="Wall_YMax",
                size=(x_len, t, z_len),
                translation=(x_center, y_max + t / 2.0, z_center),
            ),
            # ceiling: inner face at z = z_max
            dict(
                name="Wall_ZMax",
                size=(x_len, y_len, t),
                translation=(x_center, y_center, z_max + t / 2.0),
            ),
        ]

        for wall in walls:
            cfg_wall = sim_utils.CuboidCfg(
                size=wall["size"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.color),
            )
            wall_path = f"/World/Wall/{wall['name']}"
            cfg_wall.func(
                wall_path,
                cfg_wall,
                translation=wall["translation"],
            )
            print(
                f"[INFO]: 已生成围墙 {wall['name']}: "
                f"size={wall['size']}, translation={wall['translation']}"
            )

        print("[INFO]: 工作空间围墙生成完成！", flush=True)

# =============================================================================
# Env with goal buffer / energy cache / progress cache
# =============================================================================
class MyDroneRLEnv(ManagerBasedRLEnv):
    """
    Custom env that adds:
      - per-env goal buffer (goal_pos_w)
      - per-env previous velocity buffers for energy penalty
      - per-env previous goal distance buffer for progress reward
      - semantic finite gym spaces for policy observation / action
      - _reset_idx / reset_idx dual compatibility
    """

    def __init__(self, cfg=None, **kwargs):
        kwargs.pop("env_cfg_entry_point", None)
        kwargs.pop("rl_games_cfg_entry_point", None)
        kwargs.pop("rsl_rl_cfg_entry_point", None)
        kwargs.pop("skrl_cfg_entry_point", None)
        kwargs.pop("sb3_cfg_entry_point", None)

        if cfg is None:
            cfg = kwargs.pop("env_cfg", None)

        # placeholders before super().__init__()
        self.goal_pos_w = torch.zeros((1, 3), dtype=torch.float32)
        self._energy_prev_lin_vel_w = torch.zeros((1, 3), dtype=torch.float32)
        self._energy_prev_ang_vel_w = torch.zeros((1, 3), dtype=torch.float32)
        self._progress_prev_goal_dist = torch.zeros((1,), dtype=torch.float32)

        # metadata used by the training script
        self.policy_state_dim = 16
        self.policy_lidar_dim = 0
        self._batched_observation_space = None
        self._batched_action_space = None

        super().__init__(cfg=cfg)

        # Patch semantic finite spaces immediately after env construction,
        # before any external wrapper reads them.
        self._patch_semantic_single_gym_spaces_for_rl()

        # now num_envs and device are known
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._energy_prev_lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._energy_prev_ang_vel_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._progress_prev_goal_dist = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_goals(env_ids)
        self._refresh_energy_prev_buffers(env_ids)
        self._refresh_progress_prev_dist(env_ids)


        print("\n[MyDroneRLEnv] ===== Env Initialized =====", flush=True)
        print(f"[MyDroneRLEnv] num_envs={self.num_envs}, device={self.device}", flush=True)
        print(f"[MyDroneRLEnv] policy_state_dim={self.policy_state_dim}, policy_lidar_dim={self.policy_lidar_dim}", flush=True)
        try:
            print(f"[MyDroneRLEnv] step_dt={self.step_dt}", flush=True)
        except Exception:
            pass
        try:
            print(f"[MyDroneRLEnv] initial goal_pos_w[0]={self.goal_pos_w[0].detach().cpu().numpy()}", flush=True)
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # gym spaces
    # ---------------------------------------------------------------------
    def _get_state_dim_from_cfg(self) -> int:
        try:
            norm_cfg = getattr(self.cfg, "normalization", None)
            state_dim = int(getattr(norm_cfg, "state_dim", 16))
            if state_dim > 0:
                return state_dim
        except Exception:
            pass
        return 16

    def _infer_single_obs_dim(self, obs_space) -> int:
        if gym is None:
            return 0

        policy_space = None
        if isinstance(obs_space, gym.spaces.Dict):
            policy_space = obs_space.spaces.get("policy", None)
        elif isinstance(obs_space, gym.spaces.Box):
            policy_space = obs_space

        if not isinstance(policy_space, gym.spaces.Box):
            return 0

        total_dim = int(np.prod(policy_space.shape))
        num_envs = int(getattr(self, "num_envs", 1))
        if num_envs > 1 and total_dim % num_envs == 0:
            candidate = total_dim // num_envs
            if candidate > 0 and candidate != total_dim:
                return candidate
        return total_dim

    def _infer_single_act_dim(self, act_space) -> int:
        if gym is None or not isinstance(act_space, gym.spaces.Box):
            return 4

        total_dim = int(np.prod(act_space.shape))
        num_envs = int(getattr(self, "num_envs", 1))
        if num_envs > 1 and total_dim % num_envs == 0:
            candidate = total_dim // num_envs
            if candidate > 0 and candidate != total_dim:
                return candidate
        return total_dim if total_dim > 0 else 4

    def _make_policy_obs_box(self, obs_dim: int, state_dim: int) -> Box:
        low = -np.ones((obs_dim,), dtype=np.float32)
        high = np.ones((obs_dim,), dtype=np.float32)
        if obs_dim > state_dim:
            low[state_dim:] = 0.0
        return Box(low=low, high=high, dtype=np.float32)

    def _make_action_box(self, act_dim: int) -> Box:
        low = -np.ones((act_dim,), dtype=np.float32)
        high = np.ones((act_dim,), dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    def _patch_semantic_single_gym_spaces_for_rl(self):
        if gym is None or Box is None:
            return

        obs_space_before = getattr(self, "observation_space", None)
        act_space_before = getattr(self, "action_space", None)

        single_obs_dim = self._infer_single_obs_dim(obs_space_before)
        single_act_dim = self._infer_single_act_dim(act_space_before)
        if single_obs_dim <= 0:
            return

        state_dim = self._get_state_dim_from_cfg()
        if state_dim <= 0 or state_dim > single_obs_dim:
            state_dim = min(16, single_obs_dim)

        lidar_dim = max(single_obs_dim - state_dim, 0)

        self.policy_state_dim = int(state_dim)
        self.policy_lidar_dim = int(lidar_dim)

        obs_box = self._make_policy_obs_box(single_obs_dim, state_dim)
        act_box = self._make_action_box(single_act_dim)

        self._batched_observation_space = obs_space_before
        self._batched_action_space = act_space_before

        single_obs_space = gym.spaces.Dict({"policy": obs_box})

        self.observation_space = single_obs_space
        self.single_observation_space = single_obs_space
        self.action_space = act_box
        self.single_action_space = act_box

        print(
            "[MyDroneRLEnv] Using semantic single-env gym spaces: "
            f"policy_obs_dim={single_obs_dim} (state={state_dim}, lidar={lidar_dim}), "
            f"action_dim={single_act_dim}",
            flush=True,
        )

    # ---------------------------------------------------------------------
    # goal sampling / caches
    # ---------------------------------------------------------------------
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
        try:
            v = mdp.root_lin_vel_w(self, asset_cfg=SceneEntityCfg("robot"))
            w = mdp.root_ang_vel_w(self, asset_cfg=SceneEntityCfg("robot"))
            self._energy_prev_lin_vel_w[env_ids] = v[env_ids].detach()
            self._energy_prev_ang_vel_w[env_ids] = w[env_ids].detach()
        except Exception:
            self._energy_prev_lin_vel_w[env_ids] = 0.0
            self._energy_prev_ang_vel_w[env_ids] = 0.0

    def _refresh_progress_prev_dist(self, env_ids: torch.Tensor):
        try:
            pos = mdp.root_pos_w(self, asset_cfg=SceneEntityCfg("robot"))
            d = torch.norm(self.goal_pos_w - pos, dim=-1)
            self._progress_prev_goal_dist[env_ids] = d[env_ids].detach()
        except Exception:
            self._progress_prev_goal_dist[env_ids] = 0.0

    def _build_goal_info(self) -> dict:
        info = {}
        try:
            pos = mdp.root_pos_w(self, asset_cfg=SceneEntityCfg("robot"))
            info["goal_state_delta"] = (self.goal_pos_w - pos).detach()
            info["goal_pos_w"] = self.goal_pos_w.detach()
        except Exception:
            pass
        return info

    # ---------------------------------------------------------------------
    # reset compatibility
    # ---------------------------------------------------------------------
    def _call_parent_reset_idx(self, env_ids: torch.Tensor):
        parent = super()
        if hasattr(parent, "_reset_idx"):
            return parent._reset_idx(env_ids)
        if hasattr(parent, "reset_idx"):
            return parent.reset_idx(env_ids)
        raise AttributeError("Parent ManagerBasedRLEnv does not expose _reset_idx/reset_idx")

    def _reset_idx(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._sample_goals(env_ids)
        out = self._call_parent_reset_idx(env_ids)

        # refresh caches AFTER reset has written sim state
        self._refresh_energy_prev_buffers(env_ids)
        self._refresh_progress_prev_dist(env_ids)

        goal_info = self._build_goal_info()

        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            out[1].update(goal_info)
            return out

        return out

    # compatibility shim for versions/workflows that still call reset_idx
    def reset_idx(self, env_ids: torch.Tensor | None = None):
        return self._reset_idx(env_ids)
