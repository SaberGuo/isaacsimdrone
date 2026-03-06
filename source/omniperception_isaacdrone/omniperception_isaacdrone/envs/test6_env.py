from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


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


# =============================================================================
# Env: goal buffer + finite semantic gym spaces
# =============================================================================
class MyDroneRLEnv(ManagerBasedRLEnv):
    """Custom UAV RL environment.

    Responsibilities:
      - per-env goal buffer (goal_pos_w)
      - per-env prev velocity buffers for energy penalty
      - per-env prev goal distance buffer for progress reward
      - finite semantic gym spaces for policy observations/actions

    Why the explicit gym-space override is required:
      - IsaacLab's ManagerBasedRLEnv builds concatenated observation groups as
        Box(-inf, inf, ...) and the action space as Box(-inf, inf, ...) by default.
      - For this task, the actual semantic ranges are known:
          * policy state terms are normalized to [-1, 1]
          * lidar closeness is normalized to [0, 1]
          * raw policy actions are normalized to [-1, 1]
      - skrl reads `single_observation_space["policy"]` and `single_action_space`
        from the unwrapped environment. Therefore the finite semantic spaces must
        be defined directly on the environment, not only inside the model.
    """

    def __init__(self, cfg=None, **kwargs):
        kwargs.pop("env_cfg_entry_point", None)
        kwargs.pop("rl_games_cfg_entry_point", None)
        kwargs.pop("rsl_rl_cfg_entry_point", None)
        kwargs.pop("skrl_cfg_entry_point", None)
        kwargs.pop("sb3_cfg_entry_point", None)

        if cfg is None:
            cfg = kwargs.pop("env_cfg", None)

        # Temporary placeholders. Resized after super().__init__()
        self.goal_pos_w = torch.zeros((1, 3), dtype=torch.float32)
        self._energy_prev_lin_vel_w = torch.zeros((1, 3), dtype=torch.float32)
        self._energy_prev_ang_vel_w = torch.zeros((1, 3), dtype=torch.float32)
        self._progress_prev_goal_dist = torch.zeros((1,), dtype=torch.float32)

        # Filled in by _configure_gym_env_spaces during super().__init__()
        self.policy_obs_dim: int = 0
        self.policy_state_dim: int = 0
        self.policy_lidar_dim: int = 0
        self.policy_action_dim: int = 0

        super().__init__(cfg=cfg)

        # Now num_envs and device are known
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._energy_prev_lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._energy_prev_ang_vel_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._progress_prev_goal_dist = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_goals(env_ids)
        self._refresh_energy_prev_buffers(env_ids)
        self._refresh_progress_prev_dist(env_ids)

    # -------------------------------------------------------------------------
    # Gym spaces
    # -------------------------------------------------------------------------
    @staticmethod
    def _shape_tuple(dim) -> tuple[int, ...]:
        if isinstance(dim, int):
            return (int(dim),)
        return tuple(int(x) for x in dim)

    def _build_policy_observation_box(self):
        import gymnasium as gym

        try:
            obs_shape = self._shape_tuple(self.observation_manager.group_obs_dim["policy"])
        except Exception as exc:
            raise RuntimeError("Policy observation group 'policy' is missing.") from exc

        if len(obs_shape) != 1:
            raise RuntimeError(
                f"Expected 1D concatenated policy observation, but got shape={obs_shape}. "
                "Please keep policy observations concatenated for this task."
            )

        obs_dim = int(obs_shape[0])
        state_dim = int(getattr(getattr(self.cfg, "normalization", None), "state_dim", 19))
        if state_dim <= 0 or state_dim > obs_dim:
            raise RuntimeError(
                f"Invalid normalization.state_dim={state_dim}. "
                f"It must satisfy 0 < state_dim <= policy_obs_dim ({obs_dim})."
            )

        lidar_dim = obs_dim - state_dim

        low = -np.ones((obs_dim,), dtype=np.float32)
        high = np.ones((obs_dim,), dtype=np.float32)
        if lidar_dim > 0:
            low[state_dim:] = 0.0

        policy_box = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return policy_box, obs_dim, state_dim, lidar_dim

    def _build_default_non_policy_group_space(self, group_name: str):
        import gymnasium as gym

        has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
        group_dim = self.observation_manager.group_obs_dim[group_name]

        if has_concatenated_obs:
            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self._shape_tuple(group_dim),
                dtype=np.float32,
            )

        group_term_names = self.observation_manager.active_terms[group_name]
        group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]

        term_dict = {}
        for term_name, term_dim, term_cfg in zip(group_term_names, group_dim, group_term_cfgs):
            low = -np.inf if getattr(term_cfg, "clip", None) is None else term_cfg.clip[0]
            high = np.inf if getattr(term_cfg, "clip", None) is None else term_cfg.clip[1]
            term_dict[term_name] = gym.spaces.Box(
                low=low,
                high=high,
                shape=self._shape_tuple(term_dim),
                dtype=np.float32,
            )
        return gym.spaces.Dict(term_dict)

    def _configure_gym_env_spaces(self):
        """Configure semantic finite gym spaces for this UAV task."""
        import gymnasium as gym

        policy_box, obs_dim, state_dim, lidar_dim = self._build_policy_observation_box()

        action_dim = int(sum(self.action_manager.action_term_dim))
        if action_dim <= 0:
            raise RuntimeError(f"Invalid action_dim={action_dim} inferred from ActionManager.")

        action_box = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        single_obs_spaces = {}
        for group_name in self.observation_manager.active_terms.keys():
            if group_name == "policy":
                single_obs_spaces[group_name] = policy_box
            else:
                single_obs_spaces[group_name] = self._build_default_non_policy_group_space(group_name)

        self.single_observation_space = gym.spaces.Dict(single_obs_spaces)
        self.single_action_space = action_box

        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        self.policy_obs_dim = obs_dim
        self.policy_state_dim = state_dim
        self.policy_lidar_dim = lidar_dim
        self.policy_action_dim = action_dim

        print(
            "[MyDroneRLEnv] Using semantic single-env gym spaces: "
            f"policy_obs_dim={obs_dim} (state={state_dim}, lidar={lidar_dim}), "
            f"action_dim={action_dim}, "
            f"policy_low=[{float(policy_box.low.min()):.1f}, {float(policy_box.low.max()):.1f}], "
            f"policy_high=[{float(policy_box.high.min()):.1f}, {float(policy_box.high.max()):.1f}]",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Goal sampling / caches
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Reset compatibility across IsaacLab versions
    # -------------------------------------------------------------------------
    def _call_parent_reset_idx(self, env_ids: torch.Tensor):
        parent = super()
        if hasattr(parent, "_reset_idx"):
            return parent._reset_idx(env_ids)
        if hasattr(parent, "reset_idx"):
            return parent.reset_idx(env_ids)
        raise AttributeError("Parent environment provides neither _reset_idx nor reset_idx.")

    def _post_reset_refresh(self, env_ids: torch.Tensor):
        self._refresh_energy_prev_buffers(env_ids)
        self._refresh_progress_prev_dist(env_ids)
        try:
            pos = mdp.root_pos_w(self, asset_cfg=SceneEntityCfg("robot"))
            if not hasattr(self, "extras") or not isinstance(self.extras, dict):
                self.extras = {}
            log_dict = self.extras.get("log", {})
            if not isinstance(log_dict, dict):
                log_dict = {}
            log_dict["goal_state_delta"] = (self.goal_pos_w - pos).detach()
            log_dict["goal_pos_w"] = self.goal_pos_w.detach()
            self.extras["log"] = log_dict
        except Exception:
            pass

    def _reset_idx(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._sample_goals(env_ids)
        out = self._call_parent_reset_idx(env_ids)
        self._post_reset_refresh(env_ids)
        return out

    # Compatibility shim for versions/workflows that still call reset_idx
    def reset_idx(self, env_ids: torch.Tensor | None = None):
        return self._reset_idx(env_ids)
