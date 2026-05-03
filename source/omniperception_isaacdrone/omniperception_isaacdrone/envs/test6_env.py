"""Runtime helpers and custom ManagerBasedRLEnv for the Test6 drone task."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from pxr import Gf, UsdGeom

try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except Exception:  # pragma: no cover
    gym = None
    Box = None


# -----------------------------------------------------------------------------
# USD transform helpers
# -----------------------------------------------------------------------------
def _get_stage():
    import isaacsim.core.utils.prims as prim_utils

    return prim_utils.get_prim_at_path("/World").GetStage()


def _set_prim_translation(stage, prim_path: str, translation: tuple[float, float, float]) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False

    xform = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if len(translate_ops) > 0:
        translate_ops[0].Set(Gf.Vec3d(*translation))
    else:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    return True


def _set_prim_visibility(stage, prim_path: str, visible: bool) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False

    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()
    return True


# -----------------------------------------------------------------------------
# Workspace and obstacle spawners
# -----------------------------------------------------------------------------
class WallSpawner:
    """Spawn workspace boundary walls under /World/Wall."""

    def __init__(
        self,
        x_bounds: tuple = (-60.0, 60.0),
        y_bounds: tuple = (-60.0, 60.0),
        z_bounds: tuple = (0.0, 10.0),
        wall_thickness: float = 0.5,
        color: tuple = (0.7, 0.7, 0.2),
        wall_colors: dict[str, tuple[float, float, float]] | None = None,
    ):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.wall_thickness = float(wall_thickness)
        self.color = color
        self.wall_colors = wall_colors or {}

    def _get_wall_color(self, wall_name: str) -> tuple[float, float, float]:
        color = self.wall_colors.get(wall_name, self.color)
        if not isinstance(color, (tuple, list)) or len(color) != 3:
            raise ValueError(f"Invalid color for wall '{wall_name}': {color}.")
        r, g, b = float(color[0]), float(color[1]), float(color[2])
        return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)))

    def spawn_walls(self):
        import isaacsim.core.utils.prims as prim_utils
        prim_utils.create_prim("/World/Wall", "Xform")

        x_min, x_max = float(self.x_bounds[0]), float(self.x_bounds[1])
        y_min, y_max = float(self.y_bounds[0]), float(self.y_bounds[1])
        z_min, z_max = float(self.z_bounds[0]), float(self.z_bounds[1])
        t = self.wall_thickness

        x_len, y_len, z_len = x_max - x_min, y_max - y_min, z_max - z_min
        x_center, y_center, z_center = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.5 * (z_min + z_max)

        walls = [
            dict(name="Wall_XMin", size=(t, y_len, z_len), translation=(x_min, y_center, z_center)),
            dict(name="Wall_XMax", size=(t, y_len, z_len), translation=(x_max, y_center, z_center)),
            dict(name="Wall_YMin", size=(x_len, t, z_len), translation=(x_center, y_min, z_center)),
            dict(name="Wall_YMax", size=(x_len, t, z_len), translation=(x_center, y_max, z_center)),
            dict(name="Wall_ZMin", size=(x_len, y_len, t), translation=(x_center, y_center, z_min)),
            dict(name="Wall_ZMax", size=(x_len, y_len, t), translation=(x_center, y_center, z_max)),
        ]

        for wall in walls:
            wall_name = wall["name"]
            wall_color = self._get_wall_color(wall_name)
            cfg_wall = sim_utils.CuboidCfg(
                size=wall["size"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True, disable_gravity=True, kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=wall_color),
            )
            cfg_wall.func(f"/World/Wall/{wall_name}", cfg_wall, translation=wall["translation"])


def setup_global_obstacles(max_obstacles: int = 100):
    """预生成全局共享的障碍物模板到 /World/Obstacles。"""
    import isaacsim.core.utils.prims as prim_utils
    import isaaclab.sim as sim_utils

    prim_utils.create_prim("/World/Obstacles", "Xform")

    print(f"[INFO]: 预生成 {max_obstacles} 个 全局 障碍物模板到 /World/Obstacles ...", flush=True)
    for i in range(max_obstacles):
        cfg_obstacle = sim_utils.CuboidCfg(
            size=(1.0, 1.0, 10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        )
        obstacle_path = f"/World/Obstacles/obj_{i:03d}"
        cfg_obstacle.func(obstacle_path, cfg_obstacle, translation=(1000.0, 1000.0, -1000.0))


# -----------------------------------------------------------------------------
# Custom RL environment
# -----------------------------------------------------------------------------
class MyDroneRLEnv(ManagerBasedRLEnv):
    """Custom env with goal buffers."""

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

        self._lidar_grid_cache = None

        # goal visualizer settings / cache
        self._goal_vis_enabled = True
        self._goal_vis_radius = 0.35
        self._goal_vis_color = (1.0, 0.0, 0.0)
        self._goal_vis_opacity = 0.9
        self._goal_vis_paths: list[str] = []

        # metadata used by the training script
        self.policy_state_dim = 18
        self.policy_lidar_dim = 0
        self._batched_observation_space = None
        self._batched_action_space = None

        super().__init__(cfg=cfg)

        self._patch_semantic_single_gym_spaces_for_rl()

        # now num_envs and device are known
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._energy_prev_lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._energy_prev_ang_vel_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._progress_prev_goal_dist = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        if self._goal_vis_enabled:
            self._create_goal_visualizers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_goals(env_ids)
        self._refresh_energy_prev_buffers(env_ids)
        self._refresh_progress_prev_dist(env_ids)

        print("\n[MyDroneRLEnv] ===== Env Initialized =====", flush=True)
        print(f"[MyDroneRLEnv] num_envs={self.num_envs}, device={self.device}", flush=True)
        print(f"[MyDroneRLEnv] policy_state_dim={self.policy_state_dim}, policy_lidar_dim={self.policy_lidar_dim}", flush=True)

    # ------------------------------------------------------------------
    # gym spaces
    # ------------------------------------------------------------------
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
        policy_space = obs_space.spaces.get("policy", None) if isinstance(obs_space, gym.spaces.Dict) else obs_space
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
        low, high = -np.ones((obs_dim,), dtype=np.float32), np.ones((obs_dim,), dtype=np.float32)
        if obs_dim > state_dim:
            low[state_dim:] = 0.0
        return Box(low=low, high=high, dtype=np.float32)

    def _make_action_box(self, act_dim: int) -> Box:
        low, high = -np.ones((act_dim,), dtype=np.float32), np.ones((act_dim,), dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    def _patch_semantic_single_gym_spaces_for_rl(self):
        if gym is None or Box is None:
            return

        single_obs_dim = self._infer_single_obs_dim(getattr(self, "observation_space", None))
        single_act_dim = self._infer_single_act_dim(getattr(self, "action_space", None))
        if single_obs_dim <= 0:
            return

        state_dim = min(16, single_obs_dim) if (s := self._get_state_dim_from_cfg()) <= 0 or s > single_obs_dim else s
        lidar_dim = max(single_obs_dim - state_dim, 0)

        self.policy_state_dim, self.policy_lidar_dim = int(state_dim), int(lidar_dim)

        self._batched_observation_space = getattr(self, "observation_space", None)
        self._batched_action_space = getattr(self, "action_space", None)

        single_obs_space = gym.spaces.Dict({"policy": self._make_policy_obs_box(single_obs_dim, state_dim)})
        act_box = self._make_action_box(single_act_dim)

        self.observation_space = self.single_observation_space = single_obs_space
        self.action_space = self.single_action_space = act_box

    # ------------------------------------------------------------------
    # goal visualizers & tracking
    # ------------------------------------------------------------------
    def _get_stage(self):
        return self.sim.stage if hasattr(self, "sim") else self.scene.stage

    def _create_goal_visualizers(self) -> None:
        stage = self._get_stage()
        self._goal_vis_paths = []
        for env_index in range(int(self.num_envs)):
            goal_path = f"/World/envs/env_{env_index}/GoalVis"
            sphere = UsdGeom.Sphere.Define(stage, goal_path)
            sphere.CreateRadiusAttr(float(self._goal_vis_radius))
            _set_prim_translation(stage, sphere.GetPrim().GetPath().pathString, (0.0, 0.0, -1000.0))
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*self._goal_vis_color)])
            sphere.CreateDisplayOpacityAttr([float(self._goal_vis_opacity)])
            self._goal_vis_paths.append(goal_path)

    def _update_goal_visualizers(self, env_ids: torch.Tensor | None = None) -> None:
        if not self._goal_vis_enabled or len(self._goal_vis_paths) == 0:
            return
        stage = self._get_stage()
        env_ids = env_ids if env_ids is not None else torch.arange(self.num_envs, device=self.device)
        env_ids_cpu, goal_cpu = env_ids.detach().to("cpu").tolist(), self.goal_pos_w.detach().to("cpu")
        for env_id in env_ids_cpu:
            if 0 <= env_id < len(self._goal_vis_paths):
                _set_prim_translation(stage, self._goal_vis_paths[env_id], tuple(float(x) for x in goal_cpu[env_id]))

    def _sample_goals(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        self.goal_pos_w[env_ids, 0] = (torch.rand(n, device=self.device) * 2 - 1) * 35.0
        self.goal_pos_w[env_ids, 1] = (torch.rand(n, device=self.device) * 2 - 1) * 35.0
        self.goal_pos_w[env_ids, 2] = 5.0
        self._update_goal_visualizers(env_ids)

    def _refresh_energy_prev_buffers(self, env_ids: torch.Tensor):
        try:
            self._energy_prev_lin_vel_w[env_ids] = mdp.root_lin_vel_w(self, asset_cfg=SceneEntityCfg("robot"))[env_ids].detach()
            self._energy_prev_ang_vel_w[env_ids] = mdp.root_ang_vel_w(self, asset_cfg=SceneEntityCfg("robot"))[env_ids].detach()
        except Exception:
            self._energy_prev_lin_vel_w[env_ids] = 0.0
            self._energy_prev_ang_vel_w[env_ids] = 0.0

    def _refresh_progress_prev_dist(self, env_ids: torch.Tensor):
        try:
            pos = mdp.root_pos_w(self, asset_cfg=SceneEntityCfg("robot"))
            self._progress_prev_goal_dist[env_ids] = torch.norm(self.goal_pos_w - pos, dim=-1)[env_ids].detach()
        except Exception:
            self._progress_prev_goal_dist[env_ids] = 0.0

    def _clear_lidar_grid_cache(self) -> None:
        self._lidar_grid_cache = None

    def _reset_idx(self, env_ids: torch.Tensor | None = None):
        env_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids

        super()._reset_idx(env_ids)
        self._clear_lidar_grid_cache()
        self._refresh_energy_prev_buffers(env_ids)
        self._refresh_progress_prev_dist(env_ids)

    def reset_idx(self, env_ids: torch.Tensor | None = None):
        self._reset_idx(env_ids)
