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


# =============================================================================
# USD / Xform helpers
# =============================================================================
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


# =============================================================================
# Shared obstacle spawning
# =============================================================================
class ObstacleSpawner:
    """Spawn a fixed pool of shared obstacles and activate only a prefix of them."""

    _spawned: bool = False
    _prim_paths: list[str] = []
    _active_translations: list[tuple[float, float, float]] = []
    _parking_translations: list[tuple[float, float, float]] = []
    _active_count: int = 0

    def __init__(
        self,
        num_obstacles: int = 100,
        x_range: tuple = (-33.0, 33.0),
        y_range: tuple = (-33.0, 33.0),
        xy_size_range: tuple = (0.5, 1.5),
        z_height: float = 10.0,
        seed: int = 42,
    ):
        self.num_obstacles = int(num_obstacles)
        self.x_range = x_range
        self.y_range = y_range
        self.xy_size_range = xy_size_range
        self.z_height = float(z_height)
        self.seed = int(seed)

        if seed is not None:
            np.random.seed(seed)

    @classmethod
    def is_spawned(cls) -> bool:
        return cls._spawned and len(cls._prim_paths) > 0

    @classmethod
    def total_count(cls) -> int:
        return len(cls._prim_paths)

    @classmethod
    def get_active_count(cls) -> int:
        return int(cls._active_count)

    @classmethod
    def set_active_count(cls, active_count: int) -> int:
        if not cls.is_spawned():
            print("[WARN]: ObstacleSpawner.set_active_count called before shared obstacles were spawned.", flush=True)
            return 0

        active_count = int(max(0, min(active_count, cls.total_count())))
        if active_count == cls._active_count:
            return cls._active_count

        stage = _get_stage()
        for i, prim_path in enumerate(cls._prim_paths):
            translation = cls._active_translations[i] if i < active_count else cls._parking_translations[i]
            _set_prim_translation(stage, prim_path, translation)
            _set_prim_visibility(stage, prim_path, i < active_count)

        cls._active_count = active_count
        print(
            f"[INFO]: Shared obstacle curriculum applied: active={cls._active_count}/{cls.total_count()}",
            flush=True,
        )
        return cls._active_count

    def spawn_obstacles(self):
        import isaacsim.core.utils.prims as prim_utils

        if ObstacleSpawner.is_spawned():
            if self.num_obstacles <= ObstacleSpawner.total_count():
                print(
                    f"[INFO]: Reusing existing shared obstacle pool "
                    f"({ObstacleSpawner.total_count()} obstacles already spawned).",
                    flush=True,
                )
                return
            raise RuntimeError(
                "ObstacleSpawner was already initialized with fewer obstacles than requested. "
                "Please restart Isaac Sim and spawn the maximum curriculum obstacle pool once."
            )

        prim_utils.create_prim("/World/Obstacles", "Xform")

        ObstacleSpawner._prim_paths = []
        ObstacleSpawner._active_translations = []
        ObstacleSpawner._parking_translations = []
        ObstacleSpawner._active_count = self.num_obstacles

        print(f"\n[INFO]: 正在生成 {self.num_obstacles} 个共享障碍物(静态/kinematic)...", flush=True)
        for i in range(0):
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

            ObstacleSpawner._prim_paths.append(obstacle_path)
            ObstacleSpawner._active_translations.append((float(x_pos), float(y_pos), float(z_pos)))
            ObstacleSpawner._parking_translations.append((1000.0 + 5.0 * float(i), 1000.0, -1000.0))

            if (i + 1) % 10 == 0:
                print(f"[INFO]: 已生成 {i + 1}/{self.num_obstacles} 个障碍物", flush=True)

        ObstacleSpawner._spawned = True
        print("[INFO]: 共享障碍物生成完成（静态/kinematic）！", flush=True)


class WallSpawner:
    """Spawn workspace boundary walls under /World/Wall.

    Supports either:
      - one shared default color for all walls, or
      - per-wall colors via wall_colors dict.
    """

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
        """Return per-wall color if provided, else fallback to default color."""
        color = self.wall_colors.get(wall_name, self.color)

        if not isinstance(color, (tuple, list)) or len(color) != 3:
            raise ValueError(
                f"Invalid color for wall '{wall_name}': {color}. "
                f"Expected tuple/list of 3 floats in [0, 1]."
            )

        r, g, b = float(color[0]), float(color[1]), float(color[2])
        r = max(0.0, min(1.0, r))
        g = max(0.0, min(1.0, g))
        b = max(0.0, min(1.0, b))
        return (r, g, b)

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
            f"(x={self.x_bounds}, y={self.y_bounds}, z={self.z_bounds}, thickness={t})...",
            flush=True,
        )

        walls = [
            dict(
                name="Wall_XMin",
                size=(t, y_len, z_len),
                translation=(x_min, y_center, z_center),
            ),
            dict(
                name="Wall_XMax",
                size=(t, y_len, z_len),
                translation=(x_max, y_center, z_center),
            ),
            dict(
                name="Wall_YMin",
                size=(x_len, t, z_len),
                translation=(x_center, y_min, z_center),
            ),
            dict(
                name="Wall_YMax",
                size=(x_len, t, z_len),
                translation=(x_center, y_max, z_center),
            ),
            dict(
                name="Wall_ZMin",
                size=(x_len, y_len, t),
                translation=(x_center, y_center, z_min),
            )
            ,
            dict(
                name="Wall_ZMax",
                size=(x_len, y_len, t),
                translation=(x_center, y_center, z_max),
            ),
        ]

        for wall in walls:
            wall_name = wall["name"]
            wall_color = self._get_wall_color(wall_name)

            cfg_wall = sim_utils.CuboidCfg(
                size=wall["size"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=wall_color),
            )

            wall_path = f"/World/Wall/{wall_name}"
            cfg_wall.func(wall_path, cfg_wall, translation=wall["translation"])

            print(
                f"[INFO]: 已生成围墙 {wall_name}: "
                f"size={wall['size']}, translation={wall['translation']}, color={wall_color}",
                flush=True,
            )

        print("[INFO]: 工作空间围墙生成完成！", flush=True)

# =============================================================================
# Env with goal buffer / energy cache / progress cache / obstacle curriculum
# =============================================================================
class MyDroneRLEnv(ManagerBasedRLEnv):
    """Custom env with goal buffers and shared-obstacle curriculum support."""

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

        # goal visualizer settings / cache
        self._goal_vis_enabled = True
        self._goal_vis_radius = 0.35
        self._goal_vis_color = (1.0, 0.0, 0.0)
        self._goal_vis_opacity = 0.9
        self._goal_vis_paths: list[str] = []

        # metadata used by the training script
        self.policy_state_dim = 17
        self.policy_lidar_dim = 0
        self._batched_observation_space = None
        self._batched_action_space = None

        # obstacle curriculum state
        self.curriculum_obstacle_levels: tuple[int, ...] = (0, 10, 20, 40, 60, 100)
        self.curriculum_obstacle_level_idx: int = 0
        self.curriculum_active_obstacles: int = 0
        self.curriculum_success_threshold: float = 0.8
        self.curriculum_promotion_count: int = 0
        self.curriculum_last_promotion_step: int = -1
        self.curriculum_last_promotion_ratio: float = 0.0

        self.curriculum_last_batch_success_ratio: float = 0.0
        self.curriculum_last_batch_success_count: int = 0
        self.curriculum_last_batch_termination_count: int = 0

        self.curriculum_decision_success_ratio: float = 0.0
        self.curriculum_decision_window_size: int = 0
        self.curriculum_recent_success_ratio: float = 0.0
        self.curriculum_recent_termination_count: int = 0
        self._curriculum_success_window = deque(maxlen=1)

        super().__init__(cfg=cfg)

        # Patch semantic finite spaces immediately after env construction,
        # before any external wrapper reads them.
        self._patch_semantic_single_gym_spaces_for_rl()

        # now num_envs and device are known
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._energy_prev_lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._energy_prev_ang_vel_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._progress_prev_goal_dist = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        obstacle_curr_cfg = getattr(self.cfg, "obstacle_curriculum", None)
        if obstacle_curr_cfg is not None:
            self.configure_obstacle_curriculum(
                levels=tuple(int(v) for v in getattr(obstacle_curr_cfg, "levels", self.curriculum_obstacle_levels)),
                initial_level=int(getattr(obstacle_curr_cfg, "initial_level", 0)),
                window_size=int(getattr(obstacle_curr_cfg, "window_size", 200)),
                success_threshold=float(getattr(obstacle_curr_cfg, "success_threshold", 0.8)),
                reset_history=True,
            )

        if self._goal_vis_enabled:
            self._create_goal_visualizers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_goals(env_ids)
        self._refresh_energy_prev_buffers(env_ids)
        self._refresh_progress_prev_dist(env_ids)

        print("\n[MyDroneRLEnv] ===== Env Initialized =====", flush=True)
        print(f"[MyDroneRLEnv] num_envs={self.num_envs}, device={self.device}", flush=True)
        print(
            f"[MyDroneRLEnv] policy_state_dim={self.policy_state_dim}, "
            f"policy_lidar_dim={self.policy_lidar_dim}",
            flush=True,
        )
        print(
            f"[MyDroneRLEnv] obstacle curriculum levels={self.curriculum_obstacle_levels}, "
            f"level_idx={self.curriculum_obstacle_level_idx}, "
            f"active_obstacles={self.curriculum_active_obstacles}",
            flush=True,
        )
        try:
            print(f"[MyDroneRLEnv] step_dt={self.step_dt}", flush=True)
        except Exception:
            pass
        try:
            print(f"[MyDroneRLEnv] initial goal_pos_w[0]={self.goal_pos_w[0].detach().cpu().numpy()}", flush=True)
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # obstacle curriculum helpers
    # ---------------------------------------------------------------------
    def configure_obstacle_curriculum(
        self,
        levels: tuple[int, ...],
        initial_level: int = 0,
        window_size: int = 200,
        success_threshold: float = 0.8,
        reset_history: bool = False,
    ) -> None:
        levels = tuple(max(int(v), 0) for v in levels)
        if len(levels) == 0:
            levels = (0,)

        self.curriculum_obstacle_levels = levels
        self.curriculum_success_threshold = float(success_threshold)

        history = getattr(self, "_curriculum_success_window", None)
        old_values = list(history) if isinstance(history, deque) else []
        window_size = max(int(window_size), 1)
        self._curriculum_success_window = deque(old_values[-window_size:], maxlen=window_size)
        if reset_history:
            self._curriculum_success_window.clear()
            self.curriculum_promotion_count = 0
            self.curriculum_last_promotion_step = -1
            self.curriculum_last_promotion_ratio = 0.0
            self.curriculum_last_batch_success_ratio = 0.0
            self.curriculum_last_batch_success_count = 0
            self.curriculum_last_batch_termination_count = 0
            self.curriculum_decision_success_ratio = 0.0
            self.curriculum_decision_window_size = 0
            self.curriculum_recent_success_ratio = 0.0
            self.curriculum_recent_termination_count = 0

        initial_level = max(0, min(int(initial_level), len(levels) - 1))
        self.set_obstacle_curriculum_level(initial_level, force=True)

    def set_obstacle_curriculum_level(self, level_idx: int, force: bool = False) -> int:
        if len(self.curriculum_obstacle_levels) == 0:
            self.curriculum_obstacle_levels = (0,)

        level_idx = max(0, min(int(level_idx), len(self.curriculum_obstacle_levels) - 1))
        requested_obstacles = int(self.curriculum_obstacle_levels[level_idx])

        if (not force) and (level_idx == self.curriculum_obstacle_level_idx):
            return self.curriculum_active_obstacles

        self.curriculum_obstacle_level_idx = level_idx
        self.curriculum_active_obstacles = requested_obstacles

        if not ObstacleSpawner.is_spawned():
            print(
                "[WARN][MyDroneRLEnv] Shared obstacles have not been spawned yet. "
                "Curriculum state is cached and will take effect once obstacles exist.",
                flush=True,
            )
            return self.curriculum_active_obstacles

        available = ObstacleSpawner.total_count()
        if requested_obstacles > available:
            print(
                f"[WARN][MyDroneRLEnv] Curriculum requested {requested_obstacles} obstacles but only "
                f"{available} were spawned. Clamping to available pool size.",
                flush=True,
            )

        self.curriculum_active_obstacles = ObstacleSpawner.set_active_count(requested_obstacles)
        return self.curriculum_active_obstacles

    def get_obstacle_curriculum_state(self) -> dict[str, float]:
        history = getattr(self, "_curriculum_success_window", None)
        if isinstance(history, deque):
            rolling_count = len(history)
            rolling_success = int(sum(history))
        else:
            rolling_count = 0
            rolling_success = 0

        rolling_ratio = float(rolling_success) / float(rolling_count) if rolling_count > 0 else 0.0

        return {
            "level_idx": float(self.curriculum_obstacle_level_idx),
            "active_obstacles": float(self.curriculum_active_obstacles),
            "last_batch_success_ratio": float(self.curriculum_last_batch_success_ratio),
            "last_batch_success_count": float(self.curriculum_last_batch_success_count),
            "last_batch_termination_count": float(self.curriculum_last_batch_termination_count),
            "decision_success_ratio": float(self.curriculum_decision_success_ratio),
            "decision_window_size": float(self.curriculum_decision_window_size),
            "rolling_success_ratio": float(rolling_ratio),
            "rolling_window_size": float(rolling_count),
            "success_threshold": float(self.curriculum_success_threshold),
            "promotion_count": float(self.curriculum_promotion_count),
        }

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
    # goal visualizers
    # ---------------------------------------------------------------------
    def _get_stage(self):
        try:
            return self.sim.stage
        except Exception:
            pass
        try:
            return self.scene.stage
        except Exception:
            pass
        raise RuntimeError("Unable to access USD stage from environment")

    def _goal_vis_path(self, env_index: int) -> str:
        return f"/World/envs/env_{env_index}/GoalVis"

    def _create_goal_visualizers(self) -> None:
        stage = self._get_stage()
        self._goal_vis_paths = []

        for env_index in range(int(self.num_envs)):
            goal_path = self._goal_vis_path(env_index)
            sphere = UsdGeom.Sphere.Define(stage, goal_path)
            sphere.CreateRadiusAttr(float(self._goal_vis_radius))

            prim = sphere.GetPrim()
            _set_prim_translation(stage, prim.GetPath().pathString, (0.0, 0.0, -1000.0))

            sphere.CreateDisplayColorAttr([Gf.Vec3f(*self._goal_vis_color)])
            sphere.CreateDisplayOpacityAttr([float(self._goal_vis_opacity)])

            self._goal_vis_paths.append(goal_path)

        print(f"[MyDroneRLEnv] Created {len(self._goal_vis_paths)} goal visualizers", flush=True)

    def _update_goal_visualizers(self, env_ids: torch.Tensor | None = None) -> None:
        if not self._goal_vis_enabled:
            return
        if len(self._goal_vis_paths) == 0:
            return

        stage = self._get_stage()

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        env_ids_cpu = env_ids.detach().to("cpu").tolist()
        goal_cpu = self.goal_pos_w.detach().to("cpu")

        for env_id in env_ids_cpu:
            if env_id < 0 or env_id >= len(self._goal_vis_paths):
                continue

            gx, gy, gz = goal_cpu[env_id].tolist()
            _set_prim_translation(stage, self._goal_vis_paths[env_id], (float(gx), float(gy), float(gz)))

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
        # gz = torch.rand(n, device=self.device) * (goal_z_max - goal_z_min) + goal_z_min
        gz = 5.0
        
        self.goal_pos_w[env_ids, 0] = gx
        self.goal_pos_w[env_ids, 1] = gy
        self.goal_pos_w[env_ids, 2] = gz

        self._update_goal_visualizers(env_ids)

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

        self._refresh_energy_prev_buffers(env_ids)
        self._refresh_progress_prev_dist(env_ids)

        goal_info = self._build_goal_info()

        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            out[1].update(goal_info)
            return out

        return out

    def reset_idx(self, env_ids: torch.Tensor | None = None):
        return self._reset_idx(env_ids)
