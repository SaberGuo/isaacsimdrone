# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random
from typing import Any, Iterable

import torch

import omni.usd
from pxr import UsdGeom

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

from isaaclab.sensors import LidarSensor, LidarSensorCfg
from isaaclab.sensors.ray_caster.patterns import LivoxPatternCfg


@dataclass
class MultiEnvUavLidarSceneCfg:
    # multi-env
    num_envs: int = 4
    env_spacing: float = 6.0
    env_root_path: str = "/World/Envs"

    # uav
    uav_init_height: float = 1.0
    uav_scale_xyz: tuple[float, float, float] = (5.43, 5.43, 10.34)

    # keep minimal training stable
    uav_disable_gravity: bool = True
    uav_kinematic: bool = False

    # reset randomization
    init_xy_noise: float = 0.5
    init_yaw_range: float = math.pi

    # obstacles
    spawn_obstacles: bool = True
    obstacles_per_env: int = 6
    obstacle_xy_range: float = 2.0
    obstacle_size_min: float = 0.15
    obstacle_size_max: float = 0.60

    # lidar (NOTE: default samples reduced for stability)
    lidar_offset_z_m: float = 0.10
    lidar_samples: int = 4096
    lidar_update_hz: float = 25.0
    lidar_max_distance: float = 20.0
    lidar_min_range: float = 0.20
    lidar_ray_alignment: str = "base"  # base|yaw|world
    lidar_debug_vis: bool = True
    lidar_return_pointcloud: bool = False


def get_stage_meters_per_unit() -> float:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return 1.0
    return float(UsdGeom.GetStageMetersPerUnit(stage))


def get_uav_asset_path_fallback() -> str:
    # 1) package path
    try:
        import omniperception_isaacdrone as pkg  # type: ignore
        pkg_dir = os.path.dirname(os.path.abspath(pkg.__file__))
        usd_path = os.path.join(pkg_dir, "assets", "robots", "cf2x.usd")
        if os.path.isfile(usd_path):
            return usd_path
    except Exception:
        pass

    # 2) workspace fallback
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_guess = os.path.abspath(os.path.join(this_dir, "..", "..", "..", "..", ".."))
    usd_path = os.path.join(
        project_root_guess,
        "omniperception_isaacdrone/source/omniperception_isaacdrone/omniperception_isaacdrone/assets/robots/cf2x.usd",
    )
    if os.path.isfile(usd_path):
        return usd_path

    raise FileNotFoundError(f"Cannot find cf2x.usd. Last tried: {usd_path}")


def safe_lidar_update(lidar: "LidarSensor", dt: float) -> None:
    if not hasattr(lidar, "update"):
        return
    candidates = (
        lambda: lidar.update(dt, force_recompute=True),
        lambda: lidar.update(dt, force_compute=True),
        lambda: lidar.update(dt),
        lambda: lidar.update(force_recompute=True),
        lambda: lidar.update(force_compute=True),
        lambda: lidar.update(),
    )
    for fn in candidates:
        try:
            fn()
            return
        except TypeError:
            continue
        except Exception:
            return


def compute_env_origins(num_envs: int, spacing: float) -> torch.Tensor:
    if num_envs <= 0:
        return torch.zeros((0, 3), dtype=torch.float32)
    grid = int(math.ceil(math.sqrt(num_envs)))
    origins = []
    half = (grid - 1) * spacing * 0.5
    for i in range(num_envs):
        r = i // grid
        c = i % grid
        x = c * spacing - half
        y = r * spacing - half
        origins.append([x, y, 0.0])
    return torch.tensor(origins, dtype=torch.float32)


def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    cy = torch.cos(half)
    sy = torch.sin(half)
    zeros = torch.zeros_like(cy)
    return torch.stack([cy, zeros, zeros, sy], dim=-1)


def spawn_common_world() -> None:
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    cfg_dome_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg_dome_light.func("/World/Light", cfg_dome_light)


def _spawn_random_obstacles_for_env(env_ns: str, env_origin_xy: tuple[float, float], cfg: MultiEnvUavLidarSceneCfg) -> list[str]:
    obstacle_paths: list[str] = []
    if not cfg.spawn_obstacles or cfg.obstacles_per_env <= 0:
        return obstacle_paths

    prim_utils.create_prim(f"{env_ns}/Obstacles", "Xform")

    static_rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
    )
    collision_props = sim_utils.CollisionPropertiesCfg()

    def _rand(a: float, b: float) -> float:
        return float(a + (b - a) * random.random())

    ox, oy = env_origin_xy

    for k in range(cfg.obstacles_per_env):
        kind = random.choice(["box", "cyl", "sphere", "wall"])
        dx = _rand(-cfg.obstacle_xy_range, cfg.obstacle_xy_range)
        dy = _rand(-cfg.obstacle_xy_range, cfg.obstacle_xy_range)

        prim_path = f"{env_ns}/Obstacles/obs_{k:02d}"

        if kind == "box":
            sx = _rand(cfg.obstacle_size_min, cfg.obstacle_size_max)
            sy = _rand(cfg.obstacle_size_min, cfg.obstacle_size_max)
            sz = _rand(cfg.obstacle_size_min, cfg.obstacle_size_max)
            z = 0.5 * sz
            spawn_cfg = sim_utils.CuboidCfg(
                size=(sx, sy, sz),
                rigid_props=static_rigid_props,
                collision_props=collision_props,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
            )
        elif kind == "cyl":
            r = _rand(0.10, 0.30)
            h = _rand(0.30, 1.20)
            z = 0.5 * h
            spawn_cfg = sim_utils.CylinderCfg(
                radius=r,
                height=h,
                axis="Z",
                rigid_props=static_rigid_props,
                collision_props=collision_props,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            )
        elif kind == "sphere":
            r = _rand(0.12, 0.35)
            z = r
            spawn_cfg = sim_utils.SphereCfg(
                radius=r,
                rigid_props=static_rigid_props,
                collision_props=collision_props,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.9)),
            )
        else:
            sx = _rand(1.0, 2.5)
            sy = _rand(0.04, 0.10)
            sz = _rand(0.6, 1.5)
            z = 0.5 * sz
            spawn_cfg = sim_utils.CuboidCfg(
                size=(sx, sy, sz),
                rigid_props=static_rigid_props,
                collision_props=collision_props,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.2)),
            )

        obj_cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(ox + dx, oy + dy, z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        _ = RigidObject(cfg=obj_cfg)
        obstacle_paths.append(prim_path)

    return obstacle_paths


def build_multi_env_uav_lidar_scene(cfg: MultiEnvUavLidarSceneCfg, meters_per_unit: float | None = None) -> dict[str, Any]:
    if meters_per_unit is None:
        meters_per_unit = get_stage_meters_per_unit()

    spawn_common_world()
    prim_utils.create_prim(cfg.env_root_path, "Xform")

    env_origins = compute_env_origins(cfg.num_envs, cfg.env_spacing)
    uav_usd_path = get_uav_asset_path_fallback()

    uavs: list[Articulation] = []
    lidars: list[LidarSensor] = []
    obstacle_paths_per_env: list[list[str]] = []
    env_paths: list[str] = []

    for i in range(cfg.num_envs):
        env_ns = f"{cfg.env_root_path}/env_{i:03d}"
        env_paths.append(env_ns)
        prim_utils.create_prim(env_ns, "Xform")

        origin = env_origins[i].tolist()
        ox, oy = float(origin[0]), float(origin[1])

        obstacle_paths = _spawn_random_obstacles_for_env(env_ns, (ox, oy), cfg)
        obstacle_paths_per_env.append(obstacle_paths)

        sx, sy, sz = cfg.uav_scale_xyz
        uav_cfg = ArticulationCfg(
            prim_path=f"{env_ns}/UAV",
            spawn=sim_utils.UsdFileCfg(
                usd_path=uav_usd_path,
                scale=(sx, sy, sz),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=bool(cfg.uav_disable_gravity),
                    kinematic_enabled=bool(cfg.uav_kinematic),
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(ox, oy, float(cfg.uav_init_height)),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            actuators={},
        )
        uav = Articulation(cfg=uav_cfg)
        uavs.append(uav)

        # LiDAR
        update_period = 1.0 / max(float(cfg.lidar_update_hz), 1e-6)
        desired_offset_stage = float(cfg.lidar_offset_z_m) / max(float(meters_per_unit), 1e-9)
        lidar_offset_local_z = desired_offset_stage / float(sz)

        mesh_targets = ["/World/defaultGroundPlane"] + obstacle_paths

        lidar_cfg = LidarSensorCfg(
            prim_path=f"{env_ns}/UAV",
            offset=LidarSensorCfg.OffsetCfg(
                pos=(0.0, 0.0, float(lidar_offset_local_z)),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            attach_yaw_only=False,
            ray_alignment=str(cfg.lidar_ray_alignment),
            pattern_cfg=LivoxPatternCfg(sensor_type="mid360", samples=int(cfg.lidar_samples)),
            mesh_prim_paths=mesh_targets,
            max_distance=float(cfg.lidar_max_distance),
            min_range=float(cfg.lidar_min_range),
            return_pointcloud=bool(cfg.lidar_return_pointcloud),
            pointcloud_in_world_frame=True,
            enable_sensor_noise=False,
            random_distance_noise=0.0,
            update_period=update_period,
            update_frequency=float(cfg.lidar_update_hz),
            debug_vis=bool(cfg.lidar_debug_vis),
        )
        lidar = LidarSensor(cfg=lidar_cfg)
        lidars.append(lidar)

    return {
        "cfg": cfg,
        "env_paths": env_paths,
        "env_origins": env_origins,  # CPU tensor
        "uavs": uavs,
        "lidars": lidars,
        "obstacle_paths_per_env": obstacle_paths_per_env,
        "meters_per_unit": float(meters_per_unit),
    }


def reset_uavs_and_lidars(scene: dict[str, Any], seed: int | None = None) -> None:
    if seed is not None:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
    cfg: MultiEnvUavLidarSceneCfg = scene["cfg"]
    env_origins: torch.Tensor = scene["env_origins"]
    uavs: list[Articulation] = scene["uavs"]
    lidars: list[LidarSensor] = scene["lidars"]
    for i in range(len(uavs)):
        _reset_one(cfg, env_origins, uavs, lidars, i)


def reset_uavs_and_lidars_idx(scene: dict[str, Any], env_ids: Iterable[int], seed: int | None = None) -> None:
    if seed is not None:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
    cfg: MultiEnvUavLidarSceneCfg = scene["cfg"]
    env_origins: torch.Tensor = scene["env_origins"]
    uavs: list[Articulation] = scene["uavs"]
    lidars: list[LidarSensor] = scene["lidars"]
    for i in env_ids:
        i = int(i)
        if 0 <= i < len(uavs):
            _reset_one(cfg, env_origins, uavs, lidars, i)


def _reset_one(cfg: MultiEnvUavLidarSceneCfg, env_origins: torch.Tensor, uavs: list[Articulation], lidars: list[LidarSensor], i: int) -> None:
    uav = uavs[i]
    origin = env_origins[i].to(device=uav.device)

    noise_xy = (torch.rand((2,), device=uav.device) * 2.0 - 1.0) * float(cfg.init_xy_noise)
    pos = torch.tensor([origin[0], origin[1], 0.0], device=uav.device)
    pos[0] += noise_xy[0]
    pos[1] += noise_xy[1]
    pos[2] = float(cfg.uav_init_height)

    yaw = (torch.rand((1,), device=uav.device) * 2.0 - 1.0) * float(cfg.init_yaw_range)
    quat = yaw_to_quat_wxyz(yaw).squeeze(0)

    root_state = uav.data.default_root_state.clone()
    root_state[:, 0:3] = pos.unsqueeze(0)
    root_state[:, 3:7] = quat.unsqueeze(0)
    uav.write_root_state_to_sim(root_state)

    joint_pos = uav.data.default_joint_pos.clone()
    joint_vel = uav.data.default_joint_vel.clone()
    uav.write_joint_state_to_sim(joint_pos, joint_vel)

    uav.reset()
    try:
        lidars[i].reset()
    except Exception:
        pass


def try_get_lidar_ranges(lidar: "LidarSensor") -> torch.Tensor | None:
    """
    Best-effort: return 1D ranges tensor if available, else None.
    Different IsaacLab/OmniPerception builds name fields differently.
    """
    if not hasattr(lidar, "data"):
        return None
    data = lidar.data
    for name in ["ranges", "distances", "range", "distance", "ray_distances", "ray_ranges"]:
        if hasattr(data, name):
            v = getattr(data, name)
            if isinstance(v, torch.Tensor):
                if v.ndim == 2 and v.shape[0] == 1:
                    return v[0]
                if v.ndim == 1:
                    return v
                return v.reshape(-1)
    return None
