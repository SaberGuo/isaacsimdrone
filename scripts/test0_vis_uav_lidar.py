# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visualize Crazyflie UAV with a Livox Mid-360 LiDAR sensor in an empty scene,
and spawn several nearby obstacle geometries with collision volumes.

What you should see:
- UAV at /World/UAV
- LiDAR debug rays + hit points (if supported) scanning:
    - ground plane
    - /World/Obstacles/* (Box/Cylinder/Sphere/Wall)

Run:
    cd ~/hjr_isaacdrone_ws/IsaacLab-2.1.0
    ./isaaclab.sh -p ../omniperception_isaacdrone/scripts/vis_uav_lidar_mid360.py

Useful args:
    --lidar_debug_vis           (default: ON)
    --no_lidar_debug_vis
    --lidar_samples 24000
    --lidar_update_hz 25
    --lidar_offset_z 0.1
    --lidar_max_distance 20
    --lidar_ray_alignment base|yaw|world
"""

from __future__ import annotations

import argparse
import os

import torch
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Visualize UAV + Livox Mid-360 LiDAR + nearby obstacles.")

parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)

# LiDAR options
parser.add_argument("--lidar_offset_z", type=float, default=0.1, help="LiDAR offset above UAV root (m, world desired).")
parser.add_argument("--lidar_samples", type=int, default=24000, help="Number of rays per scan for Livox Mid-360.")
parser.add_argument("--lidar_update_hz", type=float, default=25.0, help="LiDAR update frequency (Hz).")
parser.add_argument("--lidar_max_distance", type=float, default=20.0, help="LiDAR max distance (m).")
parser.add_argument("--lidar_min_range", type=float, default=0.2, help="LiDAR min range (m).")
parser.add_argument(
    "--lidar_ray_alignment",
    type=str,
    default="base",
    choices=["base", "yaw", "world"],
    help="Ray alignment frame. base: follow full UAV orientation; yaw: follow yaw only; world: fixed world frame.",
)

parser.add_argument(
    "--lidar_debug_vis",
    dest="lidar_debug_vis",
    action="store_true",
    help="Enable LiDAR debug visualization (rays/hit points).",
)
parser.add_argument(
    "--no_lidar_debug_vis",
    dest="lidar_debug_vis",
    action="store_false",
    help="Disable LiDAR debug visualization.",
)
parser.set_defaults(lidar_debug_vis=True)

parser.add_argument(
    "--lidar_return_pointcloud",
    action="store_true",
    default=False,
    help="If supported, return pointcloud instead of distances (more data, slower).",
)

# Obstacle options (positions are in WORLD meters)
parser.add_argument("--spawn_obstacles", action="store_true", default=True, help="Spawn nearby obstacles.")
parser.add_argument("--no_spawn_obstacles", dest="spawn_obstacles", action="store_false", help="Do not spawn obstacles.")

parser.add_argument("--print_every", type=int, default=200, help="Print info every N sim steps. 0 to disable.")
parser.add_argument("--reset_interval", type=int, default=500, help="Reset UAV every N simulation steps.")
parser.add_argument("--sim_dt", type=float, default=0.01, help="Physics timestep (s).")

# Append AppLauncher CLI args (adds --headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports that must happen AFTER app launch
# -----------------------------------------------------------------------------
from pxr import UsdGeom  # noqa: E402
import omni.usd  # noqa: E402

import isaacsim.core.utils.prims as prim_utils  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

# LiDAR (OmniPerception / IsaacLab integration)
try:
    from isaaclab.sensors import LidarSensor, LidarSensorCfg  # noqa: E402
    from isaaclab.sensors.ray_caster.patterns import LivoxPatternCfg  # noqa: E402
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import LidarSensor/LidarSensorCfg or LivoxPatternCfg.\n"
        "Please confirm your IsaacLab build includes OmniPerception LiDAR integration."
    ) from e


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_uav_asset_path() -> str:
    """Get absolute path to cf2x.usd in this project."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    usd_path = os.path.join(
        project_root,
        "source/omniperception_isaacdrone/omniperception_isaacdrone/assets/robots/cf2x.usd",
    )
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"UAV USD model not found: {usd_path}")
    return usd_path


def safe_lidar_update(lidar: "LidarSensor", dt: float) -> None:
    """Best-effort lidar.update(...) across minor API variations."""
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


def _try_get_lidar_num_rays(lidar: "LidarSensor") -> str:
    if hasattr(lidar, "num_rays"):
        try:
            return str(lidar.num_rays)
        except Exception:
            return "unknown"
    return "unknown"


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------
def design_scene(meters_per_unit: float) -> dict[str, object]:
    """Spawn ground/light/UAV, obstacles, and create LiDAR sensor."""
    # Ground
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Light
    cfg_dome_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg_dome_light.func("/World/Light", cfg_dome_light)

    # UAV
    uav_model_path = get_uav_asset_path()

    # Same scaling you used earlier (note: this scales the whole UAV prim)
    scale_x, scale_y, scale_z = 5.43, 5.43, 10.34

    uav_cfg = ArticulationCfg(
        prim_path="/World/UAV",
        spawn=sim_utils.UsdFileCfg(
            usd_path=uav_model_path,
            scale=(scale_x, scale_y, scale_z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={},
    )
    uav = Articulation(cfg=uav_cfg)

    # Obstacles (rigid bodies with collision, kept fixed/kinematic)
    obstacle_mesh_paths: list[str] = []
    obstacles: dict[str, RigidObject] = {}

    if bool(args_cli.spawn_obstacles):
        prim_utils.create_prim("/World/Obstacles", "Xform")

        static_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        )
        collision_props = sim_utils.CollisionPropertiesCfg()

        def _spawn_obstacle(name: str, prim_path: str, spawn_cfg, pos_xyz):
            cfg = RigidObjectCfg(
                prim_path=prim_path,
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=pos_xyz,
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            obj = RigidObject(cfg=cfg)
            obstacles[name] = obj
            obstacle_mesh_paths.append(prim_path)

        # Box near UAV (right)
        box_spawn = sim_utils.CuboidCfg(
            size=(0.35, 0.35, 0.35),
            rigid_props=static_rigid_props,
            collision_props=collision_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        )
        _spawn_obstacle("box1", "/World/Obstacles/Box1", box_spawn, pos_xyz=(0.8, 0.0, 0.175))

        # Cylinder near UAV (front)
        cyl_spawn = sim_utils.CylinderCfg(
            radius=0.18,
            height=0.6,
            axis="Z",
            rigid_props=static_rigid_props,
            collision_props=collision_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
        )
        _spawn_obstacle("cyl1", "/World/Obstacles/Cylinder1", cyl_spawn, pos_xyz=(0.0, 0.9, 0.30))

        # Sphere near UAV (left)
        sph_spawn = sim_utils.SphereCfg(
            radius=0.22,
            rigid_props=static_rigid_props,
            collision_props=collision_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.9)),
        )
        _spawn_obstacle("sphere1", "/World/Obstacles/Sphere1", sph_spawn, pos_xyz=(-0.8, 0.0, 0.22))

        # Thin wall behind UAV
        wall_spawn = sim_utils.CuboidCfg(
            size=(1.6, 0.06, 1.0),
            rigid_props=static_rigid_props,
            collision_props=collision_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.2)),
        )
        _spawn_obstacle("wall", "/World/Obstacles/Wall", wall_spawn, pos_xyz=(0.0, -1.0, 0.50))

    # LiDAR sensor (Livox Mid-360)
    update_period = 1.0 / max(float(args_cli.lidar_update_hz), 1e-6)

    # You want WORLD offset (meters). Because UAV is scaled, compensate local Z by parent scale_z.
    # Stage units are in meters_per_unit, so convert meters -> stage units first.
    desired_offset_world_m = float(args_cli.lidar_offset_z)
    desired_offset_stage = desired_offset_world_m / max(float(meters_per_unit), 1e-9)
    lidar_offset_local_z = desired_offset_stage / float(scale_z)

    mesh_targets = ["/World/defaultGroundPlane"] + obstacle_mesh_paths

    lidar_cfg = LidarSensorCfg(
        prim_path="/World/UAV",
        offset=LidarSensorCfg.OffsetCfg(
            pos=(0.0, 0.0, float(lidar_offset_local_z)),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        attach_yaw_only=False,
        ray_alignment=args_cli.lidar_ray_alignment,
        pattern_cfg=LivoxPatternCfg(sensor_type="mid360", samples=int(args_cli.lidar_samples)),
        # Raycast targets: ground + obstacles (THIS is the key to scan them)
        mesh_prim_paths=mesh_targets,
        max_distance=float(args_cli.lidar_max_distance),
        min_range=float(args_cli.lidar_min_range),
        return_pointcloud=bool(args_cli.lidar_return_pointcloud),
        pointcloud_in_world_frame=True,
        enable_sensor_noise=False,
        random_distance_noise=0.0,
        update_period=update_period,
        update_frequency=float(args_cli.lidar_update_hz),
        debug_vis=bool(args_cli.lidar_debug_vis),
    )
    lidar = LidarSensor(cfg=lidar_cfg)

    return {"uav": uav, "lidar": lidar, "obstacles": obstacles}


# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------
def run_sim(sim: SimulationContext, entities: dict[str, object]):
    uav: Articulation = entities["uav"]
    lidar: "LidarSensor" = entities["lidar"]

    sim_dt = sim.get_physics_dt()
    count = 0

    print("[INFO]: Entering simulation loop...")
    while simulation_app.is_running():
        # Periodic reset
        if int(args_cli.reset_interval) > 0 and count % int(args_cli.reset_interval) == 0:
            root_state = uav.data.default_root_state.clone()
            root_state[:, :3] = torch.tensor([0.0, 0.0, 1.0], device=uav.device)
            uav.write_root_state_to_sim(root_state)

            joint_pos = uav.data.default_joint_pos.clone()
            joint_vel = uav.data.default_joint_vel.clone()
            uav.write_joint_state_to_sim(joint_pos, joint_vel)

            uav.reset()

            try:
                lidar.reset()
            except Exception:
                pass

            print("[INFO]: Reset UAV and LiDAR.")

        # Push pending commands (none here)
        uav.write_data_to_sim()

        # Step physics
        sim.step()

        # Update buffers
        uav.update(sim_dt)

        # Update LiDAR
        safe_lidar_update(lidar, sim_dt)

        # Print status
        if int(args_cli.print_every) > 0 and count % int(args_cli.print_every) == 0:
            uav_pos = uav.data.root_pos_w[0, :3].detach().cpu().numpy()
            msg = f"[INFO]: step={count:06d} | UAV pos={uav_pos}"

            if hasattr(lidar, "is_initialized") and getattr(lidar, "is_initialized"):
                msg += f" | LiDAR initialized | num_rays={_try_get_lidar_num_rays(lidar)}"
            else:
                msg += " | LiDAR (not initialized yet)"

            print(msg)

        count += 1


def main():
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=float(args_cli.sim_dt), device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Stage units
    stage = omni.usd.get_context().get_stage()
    meters_per_unit = 1.0
    if stage:
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        print("--------------------------------------------------")
        print(f"[INFO]: Stage units: 1 unit = {meters_per_unit} meters")
        print("--------------------------------------------------")

    # Camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.5])

    # Scene
    entities = design_scene(meters_per_unit)

    # Start sim
    sim.reset()
    print("[INFO]: Setup complete.")

    run_sim(sim, entities)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR]: {e}")
        raise
    finally:
        simulation_app.close()
