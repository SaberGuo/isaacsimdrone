# SPDX-License-Identifier: BSD-3-Clause

"""
Test script: spawn multi-env UAVs, each with a Livox Mid-360 LiDAR.

Run:
  cd ~/hjr_isaacdrone_ws/IsaacLab-2.1.0
  ./isaaclab.sh -p ../omniperception_isaacdrone/scripts/test1_multi_env.py --num_envs 4
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback

import torch
from isaaclab.app import AppLauncher

# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser(description="Test multi-env UAV + LiDAR debug visualization")

parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--env_spacing", type=float, default=6.0)
parser.add_argument("--sim_dt", type=float, default=0.01)
parser.add_argument("--reset_interval", type=int, default=500)
parser.add_argument("--print_every", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)

# LiDAR controls
parser.add_argument("--lidar_samples", type=int, default=24000)
parser.add_argument("--lidar_update_hz", type=float, default=25.0)
parser.add_argument("--lidar_offset_z", type=float, default=0.1)
parser.add_argument("--lidar_max_distance", type=float, default=20.0)
parser.add_argument("--lidar_min_range", type=float, default=0.2)
parser.add_argument("--lidar_ray_alignment", type=str, default="base", choices=["base", "yaw", "world"])

parser.add_argument("--lidar_debug_vis", dest="lidar_debug_vis", action="store_true")
parser.add_argument("--no_lidar_debug_vis", dest="lidar_debug_vis", action="store_false")
parser.set_defaults(lidar_debug_vis=True)

parser.add_argument("--lidar_return_pointcloud", action="store_true", default=False)

# LiDAR print controls
parser.add_argument("--lidar_print_interval", type=int, default=100)
parser.add_argument("--lidar_print_env", type=int, default=0)
parser.add_argument("--lidar_show_histogram", dest="lidar_show_histogram", action="store_true")
parser.add_argument("--no_lidar_show_histogram", dest="lidar_show_histogram", action="store_false")
parser.set_defaults(lidar_show_histogram=True)
parser.add_argument("--lidar_histogram_bins", type=int, default=10)
parser.add_argument("--lidar_save_snapshots", action="store_true", default=False)
parser.add_argument("--lidar_snapshot_dir", type=str, default="./lidar_snapshots")

# obstacles
parser.add_argument("--spawn_obstacles", dest="spawn_obstacles", action="store_true")
parser.add_argument("--no_spawn_obstacles", dest="spawn_obstacles", action="store_false")
parser.set_defaults(spawn_obstacles=True)
parser.add_argument("--obstacles_per_env", type=int, default=6)

# 添加 IsaacLab launcher 参数并解析
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------
# Imports AFTER app launch
# ----------------------------
print("[DEBUG] Importing modules after app launch...")

from pxr import UsdGeom
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from omniperception_isaacdrone.envs import (
    MultiEnvUavLidarSceneCfg,
    build_multi_env_uav_lidar_scene,
    reset_uavs_and_lidars,
    safe_lidar_update,
)

# 从 utils 导入可视化相关（这些在 lidar_visualizer.py 中）
from omniperception_isaacdrone.utils import (
    print_lidar_stats,
    save_lidar_snapshot,
)


print("[DEBUG] All modules imported successfully")


def _set_camera_for_grid(sim: SimulationContext, num_envs: int, spacing: float):
    """Set viewport camera to see all envs."""
    grid = int(math.ceil(math.sqrt(max(num_envs, 1))))
    half = (grid - 1) * spacing * 0.5
    eye = [half * 2.2, half * 2.2, max(3.0, half * 1.2)]
    target = [0.0, 0.0, 0.5]
    sim.set_camera_view(eye=eye, target=target)


def main():
    print("[DEBUG] Entering main()")
    
    # ----------------------------
    # Simulation context
    # ----------------------------
    print("[DEBUG] Creating SimulationContext...")
    sim_cfg = sim_utils.SimulationCfg(
        dt=args_cli.sim_dt,
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    print("[DEBUG] SimulationContext created")

    # ----------------------------
    # Build scene cfg from CLI
    # ----------------------------
    print("[DEBUG] Creating scene config...")
    scene_cfg = MultiEnvUavLidarSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=args_cli.env_spacing,
        lidar_samples=args_cli.lidar_samples,
        lidar_update_hz=args_cli.lidar_update_hz,
        lidar_offset_z_m=args_cli.lidar_offset_z,
        lidar_max_distance=args_cli.lidar_max_distance,
        lidar_min_range=args_cli.lidar_min_range,
        lidar_ray_alignment=args_cli.lidar_ray_alignment,
        lidar_debug_vis=args_cli.lidar_debug_vis,
        lidar_return_pointcloud=args_cli.lidar_return_pointcloud,
        spawn_obstacles=args_cli.spawn_obstacles,
        obstacles_per_env=args_cli.obstacles_per_env,
    )
    print(f"[DEBUG] Scene config: num_envs={scene_cfg.num_envs}, spacing={scene_cfg.env_spacing}")

    # Build scene - 不传 seed 参数
    print("[DEBUG] Building scene...")
    scene = build_multi_env_uav_lidar_scene(scene_cfg)
    print("[DEBUG] Scene built successfully")
    
    # 从 scene 字典中提取组件
    uavs = scene["uavs"]
    lidars = scene["lidars"]
    env_origins = scene["env_origins"]
    
    print(f"[INFO] Created {len(uavs)} UAVs + {len(lidars)} LiDARs across {scene_cfg.num_envs} envs.")

    # ----------------------------
    # Play sim
    # ----------------------------
    print("[DEBUG] Calling sim.reset()...")
    sim.reset()
    print("[DEBUG] sim.reset() completed")
    
    _set_camera_for_grid(sim, scene_cfg.num_envs, scene_cfg.env_spacing)

    # Initial reset - 传入整个 scene 字典
    print("[DEBUG] Calling reset_uavs_and_lidars()...")
    reset_uavs_and_lidars(scene, seed=args_cli.seed)
    print("[DEBUG] Initial reset completed")

    print("[INFO] Simulation running. Press Ctrl+C to exit.")

    count = 0
    dt = sim_cfg.dt

    while simulation_app.is_running():
        # Step physics
        sim.step()
        count += 1

        # Update LiDARs
        for lidar in lidars:
            safe_lidar_update(lidar, dt)

        # Write UAV data
        for uav in uavs:
            uav.write_data_to_sim()

        # ----------------------------
        # LiDAR stats printing
        # ----------------------------
        if args_cli.lidar_print_interval > 0 and count % args_cli.lidar_print_interval == 0:
            env_idx = args_cli.lidar_print_env
            if env_idx < len(lidars):
                lidar = lidars[env_idx]
                
                print_lidar_stats(
                    lidar,
                    env_idx=env_idx,
                    step=count,
                    show_histogram=args_cli.lidar_show_histogram,
                    histogram_bins=args_cli.lidar_histogram_bins,
                )
                
                if args_cli.lidar_save_snapshots:
                    save_lidar_snapshot(
                        lidar,
                        env_idx=env_idx,
                        step=count,
                        output_dir=args_cli.lidar_snapshot_dir,
                        save_pointcloud=args_cli.lidar_return_pointcloud,
                        save_distances=True,
                        save_stats=True,
                    )

        # Periodic status print
        if args_cli.print_every > 0 and count % args_cli.print_every == 0:
            if args_cli.lidar_print_interval <= 0 or count % args_cli.lidar_print_interval != 0:
                print(f"[step {count}] simulation running...")

        # Periodic reset
        if args_cli.reset_interval > 0 and count % args_cli.reset_interval == 0:
            print(f"[step {count}] resetting all envs...")
            reset_uavs_and_lidars(scene, seed=args_cli.seed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Exiting via Ctrl+C...")
    except Exception as e:
        print(f"\n[ERROR] Unhandled exception: {e}")
        traceback.print_exc()
    finally:
        print("[DEBUG] Closing simulation app...")
        simulation_app.close()
