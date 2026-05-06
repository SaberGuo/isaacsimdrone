from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)

parser = argparse.ArgumentParser("Validate LiDAR body-frame output for test6 drone env")
parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-ScanLidar-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--obstacle_distance", type=float, default=10.0)
parser.add_argument("--radius_min", type=float, default=8.0)
parser.add_argument("--radius_max", type=float, default=12.5)
parser.add_argument("--error_threshold", type=float, default=2.5)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import omniperception_isaacdrone.tasks.test_registry as _test_registry  # noqa: F401
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils import parse_env_cfg

from omniperception_isaacdrone.envs.test_env import WallSpawner, setup_global_obstacles


def euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return torch.tensor(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=torch.float32,
    )


def set_obstacle_pose(env, obstacle_index: int, position_w: torch.Tensor) -> None:
    obstacles = env.scene["obstacles"]
    env_ids = torch.tensor([int(obstacle_index)], device=env.device, dtype=torch.long)
    pose = torch.zeros((1, 7), device=env.device, dtype=torch.float32)
    pose[:, 0:3] = position_w.unsqueeze(0)
    pose[:, 3] = 1.0
    velocity = torch.zeros((1, 6), device=env.device, dtype=torch.float32)
    obstacles.write_root_pose_to_sim(pose, env_ids=env_ids)
    obstacles.write_root_velocity_to_sim(velocity, env_ids=env_ids)


def sample_case(
    env,
    obstacle_pos_w: torch.Tensor,
    name: str,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    radius_min: float,
    radius_max: float,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    lidar = env.scene["lidar"]

    pose = torch.zeros((1, 7), device=env.device, dtype=torch.float32)
    vel = torch.zeros((1, 6), device=env.device, dtype=torch.float32)
    pose[:, 0:3] = torch.tensor([[0.0, 0.0, 5.0]], device=env.device)
    pose[:, 3:7] = euler_xyz_to_quat_wxyz(
        math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    ).to(env.device)

    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(vel)

    env.sim.step()
    env.scene.update(dt=env.physics_dt)
    lidar.update(env.physics_dt, force_recompute=True)

    pointcloud = lidar.get_pointcloud(torch.tensor([0], device=env.device))[0]
    valid = torch.isfinite(pointcloud).all(dim=-1)
    ranges = torch.linalg.norm(pointcloud, dim=-1)
    valid &= (ranges >= float(radius_min)) & (ranges <= float(radius_max))
    if not valid.any():
        raise RuntimeError(f"[LIDAR_TEST] no valid points for case {name}")

    centroid = pointcloud[valid].mean(dim=0)
    expected = quat_apply_inverse(pose[:, 3:7], obstacle_pos_w.unsqueeze(0) - pose[:, 0:3])[0]
    error = float(torch.linalg.norm(centroid - expected).item())
    num_points = int(valid.sum().item())

    print(
        f"[LIDAR_TEST] case={name} expected={expected.detach().cpu().tolist()} "
        f"centroid={centroid.detach().cpu().tolist()} error={error:.6f} "
        f"num_points={num_points}",
        flush=True,
    )
    return error, centroid, expected


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=int(args_cli.num_envs),
        use_fabric=not args_cli.disable_fabric,
    )
    # Isolate the obstacle mesh for this diagnostic so ceiling / ground returns cannot pollute the result.
    env_cfg.scene.lidar.mesh_prim_paths = ["/World/Obstacles"]
    WallSpawner(
        x_bounds=(-80.0, 80.0),
        y_bounds=(-80.0, 80.0),
        z_bounds=(0.0, 10.0),
        wall_thickness=0.5,
    ).spawn_walls()
    setup_global_obstacles(1)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    try:
        env.reset(seed=int(args_cli.seed))

        lidar = env.scene["lidar"]
        print(
            f"[LIDAR_TEST] ray_alignment={getattr(lidar.cfg, 'ray_alignment', 'unknown')} "
            f"pointcloud_in_world_frame={getattr(lidar.cfg, 'pointcloud_in_world_frame', 'unknown')} "
            f"update_frequency={getattr(lidar.cfg, 'update_frequency', 'unknown')} "
            f"update_period={getattr(lidar.cfg, 'update_period', 'unknown')} "
            f"mesh_prim_paths={getattr(lidar.cfg, 'mesh_prim_paths', 'unknown')}",
            flush=True,
        )

        obstacle_pos_w = torch.tensor(
            [float(args_cli.obstacle_distance), 0.0, 5.0],
            device=env.device,
            dtype=torch.float32,
        )
        set_obstacle_pose(env, obstacle_index=0, position_w=obstacle_pos_w)

        cases = [
            ("identity", 0.0, 0.0, 0.0),
            ("yaw_90", 0.0, 0.0, 90.0),
            ("pitch_20", 0.0, 20.0, 0.0),
            ("roll_20", 20.0, 0.0, 0.0),
            ("rpy_mix", 15.0, -12.0, 60.0),
        ]

        errors = []
        for name, roll_deg, pitch_deg, yaw_deg in cases:
            error, _, _ = sample_case(
                env=env,
                obstacle_pos_w=obstacle_pos_w,
                name=name,
                roll_deg=roll_deg,
                pitch_deg=pitch_deg,
                yaw_deg=yaw_deg,
                radius_min=float(args_cli.radius_min),
                radius_max=float(args_cli.radius_max),
            )
            errors.append(error)

        max_error = max(errors)
        passed = max_error < float(args_cli.error_threshold)
        print(f"[LIDAR_TEST] max_error={max_error:.6f}", flush=True)
        print(f"[LIDAR_TEST] PASS={passed}", flush=True)
        return 0 if passed else 1
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
