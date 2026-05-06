from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)

parser = argparse.ArgumentParser(
    "Generate exact body-frame obstacle point clouds for the drone test env"
)
parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-Lidar-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=5)
parser.add_argument("--num_obstacles", type=int, default=8)
parser.add_argument("--active_obstacles", type=int, default=6)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--max_distance", type=float, default=50.0)
parser.add_argument("--lidar_max_distance", type=float, default=None)
parser.add_argument("--min_range", type=float, default=0.2)
parser.add_argument("--surface_step", type=float, default=0.25)
parser.add_argument("--max_plot_points", type=int, default=12000)
parser.add_argument("--robot_z", type=float, default=5.0)
parser.add_argument("--obstacle_height", type=float, default=10.0)
parser.add_argument("--obstacle_size_xy", type=float, default=1.0)
parser.add_argument("--state_dim", type=int, default=18)
parser.add_argument("--theta_min", type=float, default=75.0)
parser.add_argument("--theta_max", type=float, default=105.0)
parser.add_argument("--phi_min", type=float, default=0.0)
parser.add_argument("--phi_max", type=float, default=360.0)
parser.add_argument("--delta_theta", type=float, default=30.0)
parser.add_argument("--delta_phi", type=float, default=5.0)
parser.add_argument("--closeness_alpha", type=float, default=3.0)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import omniperception_isaacdrone.tasks.test6_registry as _test_registry  # noqa: F401
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_tasks.utils import parse_env_cfg

from omniperception_isaacdrone.envs.test_env import WallSpawner, setup_global_obstacles


OBSTACLE_HIDE_POS = (1200.0, 1200.0, -1200.0)


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


def make_output_dir(root_arg: str) -> Path:
    if str(root_arg).strip():
        root = Path(root_arg).expanduser()
    else:
        stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        project_root = Path(__file__).resolve().parents[1]
        root = project_root / "logs" / f"test3_exact_obstacle_pointcloud_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_max_obstacles(env) -> int:
    obstacles = env.scene["obstacles"]
    shape = obstacles.data.default_root_state.shape
    if len(shape) == 2:
        return int(shape[0])
    if len(shape) == 3:
        return int(shape[1])
    raise RuntimeError(f"Unexpected obstacles default_root_state shape: {shape}")


def deterministic_obstacle_positions(
    count: int,
    device: torch.device,
    z_height: float,
) -> torch.Tensor:
    base_xy = [
        (8.0, 0.0),
        (0.0, 8.0),
        (-8.0, 0.0),
        (0.0, -8.0),
        (7.0, 7.0),
        (-7.0, 7.0),
        (-7.0, -7.0),
        (7.0, -7.0),
        (13.0, 2.0),
        (-12.0, 4.0),
        (3.0, -13.0),
        (-4.0, -12.0),
    ]
    positions = torch.zeros((count, 3), device=device, dtype=torch.float32)
    for i in range(count):
        x, y = base_xy[i % len(base_xy)]
        ring = i // len(base_xy)
        positions[i] = torch.tensor(
            [x + 4.0 * ring, y - 3.0 * ring, z_height * 0.5],
            device=device,
            dtype=torch.float32,
        )
    return positions


def configure_obstacles(env, active_count: int, obstacle_height: float) -> int:
    obstacles = env.scene["obstacles"]
    max_obstacles = get_max_obstacles(env)
    active_count = min(max(int(active_count), 0), max_obstacles)

    positions = torch.zeros((max_obstacles, 3), device=env.device, dtype=torch.float32)
    positions[:, 0] = OBSTACLE_HIDE_POS[0]
    positions[:, 1] = OBSTACLE_HIDE_POS[1]
    positions[:, 2] = OBSTACLE_HIDE_POS[2]
    if active_count > 0:
        positions[:active_count] = deterministic_obstacle_positions(
            active_count, env.device, obstacle_height
        )

    orientations = torch.zeros((max_obstacles, 4), device=env.device, dtype=torch.float32)
    orientations[:, 0] = 1.0
    pose = torch.cat([positions, orientations], dim=-1)
    velocity = torch.zeros((max_obstacles, 6), device=env.device, dtype=torch.float32)
    env_ids = torch.arange(max_obstacles, device=env.device, dtype=torch.long)
    obstacles.write_root_pose_to_sim(pose, env_ids=env_ids)
    obstacles.write_root_velocity_to_sim(velocity, env_ids=env_ids)
    return active_count


def configure_robot_poses(env, robot_z: float) -> list[dict[str, float | str]]:
    cases = [
        ("identity", 0.0, 0.0, 0.0),
        ("yaw_90", 0.0, 0.0, 90.0),
        ("pitch_20", 0.0, 20.0, 0.0),
        ("roll_20", 20.0, 0.0, 0.0),
        ("rpy_mix", 15.0, -12.0, 60.0),
    ]
    robot = env.scene["robot"]
    num_envs = int(env.num_envs)
    pose = torch.zeros((num_envs, 7), device=env.device, dtype=torch.float32)
    velocity = torch.zeros((num_envs, 6), device=env.device, dtype=torch.float32)
    pose[:, 0:3] = torch.tensor([0.0, 0.0, robot_z], device=env.device, dtype=torch.float32)

    selected = []
    for env_id in range(num_envs):
        name, roll_deg, pitch_deg, yaw_deg = cases[env_id % len(cases)]
        quat = euler_xyz_to_quat_wxyz(
            math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
        ).to(env.device)
        pose[env_id, 3:7] = quat
        selected.append(
            {
                "env_id": env_id,
                "case": name,
                "roll_deg": roll_deg,
                "pitch_deg": pitch_deg,
                "yaw_deg": yaw_deg,
            }
        )

    env_ids = torch.arange(num_envs, device=env.device, dtype=torch.long)
    robot.write_root_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(velocity, env_ids=env_ids)
    return selected


def make_cuboid_surface_points(
    size_xy: float,
    height: float,
    step: float,
    device: torch.device,
) -> torch.Tensor:
    half_x = 0.5 * float(size_xy)
    half_y = 0.5 * float(size_xy)
    half_z = 0.5 * float(height)
    step = max(float(step), 0.02)

    xs = torch.arange(-half_x, half_x + 0.5 * step, step, device=device, dtype=torch.float32)
    ys = torch.arange(-half_y, half_y + 0.5 * step, step, device=device, dtype=torch.float32)
    zs = torch.arange(-half_z, half_z + 0.5 * step, step, device=device, dtype=torch.float32)

    faces = []
    yy, zz = torch.meshgrid(ys, zs, indexing="ij")
    faces.append(torch.stack([torch.full_like(yy, half_x), yy, zz], dim=-1).reshape(-1, 3))
    faces.append(torch.stack([torch.full_like(yy, -half_x), yy, zz], dim=-1).reshape(-1, 3))

    xx, zz = torch.meshgrid(xs, zs, indexing="ij")
    faces.append(torch.stack([xx, torch.full_like(xx, half_y), zz], dim=-1).reshape(-1, 3))
    faces.append(torch.stack([xx, torch.full_like(xx, -half_y), zz], dim=-1).reshape(-1, 3))

    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    faces.append(torch.stack([xx, yy, torch.full_like(xx, half_z)], dim=-1).reshape(-1, 3))
    faces.append(torch.stack([xx, yy, torch.full_like(xx, -half_z)], dim=-1).reshape(-1, 3))
    return torch.cat(faces, dim=0)


def generate_exact_obstacle_pointcloud_body(
    env,
    active_count: int,
    local_surface_points: torch.Tensor,
    min_range: float,
    max_distance: float,
) -> list[torch.Tensor]:
    robot = env.scene["robot"]
    obstacles = env.scene["obstacles"]

    robot_pos_w = robot.data.root_pos_w.to(torch.float32)
    robot_quat_w = robot.data.root_quat_w.to(torch.float32)
    obstacle_pos_w = obstacles.data.root_pos_w[:active_count].to(torch.float32)
    obstacle_quat_w = obstacles.data.root_quat_w[:active_count].to(torch.float32)

    if active_count <= 0:
        return [torch.empty((0, 3), device=env.device, dtype=torch.float32) for _ in range(env.num_envs)]

    surface = local_surface_points.unsqueeze(0).expand(active_count, -1, -1)
    obstacle_quat = obstacle_quat_w.unsqueeze(1).expand(-1, surface.shape[1], -1)
    points_w = quat_apply(obstacle_quat.reshape(-1, 4), surface.reshape(-1, 3)).view(active_count, -1, 3)
    points_w = points_w + obstacle_pos_w.unsqueeze(1)
    points_w = points_w.reshape(-1, 3)

    pointclouds = []
    for env_id in range(int(env.num_envs)):
        rel_w = points_w - robot_pos_w[env_id].unsqueeze(0)
        quat = robot_quat_w[env_id].unsqueeze(0).expand(rel_w.shape[0], -1)
        rel_b = quat_apply_inverse(quat, rel_w)
        ranges = torch.linalg.norm(rel_b, dim=-1)
        valid = (ranges >= float(min_range)) & (ranges <= float(max_distance))
        pointclouds.append(rel_b[valid].contiguous())
    return pointclouds


def downsample_for_plot(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
    return points[idx]


def map_pointcloud_to_lidar_grid(
    points_body: torch.Tensor,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    delta_theta: float,
    delta_phi: float,
    min_range: float,
    lidar_max_distance: float,
    closeness_alpha: float,
) -> dict[str, torch.Tensor | int]:
    """Map body-frame points to a polar LiDAR grid.

    theta is the polar angle measured from body +Z.
    phi is the azimuth angle around body Z, measured from body +X toward body +Y.
    """
    theta_span = max(float(theta_max) - float(theta_min), 1.0e-6)
    phi_span = max(float(phi_max) - float(phi_min), 1.0e-6)
    n_theta = max(int(theta_span / float(delta_theta)), 1)
    n_phi = max(int(phi_span / float(delta_phi)), 1)
    device = points_body.device

    distance_grid = torch.full(
        (n_theta, n_phi),
        float(lidar_max_distance),
        device=device,
        dtype=torch.float32,
    )
    valid_grid = torch.zeros((n_theta, n_phi), device=device, dtype=torch.bool)
    mapped_points = torch.empty((n_theta, n_phi, 3), device=device, dtype=torch.float32)

    theta_centers = torch.deg2rad(
        torch.linspace(
            float(theta_min) + 0.5 * float(delta_theta),
            float(theta_min) + (n_theta - 0.5) * float(delta_theta),
            n_theta,
            device=device,
            dtype=torch.float32,
        )
    )
    phi_centers = torch.deg2rad(
        torch.linspace(
            float(phi_min) + 0.5 * float(delta_phi),
            float(phi_min) + (n_phi - 0.5) * float(delta_phi),
            n_phi,
            device=device,
            dtype=torch.float32,
        )
    )
    tt, pp = torch.meshgrid(theta_centers, phi_centers, indexing="ij")
    center_dirs = torch.stack(
        [torch.sin(tt) * torch.cos(pp), torch.sin(tt) * torch.sin(pp), torch.cos(tt)],
        dim=-1,
    )
    mapped_points[:] = center_dirs * float(lidar_max_distance)

    if points_body.numel() > 0:
        x, y, z = points_body[:, 0], points_body[:, 1], points_body[:, 2]
        ranges = torch.linalg.norm(points_body, dim=-1)
        finite = torch.isfinite(points_body).all(dim=-1)
        valid = finite & (ranges >= float(min_range)) & (ranges <= float(lidar_max_distance))

        safe_ranges = torch.clamp(ranges, min=1.0e-12)
        theta = torch.rad2deg(torch.acos(torch.clamp(z / safe_ranges, -1.0, 1.0)))
        phi = torch.remainder(torch.rad2deg(torch.atan2(y, x)), 360.0)
        valid &= (theta >= float(theta_min)) & (theta < float(theta_max))
        valid &= (phi >= float(phi_min)) & (phi < float(phi_max))

        if valid.any():
            valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            valid_ranges = ranges[valid_indices]
            order = torch.argsort(valid_ranges)
            sorted_indices = valid_indices[order]
            t_idx = torch.clamp(
                torch.floor((theta[sorted_indices] - float(theta_min)) / float(delta_theta)).to(torch.long),
                0,
                n_theta - 1,
            )
            p_idx = torch.clamp(
                torch.floor((phi[sorted_indices] - float(phi_min)) / float(delta_phi)).to(torch.long),
                0,
                n_phi - 1,
            )
            for src_idx, ti, pi in zip(sorted_indices.tolist(), t_idx.tolist(), p_idx.tolist(), strict=False):
                if not bool(valid_grid[ti, pi]):
                    valid_grid[ti, pi] = True
                    distance_grid[ti, pi] = ranges[src_idx].to(torch.float32)
                    mapped_points[ti, pi] = points_body[src_idx].to(torch.float32)

    max_d = torch.tensor(float(lidar_max_distance), device=device, dtype=torch.float32)
    norm_dist = torch.clamp(distance_grid / max_d, 0.0, 1.0)
    alpha = max(float(closeness_alpha), 1.0e-6)
    exp_alpha = math.exp(-alpha)
    closeness_grid = (torch.exp(-alpha * norm_dist) - exp_alpha) / (1.0 - exp_alpha)
    closeness_grid = torch.clamp(closeness_grid, 0.0, 1.0)

    return {
        "distance_grid": distance_grid,
        "valid_grid": valid_grid,
        "closeness_grid": closeness_grid,
        "mapped_points": mapped_points.reshape(-1, 3),
        "occupied_mapped_points": mapped_points[valid_grid],
        "n_theta": n_theta,
        "n_phi": n_phi,
    }


def set_axes_equal(ax, points: np.ndarray, max_distance: float) -> None:
    if points.size == 0:
        radius = max_distance
        center = np.zeros(3)
    else:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = max(float((maxs - mins).max()) * 0.55, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_body_axes(ax) -> None:
    ax.scatter([0.0], [0.0], [0.0], c="red", s=35, marker="x", label="drone body origin")
    ax.quiver(0, 0, 0, 2.0, 0, 0, color="red", linewidth=1.2)
    ax.quiver(0, 0, 0, 0, 2.0, 0, color="green", linewidth=1.2)
    ax.quiver(0, 0, 0, 0, 0, 2.0, color="blue", linewidth=1.2)


def save_raw_pointcloud_plot(
    output_dir: Path,
    env_id: int,
    case_info: dict[str, float | str],
    points_body: torch.Tensor,
    max_points: int,
    max_distance: float,
) -> dict[str, float | int | str]:
    points_np = points_body.detach().cpu().numpy()
    plot_np = downsample_for_plot(points_np, max_points)

    fig = plt.figure(figsize=(7.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    if plot_np.size > 0:
        ranges = np.linalg.norm(plot_np, axis=1)
        sc = ax.scatter(
            plot_np[:, 0],
            plot_np[:, 1],
            plot_np[:, 2],
            c=ranges,
            cmap="viridis_r",
            s=2,
            alpha=0.85,
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.08, label="range in body frame (m)")
    plot_body_axes(ax)
    ax.set_xlabel("body x forward (m)")
    ax.set_ylabel("body y left (m)")
    ax.set_zlabel("body z up (m)")
    ax.set_title(
        f"env {env_id:02d} {case_info['case']} "
        f"r={case_info['roll_deg']} p={case_info['pitch_deg']} y={case_info['yaw_deg']}"
    )
    set_axes_equal(ax, plot_np, max_distance)
    ax.view_init(elev=22.0, azim=-58.0)
    ax.legend(loc="upper right")
    fig.tight_layout()

    png_path = output_dir / f"env_{env_id:02d}_{case_info['case']}_raw_pointcloud.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    if points_np.size == 0:
        min_range = max_range = mean_range = 0.0
    else:
        ranges = np.linalg.norm(points_np, axis=1)
        min_range = float(ranges.min())
        max_range = float(ranges.max())
        mean_range = float(ranges.mean())

    return {
        "env_id": int(env_id),
        "case": str(case_info["case"]),
        "roll_deg": float(case_info["roll_deg"]),
        "pitch_deg": float(case_info["pitch_deg"]),
        "yaw_deg": float(case_info["yaw_deg"]),
        "num_points": int(points_np.shape[0]),
        "min_range": min_range,
        "max_range": max_range,
        "mean_range": mean_range,
        "png": str(png_path),
    }


def save_mapped_pointcloud_plot(
    output_dir: Path,
    env_id: int,
    case_info: dict[str, float | str],
    mapped_points: torch.Tensor,
    valid_grid: torch.Tensor,
    max_distance: float,
) -> str:
    points_np = mapped_points.detach().cpu().numpy()
    valid_np = valid_grid.reshape(-1).detach().cpu().numpy().astype(bool)
    occupied = points_np[valid_np]
    empty = points_np[~valid_np]

    fig = plt.figure(figsize=(7.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    if empty.size > 0:
        ax.scatter(
            empty[:, 0],
            empty[:, 1],
            empty[:, 2],
            c="lightgray",
            s=3,
            alpha=0.16,
            linewidths=0,
            label="empty bins at max range",
        )
    if occupied.size > 0:
        ranges = np.linalg.norm(occupied, axis=1)
        sc = ax.scatter(
            occupied[:, 0],
            occupied[:, 1],
            occupied[:, 2],
            c=ranges,
            cmap="viridis_r",
            s=18,
            alpha=0.95,
            linewidths=0,
            label="nearest point per occupied bin",
        )
        fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.08, label="range in body frame (m)")
    plot_body_axes(ax)
    ax.set_xlabel("body x forward (m)")
    ax.set_ylabel("body y left (m)")
    ax.set_zlabel("body z up (m)")
    ax.set_title(
        f"mapped grid env {env_id:02d} {case_info['case']} "
        f"occupied={int(valid_np.sum())}/{valid_np.size}"
    )
    set_axes_equal(ax, points_np, max_distance)
    ax.view_init(elev=22.0, azim=-58.0)
    ax.legend(loc="upper right")
    fig.tight_layout()

    png_path = output_dir / f"env_{env_id:02d}_{case_info['case']}_mapped_pointcloud.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return str(png_path)


def save_grid_heatmap(
    output_dir: Path,
    env_id: int,
    case_info: dict[str, float | str],
    distance_grid: torch.Tensor,
    valid_grid: torch.Tensor,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    lidar_max_distance: float,
) -> str:
    distances = distance_grid.detach().cpu().numpy()
    valid = valid_grid.detach().cpu().numpy()
    masked = np.ma.array(distances, mask=~valid)

    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    image = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[float(phi_min), float(phi_max), float(theta_min), float(theta_max)],
        vmin=0.0,
        vmax=float(lidar_max_distance),
        cmap="viridis_r",
    )
    ax.set_xlabel("azimuth phi around body Z, +X toward +Y (deg)")
    ax.set_ylabel("polar theta from body +Z (deg)")
    ax.set_title(
        f"nearest range grid env {env_id:02d} {case_info['case']} "
        f"occupied={int(valid.sum())}/{valid.size}"
    )
    fig.colorbar(image, ax=ax, label="nearest range (m)")
    fig.tight_layout()

    png_path = output_dir / f"env_{env_id:02d}_{case_info['case']}_distance_grid.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return str(png_path)


def save_env_outputs(
    output_dir: Path,
    env_id: int,
    case_info: dict[str, float | str],
    points_body: torch.Tensor,
    grid: dict[str, torch.Tensor | int],
    max_points: int,
    max_distance: float,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
) -> dict[str, float | int | str]:
    item = save_raw_pointcloud_plot(
        output_dir=output_dir,
        env_id=env_id,
        case_info=case_info,
        points_body=points_body,
        max_points=max_points,
        max_distance=max_distance,
    )
    mapped_png = save_mapped_pointcloud_plot(
        output_dir=output_dir,
        env_id=env_id,
        case_info=case_info,
        mapped_points=grid["mapped_points"],
        valid_grid=grid["valid_grid"],
        max_distance=max_distance,
    )
    grid_png = save_grid_heatmap(
        output_dir=output_dir,
        env_id=env_id,
        case_info=case_info,
        distance_grid=grid["distance_grid"],
        valid_grid=grid["valid_grid"],
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
        lidar_max_distance=max_distance,
    )

    raw_np = points_body.detach().cpu().numpy()
    distance_grid_np = grid["distance_grid"].detach().cpu().numpy()
    valid_grid_np = grid["valid_grid"].detach().cpu().numpy()
    closeness_grid_np = grid["closeness_grid"].detach().cpu().numpy()
    mapped_np = grid["mapped_points"].detach().cpu().numpy()
    occupied_np = grid["occupied_mapped_points"].detach().cpu().numpy()
    npz_path = output_dir / f"env_{env_id:02d}_{case_info['case']}.npz"
    np.savez_compressed(
        npz_path,
        pointcloud_body=raw_np,
        distance_grid=distance_grid_np,
        valid_grid=valid_grid_np,
        closeness_grid=closeness_grid_np,
        mapped_pointcloud_body=mapped_np,
        occupied_mapped_pointcloud_body=occupied_np,
    )

    item.update(
        {
            "npz": str(npz_path),
            "mapped_png": mapped_png,
            "grid_png": grid_png,
            "n_theta": int(grid["n_theta"]),
            "n_phi": int(grid["n_phi"]),
            "num_bins": int(valid_grid_np.size),
            "num_occupied_bins": int(valid_grid_np.sum()),
            "mapped_num_points": int(mapped_np.shape[0]),
            "occupied_mapped_num_points": int(occupied_np.shape[0]),
        }
    )
    return item


def main() -> int:
    output_dir = make_output_dir(args_cli.output_dir)
    print(f"[TEST3_EXACT_PC] output_dir={output_dir}", flush=True)
    lidar_max_distance = (
        float(args_cli.lidar_max_distance)
        if args_cli.lidar_max_distance is not None
        else float(args_cli.max_distance)
    )

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=int(args_cli.num_envs),
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.normalization.state_dim = int(args_cli.state_dim)

    WallSpawner(
        x_bounds=(-80.0, 80.0),
        y_bounds=(-80.0, 80.0),
        z_bounds=(0.0, 10.0),
        wall_thickness=0.5,
    ).spawn_walls()
    setup_global_obstacles(int(args_cli.num_obstacles))

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    try:
        env.reset(seed=int(args_cli.seed))
        active_count = configure_obstacles(
            env,
            active_count=int(args_cli.active_obstacles),
            obstacle_height=float(args_cli.obstacle_height),
        )
        case_infos = configure_robot_poses(env, robot_z=float(args_cli.robot_z))
        env.sim.step()
        env.scene.update(dt=env.physics_dt)

        surface_points = make_cuboid_surface_points(
            size_xy=float(args_cli.obstacle_size_xy),
            height=float(args_cli.obstacle_height),
            step=float(args_cli.surface_step),
            device=env.device,
        )
        pointclouds = generate_exact_obstacle_pointcloud_body(
            env=env,
            active_count=active_count,
            local_surface_points=surface_points,
            min_range=float(args_cli.min_range),
            max_distance=lidar_max_distance,
        )

        summary = {
            "mode": "exact_obstacle_surface_points_in_drone_body_frame",
            "num_envs": int(env.num_envs),
            "active_obstacles": int(active_count),
            "surface_points_per_obstacle": int(surface_points.shape[0]),
            "min_range": float(args_cli.min_range),
            "max_distance": lidar_max_distance,
            "grid": {
                "theta_definition": "polar angle from body +Z axis, degrees",
                "phi_definition": "azimuth around body Z, measured from body +X toward body +Y, degrees",
                "theta_min": float(args_cli.theta_min),
                "theta_max": float(args_cli.theta_max),
                "phi_min": float(args_cli.phi_min),
                "phi_max": float(args_cli.phi_max),
                "delta_theta": float(args_cli.delta_theta),
                "delta_phi": float(args_cli.delta_phi),
                "lidar_max_distance": lidar_max_distance,
                "closeness_alpha": float(args_cli.closeness_alpha),
            },
            "surface_step": float(args_cli.surface_step),
            "cases": [],
        }
        for env_id, points in enumerate(pointclouds):
            grid = map_pointcloud_to_lidar_grid(
                points_body=points,
                theta_min=float(args_cli.theta_min),
                theta_max=float(args_cli.theta_max),
                phi_min=float(args_cli.phi_min),
                phi_max=float(args_cli.phi_max),
                delta_theta=float(args_cli.delta_theta),
                delta_phi=float(args_cli.delta_phi),
                min_range=float(args_cli.min_range),
                lidar_max_distance=lidar_max_distance,
                closeness_alpha=float(args_cli.closeness_alpha),
            )
            item = save_env_outputs(
                output_dir=output_dir,
                env_id=env_id,
                case_info=case_infos[env_id],
                points_body=points,
                grid=grid,
                max_points=int(args_cli.max_plot_points),
                max_distance=lidar_max_distance,
                theta_min=float(args_cli.theta_min),
                theta_max=float(args_cli.theta_max),
                phi_min=float(args_cli.phi_min),
                phi_max=float(args_cli.phi_max),
            )
            summary["cases"].append(item)
            print(
                f"[TEST3_EXACT_PC] env={env_id} case={item['case']} "
                f"points={item['num_points']} bins={item['num_occupied_bins']}/{item['num_bins']} "
                f"range=[{item['min_range']:.3f}, {item['max_range']:.3f}] "
                f"raw_png={item['png']} mapped_png={item['mapped_png']}",
                flush=True,
            )

        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[TEST3_EXACT_PC] done summary={summary_path}", flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
