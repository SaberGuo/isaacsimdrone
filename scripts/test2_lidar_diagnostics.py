from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)


parser = argparse.ArgumentParser("LiDAR point-cloud source diagnostics for the test6 drone env")
parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-ScanLidar-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--num_obstacles", type=int, default=16)
parser.add_argument("--active_obstacles", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument(
    "--case",
    type=str,
    default="all_training_sources",
    choices=[
        "frame_validation",
        "all_training_sources",
        "ground_walls_no_obstacles",
        "ground_only",
        "walls_all_only",
        "side_walls_only",
        "z_walls_only",
        "top_wall_only",
        "bottom_wall_only",
        "obstacle_only",
        "obstacle_plus_ground",
    ],
    help="Run one source-isolation case per process. Mesh targets are fixed before RayCaster init.",
)
parser.add_argument("--max_points", type=int, default=5000)
parser.add_argument("--max_env_plots", type=int, default=0, help="0 means plot every env")
parser.add_argument("--settle_steps", type=int, default=4)
parser.add_argument("--state_dim", type=int, default=22)
parser.add_argument("--theta_min", type=float, default=75.0)
parser.add_argument("--theta_max", type=float, default=105.0)
parser.add_argument("--phi_min", type=float, default=0.0)
parser.add_argument("--phi_max", type=float, default=360.0)
parser.add_argument("--delta_theta", type=float, default=30.0)
parser.add_argument("--delta_phi", type=float, default=5.0)
parser.add_argument("--force_simple_grid", action="store_true", default=False)
parser.add_argument("--save_stride_compare", action="store_true", default=True)
parser.add_argument("--no_save_stride_compare", dest="save_stride_compare", action="store_false")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import isaacsim.core.utils.prims as prim_utils
import numpy as np
import torch
import omniperception_isaacdrone.tasks.test_registry as _test_registry  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils import parse_env_cfg
from pxr import Gf, UsdGeom

import isaaclab.envs.mdp as mdp

from omniperception_isaacdrone.envs.test_env import WallSpawner, setup_global_obstacles


WALL_DEFAULT_POSES: dict[str, tuple[float, float, float]] = {
    "Wall_XMin": (-80.0, 0.0, 5.0),
    "Wall_XMax": (80.0, 0.0, 5.0),
    "Wall_YMin": (0.0, -80.0, 5.0),
    "Wall_YMax": (0.0, 80.0, 5.0),
    "Wall_ZMin": (0.0, 0.0, 0.0),
    "Wall_ZMax": (0.0, 0.0, 10.0),
}
SIDE_WALLS = {"Wall_XMin", "Wall_XMax", "Wall_YMin", "Wall_YMax"}
Z_WALLS = {"Wall_ZMin", "Wall_ZMax"}
HIDDEN_BASE = (1200.0, 1200.0, -1200.0)

WALL_ROOT = "/World/Wall"
GROUND_PATH = "/World/ground"
OBSTACLE_ROOT = "/World/Obstacles"
SIDE_WALL_PATHS = [f"{WALL_ROOT}/{name}" for name in ("Wall_XMin", "Wall_XMax", "Wall_YMin", "Wall_YMax")]
Z_WALL_PATHS = [f"{WALL_ROOT}/{name}" for name in ("Wall_ZMin", "Wall_ZMax")]

CASE_CONFIGS: dict[str, dict[str, Any]] = {
    "frame_validation": {
        "mesh_prim_paths": [OBSTACLE_ROOT],
        "active_obstacles": True,
        "description": "Single obstacle only, for body-frame validation.",
    },
    "all_training_sources": {
        "mesh_prim_paths": [GROUND_PATH, OBSTACLE_ROOT, WALL_ROOT],
        "active_obstacles": True,
        "description": "Training-equivalent LiDAR targets.",
    },
    "ground_walls_no_obstacles": {
        "mesh_prim_paths": [GROUND_PATH, WALL_ROOT],
        "active_obstacles": False,
        "description": "Ground and all workspace walls, no active obstacles.",
    },
    "ground_only": {
        "mesh_prim_paths": [GROUND_PATH],
        "active_obstacles": False,
        "description": "Ground plane only.",
    },
    "walls_all_only": {
        "mesh_prim_paths": [WALL_ROOT],
        "active_obstacles": False,
        "description": "All workspace walls only.",
    },
    "side_walls_only": {
        "mesh_prim_paths": SIDE_WALL_PATHS,
        "active_obstacles": False,
        "description": "Only X/Y side walls.",
    },
    "z_walls_only": {
        "mesh_prim_paths": Z_WALL_PATHS,
        "active_obstacles": False,
        "description": "Only floor/ceiling wall cuboids.",
    },
    "top_wall_only": {
        "mesh_prim_paths": [f"{WALL_ROOT}/Wall_ZMax"],
        "active_obstacles": False,
        "description": "Only upper workspace wall.",
    },
    "bottom_wall_only": {
        "mesh_prim_paths": [f"{WALL_ROOT}/Wall_ZMin"],
        "active_obstacles": False,
        "description": "Only lower workspace wall.",
    },
    "obstacle_only": {
        "mesh_prim_paths": [OBSTACLE_ROOT],
        "active_obstacles": True,
        "description": "Active obstacle cuboids only.",
    },
    "obstacle_plus_ground": {
        "mesh_prim_paths": [OBSTACLE_ROOT, GROUND_PATH],
        "active_obstacles": True,
        "description": "Active obstacles plus ground.",
    },
}


def _import_plotting():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def make_output_dir(root_arg: str) -> Path:
    if str(root_arg).strip():
        root = Path(root_arg).expanduser()
    else:
        stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        project_root = Path(__file__).resolve().parents[1]
        case_name = str(getattr(args, "case", "unknown"))
        root = project_root / "logs" / f"test2_lidar_diag_{case_name}_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def set_prim_translation(stage, prim_path: str, translation: tuple[float, float, float]) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    xform = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(*translation))
    else:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    return True


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


def hidden_pose(index: int) -> tuple[float, float, float]:
    return (HIDDEN_BASE[0] + 30.0 * float(index), HIDDEN_BASE[1], HIDDEN_BASE[2])


def configure_walls(stage, enabled_wall_names: set[str]) -> None:
    for idx, (name, pose) in enumerate(WALL_DEFAULT_POSES.items()):
        target = pose if name in enabled_wall_names else hidden_pose(idx)
        set_prim_translation(stage, f"/World/Wall/{name}", target)


def configure_ground(stage, enabled: bool) -> None:
    target = (0.0, 0.0, 0.0) if enabled else (0.0, 0.0, -1200.0)
    set_prim_translation(stage, "/World/ground", target)


def get_max_obstacles(env) -> int:
    obstacles = env.scene["obstacles"]
    shape = obstacles.data.default_root_state.shape
    if len(shape) == 2:
        return int(shape[0])
    if len(shape) == 3:
        return int(shape[1])
    raise RuntimeError(f"Unexpected obstacles default_root_state shape: {shape}")


def configure_obstacles(env, active: bool, active_count: int, radius: float = 10.0) -> int:
    obstacles = env.scene["obstacles"]
    max_obstacles = get_max_obstacles(env)
    active_count = min(max(int(active_count), 0), max_obstacles) if active else 0

    positions = torch.zeros((max_obstacles, 3), device=env.device, dtype=torch.float32)
    positions[:, 0] = HIDDEN_BASE[0]
    positions[:, 1] = HIDDEN_BASE[1]
    positions[:, 2] = HIDDEN_BASE[2]

    if active_count > 0:
        angles = torch.linspace(0.0, 2.0 * math.pi, active_count + 1, device=env.device)[:-1]
        rings = 1.0 + 0.25 * (torch.arange(active_count, device=env.device, dtype=torch.float32) % 3.0)
        positions[:active_count, 0] = float(radius) * rings * torch.cos(angles)
        positions[:active_count, 1] = float(radius) * rings * torch.sin(angles)
        positions[:active_count, 2] = 5.0

    quat = torch.zeros((max_obstacles, 4), device=env.device, dtype=torch.float32)
    quat[:, 0] = 1.0
    velocities = torch.zeros((max_obstacles, 6), device=env.device, dtype=torch.float32)
    root_pose = torch.cat([positions, quat], dim=-1)
    env_ids = torch.arange(max_obstacles, device=env.device, dtype=torch.long)
    obstacles.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    obstacles.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    return active_count


def configure_robot_poses(env) -> dict[str, Any]:
    num_envs = int(env.num_envs)
    robot = env.scene["robot"]
    pose = torch.zeros((num_envs, 7), device=env.device, dtype=torch.float32)
    vel = torch.zeros((num_envs, 6), device=env.device, dtype=torch.float32)

    pose[:, 0:3] = torch.tensor([0.0, 0.0, 5.0], device=env.device, dtype=torch.float32)
    angle_cases = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (-10.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
        (0.0, -10.0, 0.0),
        (8.0, 8.0, 45.0),
        (-8.0, 8.0, 90.0),
        (8.0, -8.0, 135.0),
        (0.0, 0.0, 180.0),
        (15.0, -12.0, 60.0),
        (-15.0, 12.0, -60.0),
        (5.0, 15.0, -120.0),
    ]

    pose_meta = []
    for env_id in range(num_envs):
        roll_deg, pitch_deg, yaw_deg = angle_cases[env_id % len(angle_cases)]
        quat = euler_xyz_to_quat_wxyz(
            math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
        ).to(env.device)
        pose[env_id, 3:7] = quat
        pose_meta.append(
            {
                "env_id": env_id,
                "position": [0.0, 0.0, 5.0],
                "rpy_deg": [roll_deg, pitch_deg, yaw_deg],
                "quat_wxyz": quat.detach().cpu().tolist(),
            }
        )

    robot.write_root_pose_to_sim(pose, env_ids=torch.arange(num_envs, device=env.device))
    robot.write_root_velocity_to_sim(vel, env_ids=torch.arange(num_envs, device=env.device))
    return {"robot_poses": pose_meta}


def clear_lidar_grid_cache(env) -> None:
    if hasattr(env, "_lidar_grid_cache"):
        try:
            delattr(env, "_lidar_grid_cache")
        except Exception:
            setattr(env, "_lidar_grid_cache", None)


def update_lidar_now(env, settle_steps: int) -> None:
    lidar = env.scene["lidar"]
    steps = max(int(settle_steps), 1)
    for _ in range(steps):
        env.sim.step()
        env.scene.update(dt=env.physics_dt)
    clear_lidar_grid_cache(env)
    lidar.update(env.physics_dt, force_recompute=True)


def closeness_from_distance(dist: torch.Tensor, max_d: float, alpha: float = 3.0) -> torch.Tensor:
    exp_alpha = math.exp(-alpha)
    norm_dist = torch.clamp(dist / float(max_d), 0.0, 1.0)
    return (torch.exp(-alpha * norm_dist) - exp_alpha) / (1.0 - exp_alpha)


def distance_from_closeness(close: torch.Tensor, max_d: float, alpha: float = 3.0) -> torch.Tensor:
    exp_alpha = math.exp(-alpha)
    val = torch.clamp(close * (1.0 - exp_alpha) + exp_alpha, min=exp_alpha, max=1.0)
    return -(float(max_d) / alpha) * torch.log(val)


def compute_lidar_grid_from_pointcloud(
    pointcloud: torch.Tensor,
    *,
    min_range: float,
    max_distance: float,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    delta_theta: float,
    delta_phi: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if pointcloud.dim() == 2:
        pointcloud = pointcloud.unsqueeze(0)
    env_count = pointcloud.shape[0]
    theta_bins = max(int((theta_max - theta_min) / delta_theta), 1)
    phi_bins = max(int((phi_max - phi_min) / delta_phi), 1)
    num_bins = theta_bins * phi_bins

    x, y, z = pointcloud[..., 0], pointcloud[..., 1], pointcloud[..., 2]
    finite = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z)
    r = torch.sqrt(x * x + y * y + z * z + 1.0e-12)
    valid = finite & (r > (float(min_range) + 1.0e-3)) & (r <= (float(max_distance) + 1.0e-3))

    theta = torch.rad2deg(torch.acos(torch.clamp(z / r, -1.0, 1.0)))
    phi = torch.remainder(torch.rad2deg(torch.atan2(y, x)), 360.0)
    in_theta = (theta >= theta_min) & (theta < theta_max)
    in_phi = (phi >= phi_min) & (phi < phi_max)
    selected = valid & in_theta & in_phi

    flat_min = torch.full((env_count * num_bins,), float("inf"), device=pointcloud.device, dtype=torch.float32)
    if selected.any():
        t_idx = torch.clamp(torch.floor((theta - theta_min) / delta_theta).long(), 0, theta_bins - 1)
        p_idx = torch.clamp(torch.floor((phi - phi_min) / delta_phi).long(), 0, phi_bins - 1)
        lin_idx = t_idx * phi_bins + p_idx
        env_offset = torch.arange(env_count, device=pointcloud.device, dtype=torch.long).unsqueeze(1)
        flat_idx = env_offset * num_bins + lin_idx
        flat_min.scatter_reduce_(0, flat_idx[selected], r[selected].to(torch.float32), reduce="amin", include_self=True)

    min_dist = flat_min.view(env_count, num_bins)
    max_d_t = torch.tensor(float(max_distance), device=pointcloud.device, dtype=torch.float32)
    min_dist = torch.where(torch.isfinite(min_dist), min_dist, max_d_t)
    min_dist = torch.clamp(min_dist, 0.0, max_d_t)
    closeness = torch.clamp(closeness_from_distance(min_dist, float(max_distance)), 0.0, 1.0).to(torch.float32)
    return closeness, min_dist, selected, r


def downsample_points(points: torch.Tensor, max_points: int, mode: str, seed: int) -> np.ndarray:
    pts = points.detach()
    finite = torch.isfinite(pts).all(dim=-1)
    pts = pts[finite]
    if pts.numel() == 0:
        return np.empty((0, 3), dtype=np.float32)
    max_points = int(max_points)
    if max_points > 0 and pts.shape[0] > max_points:
        if mode == "random":
            gen = torch.Generator(device=pts.device)
            gen.manual_seed(int(seed))
            idx = torch.randperm(pts.shape[0], device=pts.device, generator=gen)[:max_points]
            pts = pts[idx]
        else:
            step = max(int(pts.shape[0] // max_points), 1)
            pts = pts[::step][:max_points]
    return pts.float().cpu().numpy()


def save_pointcloud_3d(points_np: np.ndarray, path: Path, title: str) -> None:
    plt = _import_plotting()
    fig = plt.figure(figsize=(6.2, 6.2), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    if points_np.shape[0] > 0:
        ranges = np.linalg.norm(points_np, axis=1)
        sc = ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=ranges, s=1.0, alpha=0.75, cmap="viridis")
        fig.colorbar(sc, ax=ax, fraction=0.032, pad=0.08, label="range")
        mn, mx = points_np.min(axis=0), points_np.max(axis=0)
        center = (mn + mx) * 0.5
        radius = max(float((mx - mn).max() * 0.5), 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    else:
        ax.text2D(0.25, 0.5, "No finite LiDAR points", transform=ax.transAxes)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
    ax.set_xlabel("X body")
    ax.set_ylabel("Y body")
    ax.set_zlabel("Z body")
    ax.set_title(title)
    ax.view_init(elev=25, azim=45)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_topdown(points_np: np.ndarray, path: Path, title: str) -> None:
    plt = _import_plotting()
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=150)
    if points_np.shape[0] > 0:
        ranges = np.linalg.norm(points_np, axis=1)
        sc = ax.scatter(points_np[:, 0], points_np[:, 1], c=ranges, s=1.0, alpha=0.75, cmap="viridis")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="range")
    ax.scatter([0.0], [0.0], c="red", s=18, label="sensor")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X body")
    ax.set_ylabel("Y body")
    ax.set_title(title)
    ax.grid(True, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_closeness_radar(closeness_flat: torch.Tensor, grid_shape: tuple[int, int], path: Path, title: str) -> None:
    plt = _import_plotting()
    close_np = torch.clamp(closeness_flat.detach().float(), 0.0, 1.0).cpu().numpy()
    theta_bins, phi_bins = grid_shape
    grid = close_np.reshape(theta_bins, phi_bins)
    angles = np.linspace(0.0, 2.0 * np.pi, phi_bins, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    fig = plt.figure(figsize=(5.6, 5.6), dpi=150)
    ax = fig.add_subplot(111, polar=True)
    for theta_idx in range(theta_bins):
        values = np.concatenate([grid[theta_idx], grid[theta_idx, :1]])
        ax.plot(angles_closed, values, linewidth=1.2, alpha=0.9)
        ax.fill(angles_closed, values, alpha=0.08)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_case_heatmap(closeness: torch.Tensor, path: Path, title: str) -> None:
    plt = _import_plotting()
    arr = closeness.detach().float().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10.0, max(3.0, 0.18 * arr.shape[0])), dpi=150)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xlabel("phi bin")
    ax.set_ylabel("env id")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="closeness")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_pattern_diagnostics(lidar, output_dir: Path, max_points: int) -> None:
    rays = lidar.ray_directions[0].detach().float().cpu().numpy()
    rays = rays[np.isfinite(rays).all(axis=1)]
    if rays.shape[0] == 0:
        return
    if max_points > 0 and rays.shape[0] > max_points:
        step = max(int(rays.shape[0] // max_points), 1)
        rays_plot = rays[::step][:max_points]
    else:
        rays_plot = rays

    horiz = np.rad2deg(np.arctan2(rays_plot[:, 1], rays_plot[:, 0]))
    elev = np.rad2deg(np.arcsin(np.clip(rays_plot[:, 2] / np.linalg.norm(rays_plot, axis=1), -1.0, 1.0)))
    plt = _import_plotting()

    fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.0), dpi=150)
    axs[0, 0].plot(horiz, linewidth=0.5)
    axs[0, 0].set_title("horizontal angle sequence")
    axs[0, 0].set_ylabel("deg")
    axs[0, 1].plot(elev, linewidth=0.5)
    axs[0, 1].set_title("vertical angle sequence")
    axs[0, 1].set_ylabel("deg")
    axs[1, 0].scatter(horiz, elev, s=1.0, alpha=0.6)
    axs[1, 0].set_xlabel("horizontal deg")
    axs[1, 0].set_ylabel("vertical deg")
    axs[1, 0].set_title("angle coverage")
    axs[1, 1].plot(rays_plot[:, 0], rays_plot[:, 1], linewidth=0.35, alpha=0.8)
    axs[1, 1].set_aspect("equal", adjustable="box")
    axs[1, 1].set_xlabel("dir x")
    axs[1, 1].set_ylabel("dir y")
    axs[1, 1].set_title("ray direction XY sequence")
    for ax in axs.ravel():
        ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "ray_pattern_diagnostics.png")
    plt.close(fig)

    stats = {
        "num_rays": int(rays.shape[0]),
        "horizontal_deg_min": float(np.rad2deg(np.arctan2(rays[:, 1], rays[:, 0])).min()),
        "horizontal_deg_max": float(np.rad2deg(np.arctan2(rays[:, 1], rays[:, 0])).max()),
        "vertical_deg_min": float(np.rad2deg(np.arcsin(np.clip(rays[:, 2] / np.linalg.norm(rays, axis=1), -1.0, 1.0))).min()),
        "vertical_deg_max": float(np.rad2deg(np.arcsin(np.clip(rays[:, 2] / np.linalg.norm(rays, axis=1), -1.0, 1.0))).max()),
    }
    (output_dir / "ray_pattern_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def get_lidar_ranges(lidar) -> tuple[float, float]:
    min_r = float(getattr(lidar.cfg, "min_range", 0.2))
    max_d = float(getattr(lidar.cfg, "max_distance", 10.0))
    return min_r, max_d


def compare_with_observation_manager(env, grid: torch.Tensor, num_bins: int) -> float | None:
    clear_lidar_grid_cache(env)
    try:
        obs = env.observation_manager.compute_group("policy", update_history=False)
    except Exception:
        return None
    if not isinstance(obs, torch.Tensor) or obs.shape[-1] < num_bins:
        return None
    obs_lidar = obs[:, -num_bins:]
    return float(torch.max(torch.abs(obs_lidar.detach().float() - grid.detach().float())).item())


def summarize_case(
    *,
    env,
    case_dir: Path,
    case_name: str,
    active_obstacles: int,
    max_points: int,
    max_env_plots: int,
    save_stride_compare: bool,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    delta_theta: float,
    delta_phi: float,
) -> list[dict[str, Any]]:
    lidar = env.scene["lidar"]
    env_ids = torch.arange(env.num_envs, device=env.device)
    pointcloud = lidar.get_pointcloud(env_ids)
    distances = lidar.get_distances(env_ids)
    if pointcloud.dim() == 2:
        pointcloud = pointcloud.unsqueeze(0)
    if distances.dim() == 1:
        distances = distances.unsqueeze(0)

    min_range, max_distance = get_lidar_ranges(lidar)
    grid, min_dist_grid, selected, ranges = compute_lidar_grid_from_pointcloud(
        pointcloud,
        min_range=min_range,
        max_distance=max_distance,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
        delta_theta=delta_theta,
        delta_phi=delta_phi,
    )
    theta_bins = max(int((theta_max - theta_min) / delta_theta), 1)
    phi_bins = max(int((phi_max - phi_min) / delta_phi), 1)
    num_bins = theta_bins * phi_bins
    obs_grid_max_abs_err = compare_with_observation_manager(env, grid, num_bins)

    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w.detach().float()
    root_quat = robot.data.root_quat_w.detach().float()
    try:
        projected_gravity = mdp.projected_gravity(env, asset_cfg=SceneEntityCfg("robot")).detach().float()
    except Exception:
        projected_gravity = torch.full((env.num_envs, 3), float("nan"), device=env.device)

    finite = torch.isfinite(pointcloud).all(dim=-1)
    finite_ranges = torch.where(finite, ranges, torch.full_like(ranges, float("nan")))
    pc_norm = torch.linalg.norm(pointcloud, dim=-1)
    finite_dist_cmp = finite & torch.isfinite(distances)
    dist_err = torch.abs(pc_norm - distances)

    case_dir.mkdir(parents=True, exist_ok=True)
    save_case_heatmap(grid, case_dir / "closeness_heatmap.png", f"{case_name} closeness heatmap")

    rows: list[dict[str, Any]] = []
    env_plot_limit = int(env.num_envs) if int(max_env_plots) <= 0 else min(int(max_env_plots), int(env.num_envs))

    for env_id in range(int(env.num_envs)):
        env_grid = grid[env_id]
        env_min_dist = min_dist_grid[env_id]
        finite_count = int(finite[env_id].sum().item())
        filtered_count = int(selected[env_id].sum().item())
        max_close, max_idx = torch.max(env_grid, dim=0)
        min_range_grid = float(distance_from_closeness(max_close, max_distance).item())
        phi_center = float(phi_min + (int(max_idx.item()) % phi_bins + 0.5) * delta_phi)

        valid_ranges = finite_ranges[env_id][torch.isfinite(finite_ranges[env_id])]
        if valid_ranges.numel() > 0:
            range_min = float(valid_ranges.min().item())
            range_mean = float(valid_ranges.mean().item())
            range_p50 = float(torch.quantile(valid_ranges, 0.5).item())
            range_max = float(valid_ranges.max().item())
        else:
            range_min = range_mean = range_p50 = range_max = float("nan")

        if finite_dist_cmp[env_id].any():
            dist_pc_max_abs_err = float(dist_err[env_id][finite_dist_cmp[env_id]].max().item())
        else:
            dist_pc_max_abs_err = float("nan")

        row = {
            "case": case_name,
            "env_id": env_id,
            "active_obstacles": active_obstacles,
            "root_pos_x": float(root_pos[env_id, 0].item()),
            "root_pos_y": float(root_pos[env_id, 1].item()),
            "root_pos_z": float(root_pos[env_id, 2].item()),
            "root_quat_w": float(root_quat[env_id, 0].item()),
            "root_quat_x": float(root_quat[env_id, 1].item()),
            "root_quat_y": float(root_quat[env_id, 2].item()),
            "root_quat_z": float(root_quat[env_id, 3].item()),
            "projected_gravity_x": float(projected_gravity[env_id, 0].item()),
            "projected_gravity_y": float(projected_gravity[env_id, 1].item()),
            "projected_gravity_z": float(projected_gravity[env_id, 2].item()),
            "finite_points": finite_count,
            "filtered_points": filtered_count,
            "range_min": range_min,
            "range_mean": range_mean,
            "range_p50": range_p50,
            "range_max": range_max,
            "closeness_max": float(max_close.item()),
            "closeness_mean": float(env_grid.mean().item()),
            "equiv_min_distance_from_max_closeness": min_range_grid,
            "max_closeness_phi_center_deg": phi_center,
            "dist_vs_pointcloud_norm_max_abs_err": dist_pc_max_abs_err,
            "obs_manager_grid_max_abs_err": obs_grid_max_abs_err,
        }
        rows.append(row)

        if env_id < env_plot_limit:
            prefix = f"env_{env_id:04d}"
            raw_random = downsample_points(pointcloud[env_id], max_points=max_points, mode="random", seed=1009 + env_id)
            filtered_points = pointcloud[env_id][selected[env_id]]
            filtered_random = downsample_points(filtered_points, max_points=max_points, mode="random", seed=2003 + env_id)
            save_pointcloud_3d(raw_random, case_dir / f"{prefix}_pointcloud3d_random.png", f"{case_name} raw env={env_id}")
            save_pointcloud_3d(
                filtered_random,
                case_dir / f"{prefix}_pointcloud3d_filtered_theta.png",
                f"{case_name} filtered theta env={env_id}",
            )
            save_topdown(raw_random, case_dir / f"{prefix}_topdown_random.png", f"{case_name} raw XY env={env_id}")
            save_closeness_radar(env_grid, (theta_bins, phi_bins), case_dir / f"{prefix}_closeness_radar.png", f"{case_name} env={env_id}")
            if save_stride_compare:
                raw_stride = downsample_points(pointcloud[env_id], max_points=max_points, mode="stride", seed=0)
                save_pointcloud_3d(raw_stride, case_dir / f"{prefix}_pointcloud3d_stride.png", f"{case_name} raw stride env={env_id}")

    with (case_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (case_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    max_close_all = max((row["closeness_max"] for row in rows), default=float("nan"))
    max_env = max(rows, key=lambda r: r["closeness_max"])["env_id"] if rows else -1
    print(
        f"[TEST2_LIDAR] case={case_name} active_obstacles={active_obstacles} "
        f"max_closeness={max_close_all:.4f} max_env={max_env} "
        f"obs_grid_err={obs_grid_max_abs_err}",
        flush=True,
    )
    return rows


def set_single_obstacle_pose(env, position_w: torch.Tensor) -> None:
    obstacles = env.scene["obstacles"]
    max_obstacles = get_max_obstacles(env)
    positions = torch.zeros((max_obstacles, 3), device=env.device, dtype=torch.float32)
    positions[:, 0] = HIDDEN_BASE[0]
    positions[:, 1] = HIDDEN_BASE[1]
    positions[:, 2] = HIDDEN_BASE[2]
    positions[0] = position_w
    quat = torch.zeros((max_obstacles, 4), device=env.device, dtype=torch.float32)
    quat[:, 0] = 1.0
    velocities = torch.zeros((max_obstacles, 6), device=env.device, dtype=torch.float32)
    root_pose = torch.cat([positions, quat], dim=-1)
    env_ids = torch.arange(max_obstacles, device=env.device, dtype=torch.long)
    obstacles.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    obstacles.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def run_frame_validation(env, output_dir: Path, settle_steps: int) -> list[dict[str, Any]]:
    obstacle_pos_w = torch.tensor([10.0, 0.0, 5.0], device=env.device, dtype=torch.float32)
    set_single_obstacle_pose(env, obstacle_pos_w)
    cases = [
        ("identity", 0.0, 0.0, 0.0),
        ("yaw_90", 0.0, 0.0, 90.0),
        ("pitch_20", 0.0, 20.0, 0.0),
        ("roll_20", 20.0, 0.0, 0.0),
        ("rpy_mix", 15.0, -12.0, 60.0),
    ]
    robot = env.scene["robot"]
    lidar = env.scene["lidar"]
    results: list[dict[str, Any]] = []
    for name, roll_deg, pitch_deg, yaw_deg in cases:
        pose = torch.zeros((env.num_envs, 7), device=env.device, dtype=torch.float32)
        vel = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
        pose[:, 0:3] = torch.tensor([0.0, 0.0, 5.0], device=env.device, dtype=torch.float32)
        quat = euler_xyz_to_quat_wxyz(
            math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
        ).to(env.device)
        pose[:, 3:7] = quat
        robot.write_root_pose_to_sim(pose, env_ids=torch.arange(env.num_envs, device=env.device))
        robot.write_root_velocity_to_sim(vel, env_ids=torch.arange(env.num_envs, device=env.device))
        update_lidar_now(env, settle_steps)
        pointcloud = lidar.get_pointcloud(torch.tensor([0], device=env.device))[0]
        valid = torch.isfinite(pointcloud).all(dim=-1)
        ranges = torch.linalg.norm(pointcloud, dim=-1)
        valid &= (ranges >= 8.0) & (ranges <= 12.5)
        if valid.any():
            centroid = pointcloud[valid].mean(dim=0)
            expected = quat_apply_inverse(pose[:1, 3:7], obstacle_pos_w.unsqueeze(0) - pose[:1, 0:3])[0]
            error = float(torch.linalg.norm(centroid - expected).item())
            num_points = int(valid.sum().item())
        else:
            centroid = torch.full((3,), float("nan"), device=env.device)
            expected = quat_apply_inverse(pose[:1, 3:7], obstacle_pos_w.unsqueeze(0) - pose[:1, 0:3])[0]
            error = float("inf")
            num_points = 0
        result = {
            "case": name,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "expected": expected.detach().cpu().tolist(),
            "centroid": centroid.detach().cpu().tolist(),
            "error": error,
            "num_points": num_points,
        }
        results.append(result)
        print(f"[TEST2_LIDAR] frame_case={name} error={error:.4f} num_points={num_points}", flush=True)

    with (output_dir / "frame_validation.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def build_env(case_cfg: dict[str, Any]):
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=int(args.num_envs),
        use_fabric=not args.disable_fabric,
    )
    env_cfg.scene.replicate_physics = True
    env_cfg.scene.filter_collisions = True
    try:
        setattr(env_cfg, "seed", int(args.seed))
    except Exception:
        pass
    try:
        env_cfg.scene.lidar.mesh_prim_paths = list(case_cfg["mesh_prim_paths"])
    except Exception as exc:
        raise RuntimeError(f"Failed to set LiDAR mesh_prim_paths for case={args.case}: {exc}") from exc
    if bool(args.force_simple_grid):
        try:
            env_cfg.scene.lidar.pattern_cfg.use_simple_grid = True
        except Exception as exc:
            print(f"[TEST2_LIDAR] could not force simple grid: {exc}", flush=True)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    WallSpawner(
        x_bounds=(-80.0, 80.0),
        y_bounds=(-80.0, 80.0),
        z_bounds=(0.0, 10.0),
        wall_thickness=0.5,
        color=(0.7, 0.7, 0.2),
        wall_colors={
            "Wall_XMin": (0.5, 1.0, 1.0),
            "Wall_XMax": (1.0, 1.0, 0.5),
            "Wall_YMin": (0.0, 1.0, 1.0),
            "Wall_YMax": (1.0, 1.0, 0.0),
            "Wall_ZMin": (1.0, 1.0, 1.0),
            "Wall_ZMax": (0.0, 0.0, 0.0),
        },
    ).spawn_walls()
    setup_global_obstacles(max(int(args.num_obstacles), 1))
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    return env


def main() -> int:
    output_dir = make_output_dir(args.output_dir)
    print(f"[TEST2_LIDAR] output_dir={output_dir}", flush=True)
    case_cfg = CASE_CONFIGS[str(args.case)]
    print(
        f"[TEST2_LIDAR] case={args.case} mesh_prim_paths={case_cfg['mesh_prim_paths']} "
        f"description={case_cfg['description']}",
        flush=True,
    )

    env = build_env(case_cfg)
    try:
        env.reset(seed=int(args.seed))
        setattr(env, "_goal_vis_enabled", False)
        setattr(env, "_collision_print_enabled", False)
        lidar = env.scene["lidar"]
        lidar_info = {
            "case": str(args.case),
            "case_description": str(case_cfg["description"]),
            "ray_alignment": str(getattr(lidar.cfg, "ray_alignment", "unknown")),
            "pointcloud_in_world_frame": bool(getattr(lidar.cfg, "pointcloud_in_world_frame", False)),
            "update_frequency": float(getattr(lidar.cfg, "update_frequency", 0.0)),
            "update_period": float(getattr(lidar.cfg, "update_period", 0.0)),
            "min_range": float(getattr(lidar.cfg, "min_range", 0.2)),
            "max_distance": float(getattr(lidar.cfg, "max_distance", 10.0)),
            "mesh_prim_paths": list(getattr(lidar.cfg, "mesh_prim_paths", [])),
            "pattern_cfg": {
                "class": lidar.cfg.pattern_cfg.__class__.__name__,
                "sensor_type": str(getattr(lidar.cfg.pattern_cfg, "sensor_type", "")),
                "use_simple_grid": bool(getattr(lidar.cfg.pattern_cfg, "use_simple_grid", False)),
                "samples": int(getattr(lidar.cfg.pattern_cfg, "samples", -1)),
                "downsample": int(getattr(lidar.cfg.pattern_cfg, "downsample", -1)),
            },
        }
        (output_dir / "lidar_config.json").write_text(json.dumps(lidar_info, indent=2), encoding="utf-8")
        save_pattern_diagnostics(lidar, output_dir, max_points=int(args.max_points))

        if str(args.case) == "frame_validation":
            frame_results = run_frame_validation(env, output_dir, int(args.settle_steps))
            frame_max_error = max((float(item["error"]) for item in frame_results), default=float("inf"))
            report = {
                "output_dir": str(output_dir),
                "case": str(args.case),
                "mesh_prim_paths": list(getattr(lidar.cfg, "mesh_prim_paths", [])),
                "frame_validation_max_error": frame_max_error,
                "passed_under_2p5m": bool(frame_max_error < 2.5),
            }
            (output_dir / "diagnostic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("[TEST2_LIDAR] frame report:", json.dumps(report, indent=2), flush=True)
            print(f"[TEST2_LIDAR] done output={output_dir}", flush=True)
            return 0 if report["passed_under_2p5m"] else 1

        robot_meta = configure_robot_poses(env)
        (output_dir / "robot_pose_plan.json").write_text(json.dumps(robot_meta, indent=2), encoding="utf-8")

        active_obstacles = configure_obstacles(
            env,
            active=bool(case_cfg["active_obstacles"]),
            active_count=int(args.active_obstacles),
            radius=10.0,
        )
        update_lidar_now(env, int(args.settle_steps))
        rows = summarize_case(
            env=env,
            case_dir=output_dir / str(args.case),
            case_name=str(args.case),
            active_obstacles=active_obstacles,
            max_points=int(args.max_points),
            max_env_plots=int(args.max_env_plots),
            save_stride_compare=bool(args.save_stride_compare),
            theta_min=float(args.theta_min),
            theta_max=float(args.theta_max),
            phi_min=float(args.phi_min),
            phi_max=float(args.phi_max),
            delta_theta=float(args.delta_theta),
            delta_phi=float(args.delta_phi),
        )
        all_rows: list[dict[str, Any]] = rows

        with (output_dir / "all_cases_summary.json").open("w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)
        if all_rows:
            with (output_dir / "all_cases_summary.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)

        per_case = {}
        for row in all_rows:
            item = per_case.setdefault(row["case"], {"max_closeness": -1.0, "mean_closeness": []})
            item["max_closeness"] = max(item["max_closeness"], float(row["closeness_max"]))
            item["mean_closeness"].append(float(row["closeness_mean"]))
        report = {
            "output_dir": str(output_dir),
            "case": str(args.case),
            "mesh_prim_paths": list(getattr(lidar.cfg, "mesh_prim_paths", [])),
            "case_rollup": {
                name: {
                    "max_closeness": vals["max_closeness"],
                    "mean_of_env_mean_closeness": float(np.mean(vals["mean_closeness"])) if vals["mean_closeness"] else float("nan"),
                }
                for name, vals in per_case.items()
            },
            "interpretation_hints": [
                "Run each case in a fresh process because mesh_prim_paths are fixed before RayCaster initialization.",
                "Compare case_rollup across output directories, not inside one directory.",
                "If ground_only has non-trivial closeness, ground is entering the policy theta band.",
                "If top_wall_only or bottom_wall_only has non-trivial closeness, workspace Z walls are entering the policy theta band.",
                "If obstacle_only is clean but all_training_sources is not, the issue is environmental target selection rather than obstacle ray casting.",
                "Use *_pointcloud3d_filtered_theta.png for the actual theta band used by the 72-D policy LiDAR input.",
                "Compare *_pointcloud3d_random.png with *_pointcloud3d_stride.png to identify visualization aliasing from sequential Livox samples.",
            ],
        }
        (output_dir / "diagnostic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[TEST2_LIDAR] rollup:", json.dumps(report["case_rollup"], indent=2), flush=True)
        print(f"[TEST2_LIDAR] done output={output_dir}", flush=True)
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
