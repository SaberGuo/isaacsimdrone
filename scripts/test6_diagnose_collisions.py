from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SOURCE_DIR = PROJECT_DIR / "source"
PACKAGE_ROOT = SOURCE_DIR / "omniperception_isaacdrone"
for path in (PACKAGE_ROOT, SOURCE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Diagnose Test6 collision sources")
parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-Lidar-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--env_spacing", type=float, default=10.0)
parser.add_argument("--num_obstacles", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics, Gf

import omniperception_isaacdrone.tasks.test6_registry as _test6_registry  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from omniperception_isaacdrone.envs.test6_env import ObstacleSpawner, WallSpawner
from omniperception_isaacdrone.tasks.mdp.test6_terminations import termination_collision
from isaaclab.managers import SceneEntityCfg


def _vec3(v) -> tuple[float, float, float]:
    return float(v[0]), float(v[1]), float(v[2])


def _world_translation(prim) -> np.ndarray:
    cache = UsdGeom.XformCache()
    t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)


def _world_scale(prim) -> np.ndarray:
    cache = UsdGeom.XformCache()
    matrix = cache.GetLocalToWorldTransform(prim)
    sx = Gf.Vec3d(matrix[0][0], matrix[0][1], matrix[0][2]).GetLength()
    sy = Gf.Vec3d(matrix[1][0], matrix[1][1], matrix[1][2]).GetLength()
    sz = Gf.Vec3d(matrix[2][0], matrix[2][1], matrix[2][2]).GetLength()
    return np.array([float(sx), float(sy), float(sz)], dtype=np.float64)


def _cube_aabb(prim) -> tuple[np.ndarray, np.ndarray] | None:
    if not prim.IsValid():
        return None
    if not prim.IsA(UsdGeom.Cube):
        return None

    geom = UsdGeom.Cube(prim)
    size_attr = geom.GetSizeAttr()
    size = float(size_attr.Get()) if size_attr and size_attr.HasValue() else 1.0
    center = _world_translation(prim)
    scale = _world_scale(prim)
    full_size = size * scale
    half = 0.5 * full_size
    return center - half, center + half


def _iter_prefixed_cube_aabbs(stage, prefix: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(prefix)):
        if not prim.IsValid() or not prim.IsA(UsdGeom.Cube):
            continue
        aabb = _cube_aabb(prim)
        if aabb is None:
            continue
        out.append((prim.GetPath().pathString, aabb[0], aabb[1]))
    return out


def _overlap(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> bool:
    return bool(np.all(a_max >= b_min) and np.all(b_max >= a_min))


def _contact_force_summary(contact_sensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = contact_sensor.data
    fm_hist = getattr(data, "force_matrix_w_history", None)
    fm = None
    if isinstance(fm_hist, torch.Tensor) and fm_hist.dim() == 5 and fm_hist.shape[1] > 0:
        fm = fm_hist[:, 0]
    if fm is None:
        fm_now = getattr(data, "force_matrix_w", None)
        if isinstance(fm_now, torch.Tensor) and fm_now.dim() == 4:
            fm = fm_now
    if fm is not None:
        pair_norms = torch.norm(torch.nan_to_num(fm, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
        per_body_norms = pair_norms.max(dim=-1).values
        max_force = per_body_norms.max(dim=-1).values
        return max_force, torch.tensor([1] * max_force.shape[0], device=max_force.device), per_body_norms

    nf_hist = getattr(data, "net_forces_w_history", None)
    nf = None
    if isinstance(nf_hist, torch.Tensor) and nf_hist.dim() == 4 and nf_hist.shape[1] > 0:
        nf = nf_hist[:, 0]
    if nf is None:
        nf_now = getattr(data, "net_forces_w", None)
        if isinstance(nf_now, torch.Tensor) and nf_now.dim() == 3:
            nf = nf_now
    if nf is None:
        z = torch.zeros((contact_sensor.data.pos_w.shape[0],), device=contact_sensor.data.pos_w.device)
        return z, z, torch.zeros((z.shape[0], 1), device=z.device)

    per_body_norms = torch.norm(torch.nan_to_num(nf, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
    max_force = per_body_norms.reshape(per_body_norms.shape[0], -1).max(dim=-1).values
    return max_force, torch.zeros_like(max_force), per_body_norms


def _nearest_gap(a_min: np.ndarray, a_max: np.ndarray, boxes: list[tuple[str, np.ndarray, np.ndarray]]) -> tuple[str, float]:
    best_name = ""
    best_gap = float("inf")
    center_a = 0.5 * (a_min + a_max)
    for name, b_min, b_max in boxes:
        center_b = 0.5 * (b_min + b_max)
        gap = float(np.linalg.norm(center_a - center_b))
        if gap < best_gap:
            best_gap = gap
            best_name = name
    return best_name, best_gap


def diagnose(env, label: str) -> None:
    stage = env.sim.stage
    env_origins = env.scene.env_origins.detach().cpu().numpy()
    root_pos = env.scene["robot"].data.root_pos_w.detach().cpu().numpy()
    contact_sensor = env.scene["contact_sensor"]
    collision_flags = termination_collision(env, sensor_cfg=SceneEntityCfg("contact_sensor"), threshold=1.0)
    max_forces, used_force_matrix, per_body_norms = _contact_force_summary(contact_sensor)
    robot_body_names = list(env.scene["robot"].body_names)

    obstacle_boxes = _iter_prefixed_cube_aabbs(stage, "/World/Obstacles")
    wall_boxes = _iter_prefixed_cube_aabbs(stage, "/World/Wall")

    print(f"\n===== {label} =====", flush=True)
    print(f"obstacle_boxes={len(obstacle_boxes)}, wall_boxes={len(wall_boxes)}", flush=True)

    for env_id in range(env.num_envs):
        robot_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot/body/body_collision/geometry")
        robot_box = _cube_aabb(robot_prim)
        if robot_box is None:
            print(f"env_{env_id}: robot collision cube not found", flush=True)
            continue
        r_min, r_max = robot_box

        obstacle_hits = [name for name, b_min, b_max in obstacle_boxes if _overlap(r_min, r_max, b_min, b_max)]
        wall_hits = [name for name, b_min, b_max in wall_boxes if _overlap(r_min, r_max, b_min, b_max)]
        ground_hit = bool(r_min[2] <= 0.0)
        nearest_obstacle_name, nearest_obstacle_gap = _nearest_gap(r_min, r_max, obstacle_boxes)
        nearest_wall_name, nearest_wall_gap = _nearest_gap(r_min, r_max, wall_boxes)

        print(
            f"env_{env_id}: origin={tuple(np.round(env_origins[env_id], 3))} "
            f"root_pos={tuple(np.round(root_pos[env_id], 3))} "
            f"collision={bool(collision_flags[env_id].item())} "
            f"max_force={float(max_forces[env_id].item()):.4f} "
            f"force_source={'matrix' if int(used_force_matrix[env_id].item()) == 1 else 'net'}",
            flush=True,
        )
        print(
            f"         robot_aabb_min={tuple(np.round(r_min, 3))} robot_aabb_max={tuple(np.round(r_max, 3))}",
            flush=True,
        )
        print(
            f"         overlaps: ground={ground_hit} walls={len(wall_hits)} obstacles={len(obstacle_hits)}",
            flush=True,
        )
        if wall_hits:
            print(f"         wall_hits={wall_hits[:5]}", flush=True)
        if obstacle_hits:
            print(f"         obstacle_hits={obstacle_hits[:5]}", flush=True)
        print(
            f"         nearest_wall={nearest_wall_name} gap={nearest_wall_gap:.4f} "
            f"nearest_obstacle={nearest_obstacle_name} gap={nearest_obstacle_gap:.4f}",
            flush=True,
        )
        if per_body_norms.shape[1] > 0:
            body_force_pairs = []
            for body_index in range(min(per_body_norms.shape[1], len(robot_body_names))):
                body_force_pairs.append((robot_body_names[body_index], float(per_body_norms[env_id, body_index].item())))
            body_force_pairs.sort(key=lambda item: item[1], reverse=True)
            print(f"         top_body_forces={body_force_pairs[:3]}", flush=True)


def main() -> None:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.scene.env_spacing = float(args.env_spacing)
    setattr(env_cfg, "seed", int(args.seed))

    ObstacleSpawner(num_obstacles=int(max(args.num_obstacles, 100)), seed=int(args.seed)).spawn_obstacles()
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

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.scene.filter_collisions(global_prim_paths=["/World/ground", "/World/Obstacles", "/World/Wall"])

    env.reset()
    diagnose(env, "after reset")

    zero_actions = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
    env.step(zero_actions)
    diagnose(env, "after one zero-action step")

    env.close()


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
