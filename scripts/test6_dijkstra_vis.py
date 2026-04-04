"""
Dijkstra Navigation Reward Visualization Script for IsaacLab Drone.

This script visualizes the Dijkstra distance field and geodesic path in real-time,
helping to debug and verify the Dijkstra-based navigation reward.

Usage:
    python test6_dijkstra_vis.py --num_envs 1 --num_obstacles 20
    python test6_dijkstra_vis.py --num_envs 4 --num_obstacles 50 --grid_size 160
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    "Visualize Dijkstra navigation reward for IsaacLab drone"
)
parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-Lidar-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--num_obstacles", type=int, default=30, help="Number of obstacles to spawn")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--grid_size", type=int, default=160, help="Dijkstra grid resolution")
parser.add_argument("--cell_size", type=float, default=1.0, help="Grid cell size in meters")
parser.add_argument("--update_interval", type=int, default=5, help="Distance field update interval")
parser.add_argument("--max_steps", type=int, default=10000, help="Maximum simulation steps")
parser.add_argument("--print_freq", type=int, default=50, help="Print stats every N steps")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Launch Isaac Sim
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Runtime imports
# -----------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch

import omniperception_isaacdrone.tasks.test6_registry as _test6_registry  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from omniperception_isaacdrone.envs.test6_env import WallSpawner, setup_global_obstacles


def print_dijkstra_stats(env, step: int):
    """Print Dijkstra navigation statistics."""
    if not hasattr(env, "_dijkstra_navigator") or env._dijkstra_navigator is None:
        print(f"[Step {step}] Dijkstra navigator not initialized yet")
        return

    nav = env._dijkstra_navigator
    num_envs = env.num_envs

    print(f"\n{'='*60}")
    print(f"[Step {step}] Dijkstra Navigation Stats")
    print(f"{'='*60}")
    print(f"Grid size: {nav.grid_size}x{nav.grid_size}, Cell size: {nav.cell_size}m")
    print(f"Workspace origin: {nav.workspace_origin}")

    # Get current positions and goals
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w[:, :3]
    goal = env._goal_pos_w  # (N, 3)

    for i in range(min(num_envs, 4)):  # Print first 4 envs
        print(f"\n[Env {i}]")
        print(f"  Position: ({pos[i, 0]:.2f}, {pos[i, 1]:.2f}, {pos[i, 2]:.2f})")
        print(f"  Goal:     ({goal[i, 0]:.2f}, {goal[i, 1]:.2f}, {goal[i, 2]:.2f})")

        if env._dijkstra_distance_fields is not None:
            d_current = nav.get_geodesic_distance(
                pos[i:i+1], env._dijkstra_distance_fields[i]
            )[0].item()
            d_euclidean = torch.norm(pos[i, :2] - goal[i, :2]).item()

            print(f"  Geodesic dist: {d_current:.2f}m")
            print(f"  Euclidean dist: {d_euclidean:.2f}m")
            print(f"  Path ratio: {d_current / max(d_euclidean, 0.1):.2f}x")

            # Check if goal is reachable
            goal_grid = nav.world_to_grid(goal[i:i+1])[0]
            gx, gy = int(goal_grid[0].item()), int(goal_grid[1].item())
            if 0 <= gx < nav.grid_size and 0 <= gy < nav.grid_size:
                goal_dist = env._dijkstra_distance_fields[i, gy, gx].item()
                if goal_dist >= nav.max_distance * 0.99:
                    print(f"  WARNING: Goal may be unreachable!")

    # Print update counter stats
    if hasattr(env, "_dijkstra_update_counter"):
        counters = env._dijkstra_update_counter.cpu().numpy()
        print(f"\n[Update Counters] mean={counters.mean():.1f}, max={counters.max()}")


def visualize_distance_field_2d(env, env_idx: int = 0):
    """Print ASCII visualization of distance field for one environment."""
    if env._dijkstra_distance_fields is None:
        return

    nav = env._dijkstra_navigator
    field = env._dijkstra_distance_fields[env_idx].cpu().numpy()

    # Get robot and goal positions in grid coords
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w[env_idx:env_idx+1, :3]
    goal = env._goal_pos_w[env_idx:env_idx+1, :3]

    robot_grid = nav.world_to_grid(pos)[0].cpu().numpy().astype(int)
    goal_grid = nav.world_to_grid(goal)[0].cpu().numpy().astype(int)

    rx, ry = int(robot_grid[0]), int(robot_grid[1])
    gx, gy = int(goal_grid[0]), int(goal_grid[1])

    # Create a compact visualization
    size = nav.grid_size
    step = max(1, size // 40)  # Downsample to ~40 chars

    print(f"\n[Env {env_idx}] Distance Field (R=robot, G=goal, #=obstacle, .=free):")
    print("-" * (size // step + 2))

    for y in range(0, size, step):
        row = "|"
        for x in range(0, size, step):
            if abs(x - rx) < step and abs(y - ry) < step:
                row += "R"
            elif abs(x - gx) < step and abs(y - gy) < step:
                row += "G"
            elif field[y, x] >= nav.max_distance * 0.99:
                row += "#"
            elif field[y, x] < 0:
                row += "?"
            else:
                # Show distance as intensity
                max_show = 100.0
                val = field[y, x]
                if val > max_show:
                    row += "."
                elif val > max_show * 0.7:
                    row += "o"
                elif val > max_show * 0.4:
                    row += "+"
                elif val > max_show * 0.1:
                    row += "="
                else:
                    row += "*"
        row += "|"
        print(row)
    print("-" * (size // step + 2))


def main():
    print(f"[INFO] Starting Dijkstra visualization")
    print(f"[INFO] num_envs={args.num_envs}, grid_size={args.grid_size}x{args.grid_size}")

    # Parse environment config
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )

    # Override Dijkstra params in config
    env_cfg.rewards.dijkstra_progress.params["grid_size"] = args.grid_size
    env_cfg.rewards.dijkstra_progress.params["cell_size"] = args.cell_size
    env_cfg.rewards.dijkstra_progress.params["update_interval"] = args.update_interval

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Spawn walls
    print("[INFO] Spawning workspace walls...")
    WallSpawner(
        x_bounds=(-80.0, 80.0),
        y_bounds=(-80.0, 80.0),
        z_bounds=(0.0, 10.0),
        wall_thickness=0.5,
        color=(0.7, 0.7, 0.2),
    ).spawn_walls()

    # Setup obstacles
    print(f"[INFO] Setting up {args.num_obstacles} global obstacles...")
    setup_global_obstacles(args.num_obstacles)

    # Create environment
    print("[INFO] Creating environment...")
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    # Reset to initialize
    print("[INFO] Resetting environment...")
    obs, info = env.reset()

    print("\n" + "="*60)
    print("Dijkstra Navigation Reward Visualization")
    print("="*60)
    print("Controls:")
    print("  - Press Ctrl+C to exit")
    print("  - Watch the console for distance field updates")
    print("="*60 + "\n")

    step = 0
    try:
        while step < args.max_steps:
            # Random actions for demonstration (drone will move randomly)
            actions = torch.randn(
                args.num_envs, 4, device=env.device, dtype=torch.float32
            )
            actions = torch.clamp(actions, -1.0, 1.0)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)

            # Print stats periodically
            if step % args.print_freq == 0:
                print_dijkstra_stats(env, step)

                # Visualize first environment's distance field
                if args.num_envs == 1 and step % (args.print_freq * 5) == 0:
                    visualize_distance_field_2d(env, env_idx=0)

            # Reset if done
            done = terminated | truncated
            if done.any():
                print(f"\n[Step {step}] Environments done: {done.sum().item()}")
                # Get reset indices
                env_ids = done.nonzero(as_tuple=False).squeeze(-1)
                for idx in env_ids:
                    print(f"  Env {idx.item()} reached goal or collided!")

            # Render
            if not args.headless:
                env.render()

            step += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        print("[INFO] Closing environment...")
        env.close()
        print(f"[INFO] Total steps: {step}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Unhandled exception:\n")
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
        print("[INFO] Simulation app closed")
