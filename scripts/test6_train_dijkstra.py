"""
Training script for IsaacLab Drone with Dijkstra-based navigation reward.

This script extends the standard SKRL training with additional monitoring
and logging for the Dijkstra navigation reward.

Usage:
    # Train with Dijkstra reward (default settings)
    python test6_train_dijkstra.py --num_envs 32 --num_obstacles 100

    # Train with custom Dijkstra parameters
    python test6_train_dijkstra.py --num_envs 64 --grid_size 200 --update_interval 3

    # Quick test run
    python test6_train_dijkstra.py --num_envs 4 --timesteps 100000 --num_obstacles 20
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)

# -----------------------------------------------------------------------------
# CLI with Dijkstra-specific arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    "Train IsaacLab drone with Dijkstra navigation reward (SKRL PPO)"
)
parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-Lidar-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_obstacles", type=int, default=100)
parser.add_argument("--timesteps", type=int, default=2_000_000)
parser.add_argument("--seed", type=int, default=42)

# Dijkstra-specific parameters
parser.add_argument("--dijkstra_weight", type=float, default=20.0,
                    help="Weight for Dijkstra progress reward")
parser.add_argument("--grid_size", type=int, default=160,
                    help="Dijkstra grid resolution (grid_size x grid_size)")
parser.add_argument("--cell_size", type=float, default=1.0,
                    help="Grid cell size in meters")
parser.add_argument("--update_interval", type=int, default=5,
                    help="Recompute distance field every N steps")
parser.add_argument("--dijkstra_speed_ref", type=float, default=4.0,
                    help="Reference speed for Dijkstra reward normalization")
parser.add_argument("--dijkstra_clip", type=float, default=1.0,
                    help="Clip Dijkstra reward to [-clip, clip]")

# Reward ablation options
parser.add_argument("--disable_progress_reward", action="store_true",
                    help="Disable standard progress_to_goal reward")
parser.add_argument("--disable_dist_reward", action="store_true",
                    help="Disable dist_to_goal reward")
parser.add_argument("--progress_weight", type=float, default=40.0)
parser.add_argument("--dist_weight", type=float, default=5.0)

# Standard training parameters
parser.add_argument("--state_dim", type=int, default=17)
parser.add_argument("--lidar_dim", type=int, default=432)
parser.add_argument("--feat_dim", type=int, default=256)
parser.add_argument("--rollouts", type=int, default=256)
parser.add_argument("--learning_epochs", type=int, default=8)
parser.add_argument("--mini_batches", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--_lambda", type=float, default=0.97)
parser.add_argument("--discount_factor", type=float, default=0.99)
parser.add_argument("--ratio_clip", type=float, default=0.2)
parser.add_argument("--value_clip", type=float, default=0.2)
parser.add_argument("--entropy_coef", type=float, default=1e-2)
parser.add_argument("--grad_norm_clip", type=float, default=1.0)
parser.add_argument("--reward_scale", type=float, default=0.1)
parser.add_argument("--reward_clip", type=float, default=100.0)

# Logging
parser.add_argument("--tb_interval", type=int, default=500)
parser.add_argument("--checkpoint_interval", type=int, default=50000)
parser.add_argument("--log_dijkstra_freq", type=int, default=100,
                    help="Log detailed Dijkstra stats every N steps")

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
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import omniperception_isaacdrone.tasks.test6_registry as _test6_registry  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from omniperception_isaacdrone.envs.test6_env import WallSpawner, setup_global_obstacles

# Import the training utilities from the main training script
import sys
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
from test6_train_skrl import (
    Policy, Value, SkrlSpaceAdapter, extract_policy_obs,
    sanitize_states, ensure_obs_shape, ensure_action_shape,
    extract_actions, scale_rewards, build_skrl_spaces,
    get_state_lidar_dims, get_env_step_dt, extract_reward_weights,
    build_reward_term_views, clear_tb_caches, snapshot_models, models_are_finite,
    RewardBreakdownAccumulator, InfoTerminationRatioAccumulator,
    RollingHistogramLogger, SkrlLossMirror, log_gradients,
    build_state_names, build_action_names, to_float, sanitize_tb_tag,
    extract_prefixed_log_scalars, gaussian_mixin_kwargs,
)


def log_dijkstra_stats(writer: SummaryWriter, env, step: int):
    """Log detailed Dijkstra statistics to TensorBoard."""
    if not hasattr(env, "_dijkstra_navigator") or env._dijkstra_navigator is None:
        return

    nav = env._dijkstra_navigator
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w[:, :3]
    goal = env._goal_pos_w

    # Compute statistics across all environments
    geodesic_dists = []
    euclidean_dists = []

    for i in range(env.num_envs):
        if env._dijkstra_distance_fields is not None:
            d_geo = nav.get_geodesic_distance(
                pos[i:i+1], env._dijkstra_distance_fields[i]
            )[0].item()
            d_euc = torch.norm(pos[i, :2] - goal[i, :2]).item()
            geodesic_dists.append(d_geo)
            euclidean_dists.append(d_euc)

    if geodesic_dists:
        geo_tensor = torch.tensor(geodesic_dists)
        euc_tensor = torch.tensor(euclidean_dists)
        ratio = geo_tensor / torch.clamp(euc_tensor, min=0.1)

        writer.add_scalar("Dijkstra/geodesic_mean", geo_tensor.mean().item(), step)
        writer.add_scalar("Dijkstra/geodesic_std", geo_tensor.std().item(), step)
        writer.add_scalar("Dijkstra/euclidean_mean", euc_tensor.mean().item(), step)
        writer.add_scalar("Dijkstra/path_ratio_mean", ratio.mean().item(), step)
        writer.add_scalar("Dijkstra/path_ratio_max", ratio.max().item(), step)

        # Histogram of path ratios (shows how much detour is needed)
        writer.add_histogram("Dijkstra/path_ratio_dist", ratio, step)

        # Log grid coverage stats
        if env._dijkstra_distance_fields is not None:
            unreachable = (env._dijkstra_distance_fields >= nav.max_distance * 0.99).float()
            writer.add_scalar("Dijkstra/unreachable_cells_mean",
                              unreachable.mean().item() * 100, step)  # Percentage


def main():
    print(f"[INFO] Dijkstra Navigation Training")
    print(f"[INFO] num_envs={args.num_envs}, grid_size={args.grid_size}x{args.grid_size}")
    print(f"[INFO] Dijkstra weight={args.dijkstra_weight}, update_interval={args.update_interval}")

    # Parse environment config
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )

    # Configure Dijkstra reward parameters
    env_cfg.rewards.dijkstra_progress.weight = args.dijkstra_weight
    env_cfg.rewards.dijkstra_progress.params["grid_size"] = args.grid_size
    env_cfg.rewards.dijkstra_progress.params["cell_size"] = args.cell_size
    env_cfg.rewards.dijkstra_progress.params["update_interval"] = args.update_interval
    env_cfg.rewards.dijkstra_progress.params["speed_ref"] = args.dijkstra_speed_ref
    env_cfg.rewards.dijkstra_progress.params["clip"] = args.dijkstra_clip

    # Optionally disable other navigation rewards for ablation study
    if args.disable_progress_reward:
        print("[INFO] Disabling progress_to_goal reward")
        env_cfg.rewards.progress_to_goal.weight = 0.0
    else:
        env_cfg.rewards.progress_to_goal.weight = args.progress_weight

    if args.disable_dist_reward:
        print("[INFO] Disabling dist_to_goal reward")
        env_cfg.rewards.dist_to_goal.weight = 0.0
    else:
        env_cfg.rewards.dist_to_goal.weight = args.dist_weight

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Spawn walls and obstacles
    print("[INFO] Spawning workspace walls...")
    WallSpawner(
        x_bounds=(-80.0, 80.0),
        y_bounds=(-80.0, 80.0),
        z_bounds=(0.0, 10.0),
        wall_thickness=0.5,
        color=(0.7, 0.7, 0.2),
    ).spawn_walls()

    print(f"[INFO] Setting up {args.num_obstacles} global obstacles...")
    setup_global_obstacles(args.num_obstacles)

    # Create environment
    print("[INFO] Creating environment...")
    base_env = gym.make(args.task, cfg=env_cfg).unwrapped

    # Build observation/action spaces for SKRL
    space = getattr(base_env, "single_observation_space", None)
    policy_space = space.spaces.get("policy", None) if isinstance(space, gym.spaces.Dict) else getattr(base_env, "observation_space", None)
    obs_dim_raw = int(np.prod(policy_space.shape))
    state_dim, lidar_dim = get_state_lidar_dims(base_env, obs_dim_raw)
    obs_dim, act_dim, obs_space, act_space = build_skrl_spaces(base_env, state_dim, lidar_dim)

    adapted_env = SkrlSpaceAdapter(base_env, obs_space=obs_space, act_space=act_space,
                                   state_dim=state_dim, lidar_dim=lidar_dim)
    env = wrap_env(adapted_env, wrapper="isaaclab")

    num_envs = int(getattr(env, "num_envs", args.num_envs))
    device = torch.device(getattr(env, "device", args.device))
    step_dt = get_env_step_dt(base_env)

    # Build models
    models = {
        "policy": Policy(obs_space, act_space, device, state_dim, lidar_dim, args.feat_dim),
        "value": Value(obs_space, act_space, device, state_dim, lidar_dim, args.feat_dim),
    }

    # PPO configuration
    cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
    cfg["rollouts"] = int(args.rollouts)
    cfg["learning_epochs"] = int(args.learning_epochs)
    cfg["mini_batches"] = int(args.mini_batches)
    cfg["discount_factor"] = float(args.discount_factor)
    cfg["lambda"] = float(args._lambda)
    cfg["learning_rate"] = float(args.learning_rate)
    cfg["ratio_clip"] = float(args.ratio_clip)
    cfg["value_clip"] = float(args.value_clip)
    cfg["entropy_loss_scale"] = float(args.entropy_coef)
    cfg["grad_norm_clip"] = float(args.grad_norm_clip)
    cfg["clip_predicted_values"] = True

    # Setup logging directories
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    log_root = project_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f") + "_DijkstraPPO"
    exp_dir = log_root / run_name
    tb_dir = exp_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    cfg["experiment"]["directory"] = str(log_root)
    cfg["experiment"]["experiment_name"] = run_name
    cfg["experiment"]["write_interval"] = int(args.tb_interval)
    cfg["experiment"]["checkpoint_interval"] = int(args.checkpoint_interval)

    writer = SummaryWriter(log_dir=str(tb_dir))

    # Log hyperparameters
    hparams = {
        "num_envs": args.num_envs,
        "num_obstacles": args.num_obstacles,
        "dijkstra_weight": args.dijkstra_weight,
        "grid_size": args.grid_size,
        "cell_size": args.cell_size,
        "update_interval": args.update_interval,
        "progress_weight": 0.0 if args.disable_progress_reward else args.progress_weight,
        "dist_weight": 0.0 if args.disable_dist_reward else args.dist_weight,
        "learning_rate": args.learning_rate,
        "rollouts": args.rollouts,
    }
    writer.add_hparams(hparams, {})

    # Create agent
    memory = RandomMemory(memory_size=int(args.rollouts), num_envs=num_envs, device=device)
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )
    agent.init()

    # Training utilities
    loss_mirror = SkrlLossMirror()
    loss_mirror.bind(agent)

    raw_obs, infos = env.reset()
    states = sanitize_states(ensure_obs_shape(extract_policy_obs(raw_obs), num_envs, obs_dim),
                             state_dim=state_dim, lidar_dim=lidar_dim)
    last_good_snapshot = snapshot_models(models)

    reward_weights = extract_reward_weights(base_env)
    reward_window = RewardBreakdownAccumulator()
    termination_ratio_window = InfoTerminationRatioAccumulator()
    hist_logger = RollingHistogramLogger(
        obs_names=build_state_names(state_dim),
        action_names=build_action_names(act_dim),
        window=10,
        max_samples=2048,
    )

    latest_curriculum_log = {}
    pbar = tqdm(range(int(args.timesteps)), ncols=110)

    try:
        for t in pbar:
            global_step = t + 1
            agent.pre_interaction(timestep=t, timesteps=int(args.timesteps))

            with torch.no_grad():
                act_output = agent.act(states, timestep=t, timesteps=int(args.timesteps))

            actions = ensure_action_shape(extract_actions(act_output, act_dim), num_envs, act_dim).float()
            actions = torch.clamp(torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

            rollout_boundary = (global_step % int(args.rollouts) == 0)
            if rollout_boundary:
                last_good_snapshot = snapshot_models(models)

            next_obs, rewards, terminated, truncated, infos = env.step(actions)

            next_states = sanitize_states(ensure_obs_shape(extract_policy_obs(next_obs), num_envs, obs_dim),
                                         state_dim=state_dim, lidar_dim=lidar_dim)
            train_rewards = scale_rewards(rewards, scale=args.reward_scale, clip=args.reward_clip)

            raw_terms, weighted_terms, scaled_terms = build_reward_term_views(
                base_env, reward_weights=reward_weights,
                reward_scale=float(args.reward_scale), reward_clip=float(args.reward_clip)
            )
            reward_window.update(raw_terms=raw_terms, weighted_terms=weighted_terms, scaled_terms=scaled_terms)
            clear_tb_caches(base_env)

            done_count = int((terminated | truncated).sum().item())
            termination_ratio_window.update(infos, done_count=done_count)

            if done_count > 0:
                if len(curriculum_log_dict := extract_prefixed_log_scalars(infos, "Curriculum/")) > 0:
                    latest_curriculum_log.update(curriculum_log_dict)

            with torch.no_grad():
                agent.record_transition(
                    states=states, actions=actions, rewards=train_rewards,
                    next_states=next_states, terminated=terminated, truncated=truncated,
                    infos=infos, timestep=t, timesteps=int(args.timesteps)
                )

            agent.post_interaction(timestep=t, timesteps=int(args.timesteps))
            if rollout_boundary:
                loss_mirror.flush(writer, global_step)

            if rollout_boundary and not models_are_finite(models):
                debug_dir = exp_dir / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                torch.save(last_good_snapshot, debug_dir / f"last_good_before_nan_t{t}.pt")
                raise RuntimeError(f"Non-finite model parameters at t={t}")

            if not args.headless:
                try:
                    env.render()
                except Exception:
                    pass

            # Logging
            should_log_scalars = int(args.tb_interval) > 0 and ((global_step % int(args.tb_interval) == 0)
                                                                or (global_step == int(args.timesteps)))

            if should_log_scalars:
                for key, value in sorted(latest_curriculum_log.items()):
                    writer.add_scalar(sanitize_tb_tag(key), value, global_step)
                reward_window.flush(writer, global_step)
                termination_ratio_window.flush(writer, global_step)

                # Log Dijkstra-specific stats
                log_dijkstra_stats(writer, base_env, global_step)

                writer.flush()
                reward_window.reset()
                termination_ratio_window.reset()

            if int(args.checkpoint_interval) > 0 and (global_step % int(args.checkpoint_interval) == 0):
                ckpt_dir = exp_dir / "manual_checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({name: model.state_dict() for name, model in models.items()},
                          ckpt_dir / f"models_t{global_step}.pt")

            if int(args.cuda_clean_interval) > 0 and (global_step % int(args.cuda_clean_interval) == 0):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.set_description(f"t={t} envR={rewards.mean().item():+.3f} trainR={train_rewards.mean().item():+.3f} done={done_count}")
            states = next_states

    except KeyboardInterrupt:
        print("\n[WARN] KeyboardInterrupt: stopping training early")
    finally:
        print("[INFO] Training loop exited")
        try:
            writer.close()
        except:
            pass
        try:
            env.close()
        except:
            pass


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
