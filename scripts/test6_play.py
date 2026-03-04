from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser("Test6 skrl PPO play (IsaacLab)")

parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-Lidar-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_obstacles", type=int, default=50)

parser.add_argument("--num_steps", type=int, default=3000)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to skrl checkpoint file (e.g. .../logs/<run>/checkpoints/<file>.pt)",
)

parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions (sampled)")
parser.add_argument("--state_dim", type=int, default=19)
parser.add_argument("--lidar_dim", type=int, default=4320)
parser.add_argument("--feat_dim", type=int, default=256)

# AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Launch Isaac Sim
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports AFTER app launch
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm

# register tasks
import omniperception_isaacdrone.tasks.test6_registry as _test6_registry  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg
from omniperception_isaacdrone.envs.test6_env import ObstacleSpawner

# skrl
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


# -----------------------------------------------------------------------------
# Same models as training
# -----------------------------------------------------------------------------
class StructuredFeatureExtractor(nn.Module):
    def __init__(self, state_dim: int, lidar_dim: int, feat_dim: int = 256):
        super().__init__()
        self.state_dim = int(state_dim)
        self.lidar_dim = int(lidar_dim)
        self.feat_dim = int(feat_dim)

        self.state_net = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.Tanh(),
        )
        self.lidar_net = nn.Sequential(
            nn.Linear(self.lidar_dim, 256),
            nn.Tanh(),
        )
        self.fuse_net = nn.Sequential(
            nn.Linear(512, self.feat_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        s = obs[:, : self.state_dim]
        l = obs[:, self.state_dim : self.state_dim + self.lidar_dim]
        s_feat = self.state_net(s)
        l_feat = self.lidar_net(l)
        return self.fuse_net(torch.cat([s_feat, l_feat], dim=-1))


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, state_dim: int, lidar_dim: int, feat_dim: int = 256):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True)

        obs_dim = self.num_observations
        expected = int(state_dim) + int(lidar_dim)
        if obs_dim != expected:
            raise RuntimeError(
                f"[Policy] Observation dim mismatch: obs_dim={obs_dim} but state_dim+lidar_dim={expected} "
                f"(state_dim={state_dim}, lidar_dim={lidar_dim})."
            )

        self.fe = StructuredFeatureExtractor(state_dim=state_dim, lidar_dim=lidar_dim, feat_dim=feat_dim)
        self.mean = nn.Linear(feat_dim, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        obs = inputs["states"]
        feat = self.fe(obs)
        mean = self.mean(feat)
        log_std = self.log_std_parameter.expand_as(mean)
        return mean, log_std, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, state_dim: int, lidar_dim: int, feat_dim: int = 256):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        obs_dim = self.num_observations
        expected = int(state_dim) + int(lidar_dim)
        if obs_dim != expected:
            raise RuntimeError(
                f"[Value] Observation dim mismatch: obs_dim={obs_dim} but state_dim+lidar_dim={expected} "
                f"(state_dim={state_dim}, lidar_dim={lidar_dim})."
            )

        self.fe = StructuredFeatureExtractor(state_dim=state_dim, lidar_dim=lidar_dim, feat_dim=feat_dim)
        self.v = nn.Linear(feat_dim, 1)

    def compute(self, inputs, role):
        obs = inputs["states"]
        feat = self.fe(obs)
        v = self.v(feat)
        return v, {}


def _try_load_checkpoint(agent: PPO, checkpoint_path: str):
    # 1) Preferred: agent.load(...)
    if hasattr(agent, "load"):
        try:
            agent.load(checkpoint_path)
            print(f"[INFO] Loaded checkpoint via agent.load: {checkpoint_path}", flush=True)
            return
        except Exception as e:
            print(f"[WARN] agent.load failed: {e}", flush=True)

    # 2) Fallback: torch.load and try common keys
    ckpt = torch.load(checkpoint_path, map_location=agent.device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unknown checkpoint format (not a dict): type={type(ckpt)}")

    # common patterns
    if "models" in ckpt and isinstance(ckpt["models"], dict):
        models_dict = ckpt["models"]
        if "policy" in models_dict:
            agent.models["policy"].load_state_dict(models_dict["policy"])
        if "value" in models_dict:
            agent.models["value"].load_state_dict(models_dict["value"])
        print(f"[INFO] Loaded checkpoint from ckpt['models']: {checkpoint_path}", flush=True)
        return

    if "policy" in ckpt and isinstance(ckpt["policy"], dict):
        agent.models["policy"].load_state_dict(ckpt["policy"])
        if "value" in ckpt and isinstance(ckpt["value"], dict):
            agent.models["value"].load_state_dict(ckpt["value"])
        print(f"[INFO] Loaded checkpoint from ckpt['policy'/'value']: {checkpoint_path}", flush=True)
        return

    raise RuntimeError(f"Unsupported checkpoint dict keys: {list(ckpt.keys())[:30]}")


def main():
    print(f"[INFO] task={args.task}, num_envs={args.num_envs}, device={args.device}", flush=True)
    ckpt_path = str(Path(args.checkpoint).expanduser().resolve())
    print(f"[INFO] checkpoint: {ckpt_path}", flush=True)

    # env cfg
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    print("[INFO] env_cfg parsed", flush=True)

    # obstacles
    print("[INFO] Spawning shared obstacles...", flush=True)
    ObstacleSpawner(num_obstacles=int(args.num_obstacles)).spawn_obstacles()

    # create base env
    base_env = gym.make(args.task, cfg=env_cfg).unwrapped
    try:
        base_env.reset()
        print("[INFO] base_env.reset() ok", flush=True)
    except Exception as e:
        print(f"[WARN] base_env.reset() failed before wrap (will continue): {e}", flush=True)

    env = wrap_env(base_env, wrapper="isaaclab")

    print(f"[INFO] observation_space: {env.observation_space}", flush=True)
    print(f"[INFO] action_space: {env.action_space}", flush=True)

    # build models/agent (for convenience)
    models = {
        "policy": Policy(env.observation_space, env.action_space, env.device, args.state_dim, args.lidar_dim, args.feat_dim),
        "value": Value(env.observation_space, env.action_space, env.device, args.state_dim, args.lidar_dim, args.feat_dim),
    }

    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg["rollouts"] = 1  # not used in play
    agent_cfg["learning_epochs"] = 1
    agent_cfg["mini_batches"] = 1
    agent_cfg["experiment"]["write_interval"] = 0
    agent_cfg["experiment"]["checkpoint_interval"] = 0

    memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=env.device)

    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # load checkpoint
    _try_load_checkpoint(agent, ckpt_path)

    # play loop
    states, infos = env.reset()

    print("[INFO] Start playing...", flush=True)
    for t in tqdm(range(int(args.num_steps)), ncols=100):
        with torch.no_grad():
            outputs = agent.act(states, timestep=t, timesteps=int(args.num_steps))
            if args.stochastic:
                actions = outputs[0]
            else:
                # deterministic: mean actions if available
                extra = outputs[-1] if isinstance(outputs[-1], dict) else {}
                actions = extra.get("mean_actions", outputs[0])

        states, rewards, terminated, truncated, infos = env.step(actions)

        if not args.headless:
            env.render()

    env.close()
    print("[INFO] Play finished", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Unhandled exception:\n", flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
        print("[INFO] Simulation app closed", flush=True)
