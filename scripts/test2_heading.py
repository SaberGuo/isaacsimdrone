from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)

parser = argparse.ArgumentParser("Validate yaw-rate control for test6 drone env")
parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--warmup_steps", type=int, default=30)
parser.add_argument("--yaw_action", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaacsim.core.utils.prims as prim_utils
import omniperception_isaacdrone.tasks.test6_registry as _test6_registry  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from pxr import Gf, UsdGeom

from omniperception_isaacdrone.envs.test6_env import WallSpawner, setup_global_obstacles


def quat_to_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def set_prim_translation(prim_path: str, translation: tuple[float, float, float]) -> None:
    stage = prim_utils.get_prim_at_path("/World").GetStage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(*translation))
    else:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=int(args_cli.num_envs),
        use_fabric=not args_cli.disable_fabric,
    )
    WallSpawner(
        x_bounds=(-80.0, 80.0),
        y_bounds=(-80.0, 80.0),
        z_bounds=(0.0, 10.0),
        wall_thickness=0.5,
    ).spawn_walls()
    setup_global_obstacles(1)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    try:
        obs, _ = env.reset(seed=int(args_cli.seed))
        _ = obs
        set_prim_translation("/World/Obstacles/obj_000", (1000.0, 1000.0, -1000.0))

        zero_action = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
        for _ in range(int(args_cli.warmup_steps)):
            env.step(zero_action)

        yaw_action = float(args_cli.yaw_action)
        action = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
        action[:, 3] = yaw_action

        yaw_rate_scale = float(getattr(env.cfg.actions.root_twist, "params", {}).get("yaw_rate_scale", 1.0))
        expected_yaw_rate = yaw_action * yaw_rate_scale
        step_dt = float(env.step_dt)

        yaw_history = []
        body_yaw_rate_history = []

        for _ in range(int(args_cli.steps)):
            env.step(action)
            quat = env.scene["robot"].data.root_quat_w.detach()[0:1]
            ang_vel_b = env.scene["robot"].data.root_ang_vel_b.detach()[0:1]
            yaw_history.append(float(quat_to_yaw_wxyz(quat)[0].item()))
            body_yaw_rate_history.append(float(ang_vel_b[0, 2].item()))

        yaw_tensor = torch.tensor(yaw_history, dtype=torch.float32)
        yaw_diff = wrap_to_pi(yaw_tensor[1:] - yaw_tensor[:-1])
        achieved_yaw_rate = yaw_diff / step_dt

        achieved_mean = float(achieved_yaw_rate.mean().item())
        achieved_std = float(achieved_yaw_rate.std(unbiased=False).item())
        body_rate_mean = float(torch.tensor(body_yaw_rate_history, dtype=torch.float32).mean().item())
        yaw_span = float((yaw_tensor[-1] - yaw_tensor[0]).item())
        abs_error = abs(achieved_mean - expected_yaw_rate)

        print("[HEADING_TEST] ===== Result =====", flush=True)
        print(f"[HEADING_TEST] step_dt={step_dt:.6f}s", flush=True)
        print(f"[HEADING_TEST] commanded_action={yaw_action:+.3f}", flush=True)
        print(f"[HEADING_TEST] expected_yaw_rate={expected_yaw_rate:+.6f} rad/s", flush=True)
        print(f"[HEADING_TEST] achieved_yaw_rate_mean={achieved_mean:+.6f} rad/s", flush=True)
        print(f"[HEADING_TEST] achieved_yaw_rate_std={achieved_std:+.6f} rad/s", flush=True)
        print(f"[HEADING_TEST] body_z_rate_mean={body_rate_mean:+.6f} rad/s", flush=True)
        print(f"[HEADING_TEST] yaw_span={yaw_span:+.6f} rad ({math.degrees(yaw_span):+.2f} deg)", flush=True)
        print(f"[HEADING_TEST] abs_error={abs_error:.6f} rad/s", flush=True)

        passed = abs(yaw_span) > 0.25 and abs_error < 0.60
        print(f"[HEADING_TEST] PASS={passed}", flush=True)
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
