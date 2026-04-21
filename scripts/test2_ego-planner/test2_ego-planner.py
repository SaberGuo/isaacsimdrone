from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

_SCRIPT_PATH = Path(__file__).resolve()
_WORKSPACE_PATH = _SCRIPT_PATH.parents[3]
_EXTRA_PYTHON_PATHS = [
    _WORKSPACE_PATH / "IsaacLab" / "source" / "isaaclab",
    _WORKSPACE_PATH / "IsaacLab" / "source" / "isaaclab_assets",
    _WORKSPACE_PATH / "IsaacLab" / "source" / "isaaclab_tasks",
    _WORKSPACE_PATH / "IsaacLab" / "source" / "isaaclab_rl",
    _WORKSPACE_PATH / "IsaacLab" / "source" / "isaaclab_mimic",
    _WORKSPACE_PATH / "IsaacLab" / "source" / "isaaclab_contrib",
    _WORKSPACE_PATH / "omniperception_isaacdrone" / "source" / "omniperception_isaacdrone",
]
for _path in _EXTRA_PYTHON_PATHS:
    if _path.exists():
        sys.path.insert(0, str(_path))

from isaaclab.app import AppLauncher

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)

parser = argparse.ArgumentParser("EGO-style local planner + PID tracker for IsaacLab drone lidar task")
parser.add_argument("--task", type=str, default="Isaac-OmniPerception-Drone-Lidar-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_obstacles", type=int, default=40)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--warmup_steps", type=int, default=6)
parser.add_argument("--print_every", type=int, default=30)
parser.add_argument("--auto_reset", action="store_true", default=True)
parser.add_argument("--no_auto_reset", dest="auto_reset", action="store_false")

parser.add_argument("--goal_x", type=float, default=None)
parser.add_argument("--goal_y", type=float, default=None)
parser.add_argument("--goal_z", type=float, default=None)

parser.add_argument("--local_map_resolution", type=float, default=0.4)
parser.add_argument("--local_map_x_half_range", type=float, default=16.0)
parser.add_argument("--local_map_y_half_range", type=float, default=16.0)
parser.add_argument("--local_map_z_down", type=float, default=4.0)
parser.add_argument("--local_map_z_up", type=float, default=4.0)
parser.add_argument("--inflation_radius", type=float, default=1.8)

parser.add_argument("--planner_max_vel", type=float, default=4.5)
parser.add_argument("--planner_max_acc", type=float, default=6.5)
parser.add_argument("--planning_horizon", type=float, default=10.0)
parser.add_argument("--ctrl_pt_dist", type=float, default=0.9)
parser.add_argument("--replan_period", type=float, default=0.9)
parser.add_argument("--goal_tolerance", type=float, default=0.6)
parser.add_argument("--action_vel_scale", type=float, default=4.5)
parser.add_argument("--action_vel_clip", type=float, default=5.0)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import omniperception_isaacdrone.tasks.test6_registry as _test6_registry  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from local_planner import EgoStyleLocalPlanner, PlannerConfig
from occupancy_map import LocalOccupancyMap, OccupancyStats, quat_to_rotmat_wxyz, quat_to_yaw, wrap_to_pi
from pid_tracker import PIDConfig, PositionPIDTracker

from omniperception_isaacdrone.envs.test6_env import WallSpawner, setup_global_obstacles


def get_step_dt(base_env: Any) -> float:
    if hasattr(base_env, "step_dt"):
        try:
            return float(base_env.step_dt)
        except Exception:
            pass
    try:
        return float(base_env.cfg.sim.dt) * float(getattr(base_env.cfg, "decimation", 1))
    except Exception:
        return 1.0 / 60.0


def freeze_obstacle_count_for_play(base_env: Any, obstacle_count: int) -> None:
    u = base_env.unwrapped if hasattr(base_env, "unwrapped") else base_env
    count = max(0, int(obstacle_count))
    u.curr_obstacle_count = count
    u.obstacle_level_changed = True

    if not hasattr(u, "curr_history"):
        device = torch.device(getattr(u, "device", "cpu"))
        num_envs = int(getattr(u, "num_envs", 1))
        u.curr_levels = [count]
        u.curr_num_envs = num_envs
        u.curr_k_roll = 1
        u.curr_window_size = num_envs
        u.curr_history = torch.full((num_envs,), -1.0, dtype=torch.float32, device=device)
        u.curr_write_round = torch.zeros(num_envs, dtype=torch.long, device=device)
        u.curr_level_idx = 0
        u.curr_device = device
    else:
        u.curr_levels = [count]
        u.curr_level_idx = 0
        u.curr_history.fill_(-1.0)

    print(f"[PLAY] 障碍物数量已固定为 {count}，课程学习晋级已禁用。", flush=True)


def maybe_override_goal(base_env: Any, goal_override: np.ndarray | None) -> np.ndarray:
    if goal_override is None:
        goal = base_env.goal_pos_w[0].detach().cpu().numpy().astype(np.float64)
        return goal

    goal_tensor = torch.as_tensor(goal_override, device=base_env.device, dtype=torch.float32).reshape(1, 3)
    base_env.goal_pos_w[:1] = goal_tensor
    if hasattr(base_env, "_update_goal_visualizers"):
        env_ids = torch.arange(1, device=base_env.device, dtype=torch.long)
        base_env._update_goal_visualizers(env_ids)
    return goal_override.astype(np.float64)


def get_robot_state(base_env: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot = base_env.scene["robot"]
    root = robot.data.root_link_state_w[0].detach().cpu().numpy().astype(np.float64)
    pos = root[0:3]
    quat = root[3:7]
    vel = root[7:10]
    return pos, quat, vel


def get_root_twist_action_scales(base_env: Any) -> tuple[float, float]:
    vel_scale = 1.0
    vel_clip = 1.0
    try:
        actions_cfg = getattr(base_env.cfg, "actions", None)
        root_twist = getattr(actions_cfg, "root_twist", None)
        params = getattr(root_twist, "params", None) or {}
        p_get = params.get if isinstance(params, dict) else lambda k, d=None: getattr(params, k, d)
        vel_scale = float(p_get("vel_scale", 1.0))
        vel_clip = float(p_get("vel_clip", vel_scale))
    except Exception:
        pass
    vel_scale = max(vel_scale, 1.0e-6)
    vel_clip = max(vel_clip, vel_scale)
    return vel_scale, vel_clip


def get_lidar_pose_env0(base_env: Any) -> tuple[np.ndarray, np.ndarray]:
    lidar = base_env.scene["lidar"]
    pos = lidar.data.pos_w[0].detach().cpu().numpy().astype(np.float64)
    quat = lidar.data.quat_w[0].detach().cpu().numpy().astype(np.float64)
    return pos, quat


def world_points_to_sensor_frame(
    points_world: np.ndarray,
    sensor_pos_w: np.ndarray,
    sensor_quat_wxyz: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return pts.reshape(0, 3)
    rel = pts - sensor_pos_w.reshape(1, 3)
    rot_w = quat_to_rotmat_wxyz(sensor_quat_wxyz)
    return rel @ rot_w


def sensor_points_to_world_frame(
    points_sensor: np.ndarray,
    sensor_pos_w: np.ndarray,
    sensor_quat_wxyz: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return pts.reshape(0, 3)
    rot_w = quat_to_rotmat_wxyz(sensor_quat_wxyz)
    return pts @ rot_w.T + sensor_pos_w.reshape(1, 3)


def get_pointcloud_sensor_env0(base_env: Any) -> tuple[np.ndarray, dict[str, Any]]:
    lidar = base_env.scene["lidar"]
    env_ids = torch.tensor([0], device=base_env.device, dtype=torch.long)
    pc = lidar.get_pointcloud(env_ids)
    if pc is None:
        return np.zeros((0, 3), dtype=np.float64), {}

    pc0 = pc[0] if pc.dim() == 3 else pc
    pc0 = pc0.detach().cpu().numpy().astype(np.float64)
    if pc0.ndim != 2 or pc0.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float64), {}

    sensor_pos_w, sensor_quat_wxyz = get_lidar_pose_env0(base_env)
    pointcloud_in_world_frame = bool(getattr(lidar.cfg, "pointcloud_in_world_frame", False))
    finite_mask = np.isfinite(pc0).all(axis=1)
    pc_finite = pc0[finite_mask]
    if pointcloud_in_world_frame:
        pc_sensor = world_points_to_sensor_frame(pc_finite, sensor_pos_w, sensor_quat_wxyz)
        frame_name = "world_to_sensor"
    else:
        pc_sensor = pc_finite
        frame_name = "sensor"

    return pc_sensor, {
        "pointcloud_in_world_frame": pointcloud_in_world_frame,
        "ray_alignment": str(getattr(lidar.cfg, "ray_alignment", "unknown")),
        "sensor_pos_w": sensor_pos_w,
        "sensor_quat_wxyz": sensor_quat_wxyz,
        "frame_name": frame_name,
    }


def evaluate_lidar_pointcloud_consistency_env0(base_env: Any) -> dict[str, float] | None:
    lidar = base_env.scene["lidar"]
    if not bool(getattr(lidar.cfg, "return_pointcloud", False)):
        return None
    if not hasattr(lidar.data, "ray_hits_w"):
        return None

    env_ids = torch.tensor([0], device=base_env.device, dtype=torch.long)
    pc = lidar.get_pointcloud(env_ids)
    if pc is None:
        return None

    pc0 = pc[0] if pc.dim() == 3 else pc
    hits0 = lidar.data.ray_hits_w[0]
    pc0 = pc0.detach().cpu().numpy().astype(np.float64)
    hits0 = hits0.detach().cpu().numpy().astype(np.float64)
    if pc0.ndim != 2 or pc0.shape[1] != 3 or hits0.ndim != 2 or hits0.shape[1] != 3:
        return None

    finite_mask = np.isfinite(pc0).all(axis=1) & np.isfinite(hits0).all(axis=1)
    if not np.any(finite_mask):
        return None

    pc_valid = pc0[finite_mask]
    hits_valid = hits0[finite_mask]
    if bool(getattr(lidar.cfg, "pointcloud_in_world_frame", False)):
        hits_reconstructed = pc_valid
    else:
        sensor_pos_w, sensor_quat_wxyz = get_lidar_pose_env0(base_env)
        hits_reconstructed = sensor_points_to_world_frame(pc_valid, sensor_pos_w, sensor_quat_wxyz)

    err = np.linalg.norm(hits_reconstructed - hits_valid, axis=1)
    return {
        "num_points": float(err.shape[0]),
        "mean_error": float(err.mean()),
        "max_error": float(err.max()),
    }


def compute_nearest_obstacle_distance(points_sensor: np.ndarray) -> float:
    pts = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return float("inf")
    dist = np.linalg.norm(pts, axis=1)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return float("inf")
    return float(dist.min())


def compute_forward_clearance(
    points_sensor: np.ndarray,
    cone_half_angle_deg: float = 40.0,
    z_limit: float = 1.8,
) -> float:
    pts = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return float("inf")
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.size == 0:
        return float("inf")
    dist = np.linalg.norm(pts, axis=1)
    valid = dist > 1.0e-6
    pts = pts[valid]
    dist = dist[valid]
    if dist.size == 0:
        return float("inf")
    cos_half_angle = float(np.cos(np.deg2rad(cone_half_angle_deg)))
    forward_cos = pts[:, 0] / np.maximum(dist, 1.0e-6)
    mask = (pts[:, 0] > 0.0) & (np.abs(pts[:, 2]) <= float(z_limit)) & (forward_cos >= cos_half_angle)
    if not np.any(mask):
        return float("inf")
    return float(dist[mask].min())


def compute_safety_speed_scale(
    nearest_obstacle_dist: float,
    forward_clearance: float,
    occ_ratio: float,
    emergency_brake: bool,
) -> float:
    scale = 1.0
    if np.isfinite(forward_clearance):
        if forward_clearance < 1.8:
            scale = min(scale, 0.18)
        elif forward_clearance < 2.8:
            scale = min(scale, 0.18 + 0.42 * (forward_clearance - 1.8))
        elif forward_clearance < 4.5:
            scale = min(scale, 0.60 + 0.25 * (forward_clearance - 2.8) / 1.7)
        elif forward_clearance < 6.5:
            scale = min(scale, 0.85 + 0.15 * (forward_clearance - 4.5) / 2.0)
    if np.isfinite(nearest_obstacle_dist):
        if nearest_obstacle_dist < 1.2:
            scale = min(scale, 0.10)
        elif nearest_obstacle_dist < 1.8:
            scale = min(scale, 0.10 + 0.30 * (nearest_obstacle_dist - 1.2) / 0.6)
        elif nearest_obstacle_dist < 2.5:
            scale = min(scale, 0.40 + 0.35 * (nearest_obstacle_dist - 1.8) / 0.7)
    if occ_ratio > 0.24:
        scale = min(scale, 0.90)
    elif occ_ratio > 0.18:
        scale = min(scale, 0.95)
    if emergency_brake:
        scale = min(scale, 0.20)
    return float(np.clip(scale, 0.0, 1.0))


def get_episode_end_reasons(base_env: Any) -> list[str]:
    reasons: list[str] = []
    tm = getattr(base_env, "termination_manager", None)
    if tm is not None:
        for name in ("time_out", "collision", "oob", "reached_goal"):
            try:
                flag = bool(tm.get_term(name).reshape(-1)[0].item())
            except Exception:
                flag = False
            if flag:
                reasons.append(name)
    if not reasons:
        try:
            if bool(base_env.reset_time_outs.reshape(-1)[0].item()):
                reasons.append("time_out")
        except Exception:
            pass
        try:
            if bool(base_env.reset_terminated.reshape(-1)[0].item()) and not reasons:
                reasons.append("terminated")
        except Exception:
            pass
    return reasons or ["unknown"]


def build_hover_action(current_yaw: float) -> np.ndarray:
    action = np.zeros(4, dtype=np.float32)
    action[3] = np.clip(wrap_to_pi(current_yaw) / np.pi, -1.0, 1.0)
    return action


def warmup_sensors(base_env: Any, num_steps: int, device: torch.device) -> None:
    zero_action = torch.zeros((1, 4), device=device, dtype=torch.float32)
    for _ in range(max(int(num_steps), 0)):
        base_env.step(zero_action)


def main() -> None:
    if int(args.num_envs) != 1:
        raise ValueError("test2_ego-planner 当前只支持 --num_envs 1。")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    goal_override = None
    if args.goal_x is not None and args.goal_y is not None and args.goal_z is not None:
        goal_override = np.array([args.goal_x, args.goal_y, args.goal_z], dtype=np.float64)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.scene.replicate_physics = True
    env_cfg.scene.filter_collisions = True
    try:
        setattr(env_cfg, "seed", int(args.seed))
    except Exception:
        pass
    try:
        root_twist_params = dict(getattr(env_cfg.actions.root_twist, "params", {}) or {})
        root_twist_params["vel_scale"] = float(args.action_vel_scale)
        root_twist_params["vel_clip"] = max(float(args.action_vel_clip), float(args.action_vel_scale))
        env_cfg.actions.root_twist.params = root_twist_params
    except Exception:
        pass

    print(
        f"[INFO] task={args.task}, num_envs={args.num_envs}, device={args.device}, "
        f"headless={args.headless}, steps={args.steps}",
        flush=True,
    )

    print("[INFO] Spawning workspace walls...", flush=True)
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

    print("[INFO] Setting up global obstacles template...", flush=True)
    setup_global_obstacles(int(args.num_obstacles))

    print("[INFO] Creating env...", flush=True)
    base_env = gym.make(args.task, cfg=env_cfg).unwrapped
    setattr(base_env, "_collision_print_enabled", True)
    device = torch.device(getattr(base_env, "device", args.device))
    dt = get_step_dt(base_env)

    freeze_obstacle_count_for_play(base_env, int(args.num_obstacles))

    occ_map = LocalOccupancyMap(
        resolution=float(args.local_map_resolution),
        x_range=(-float(args.local_map_x_half_range), float(args.local_map_x_half_range)),
        y_range=(-float(args.local_map_y_half_range), float(args.local_map_y_half_range)),
        z_range=(-float(args.local_map_z_down), float(args.local_map_z_up)),
        inflation_radius=float(args.inflation_radius),
    )
    planner = EgoStyleLocalPlanner(
        PlannerConfig(
            max_vel=float(args.planner_max_vel),
            max_acc=float(args.planner_max_acc),
            planning_horizon=float(args.planning_horizon),
            ctrl_pt_dist=float(args.ctrl_pt_dist),
            replan_period_s=float(args.replan_period),
            goal_tolerance=float(args.goal_tolerance),
        ),
        occ_map,
    )
    action_vel_scale, action_vel_clip = get_root_twist_action_scales(base_env)
    tracker = PositionPIDTracker(
        PIDConfig(
            action_vel_scale=action_vel_scale,
            max_speed_xy=min(max(action_vel_scale * 0.88, 3.8), action_vel_clip * 0.95),
            max_speed_z=min(1.0, action_vel_clip * 0.30),
        )
    )
    print(
        f"[INFO] root_twist action scaling: vel_scale={action_vel_scale:.2f}, vel_clip={action_vel_clip:.2f}",
        flush=True,
    )

    current_plan = None
    sim_time = 0.0
    episode_index = 0
    episode_step = 0
    previous_vel = np.zeros(3, dtype=np.float64)
    latest_stats = OccupancyStats(0, 0, 0.0, 0)
    lidar_frame_checked = False

    def reset_episode() -> np.ndarray:
        nonlocal current_plan, episode_index, episode_step, previous_vel, latest_stats, lidar_frame_checked
        base_env.reset()
        warmup_sensors(base_env, int(args.warmup_steps), device=device)
        tracker.reset()
        planner.reset()
        current_plan = None
        episode_step = 0
        episode_index += 1
        pos, _, vel = get_robot_state(base_env)
        previous_vel = vel.copy()
        latest_stats = OccupancyStats(0, 0, 0.0, 0)
        lidar_frame_checked = False
        goal = maybe_override_goal(base_env, goal_override)
        print(f"[EPISODE] episode={episode_index} start={pos} goal={goal}", flush=True)
        return goal

    goal_world = reset_episode()

    try:
        for global_step in range(int(args.steps)):
            if not simulation_app.is_running():
                break

            pos_w, quat_wxyz, vel_w = get_robot_state(base_env)
            yaw_w = quat_to_yaw(quat_wxyz)
            acc_w = (vel_w - previous_vel) / max(dt, 1.0e-6)
            previous_vel = vel_w.copy()

            if planner.goal_reached(pos_w, goal_world):
                hover = build_hover_action(yaw_w)
                action = torch.from_numpy(hover).to(device=device).unsqueeze(0)
                base_env.step(action)
                sim_time += dt
                if args.auto_reset:
                    goal_world = reset_episode()
                continue

            pointcloud_sensor, lidar_info = get_pointcloud_sensor_env0(base_env)
            sensor_quat_wxyz = np.asarray(
                lidar_info.get("sensor_quat_wxyz", quat_wxyz), dtype=np.float64
            ).reshape(4)
            latest_stats = occ_map.update_from_pointcloud_body(pointcloud_sensor, sensor_quat_wxyz)
            nearest_obstacle_dist = compute_nearest_obstacle_distance(pointcloud_sensor)
            forward_clearance = compute_forward_clearance(pointcloud_sensor)
            lin_speed = float(np.linalg.norm(vel_w))

            if not lidar_frame_checked and latest_stats.num_points_used > 0:
                lidar_check = evaluate_lidar_pointcloud_consistency_env0(base_env)
                if lidar_check is not None:
                    print(
                        "[LIDAR] "
                        f"frame={lidar_info.get('frame_name', 'unknown')} "
                        f"pointcloud_in_world_frame={lidar_info.get('pointcloud_in_world_frame', 'unknown')} "
                        f"ray_alignment={lidar_info.get('ray_alignment', 'unknown')} "
                        f"recon_mean_err={lidar_check['mean_error']:.5f}m "
                        f"recon_max_err={lidar_check['max_error']:.5f}m "
                        f"num={int(lidar_check['num_points'])}",
                        flush=True,
                    )
                lidar_frame_checked = True

            need_replan, replan_reason = planner.should_replan(
                sim_time=sim_time,
                current_pos=pos_w,
                current_quat=quat_wxyz,
                current_plan=current_plan,
            )
            if not need_replan and np.isfinite(forward_clearance) and forward_clearance < 3.8:
                need_replan, replan_reason = True, "obstacle_close"
            if not need_replan and np.isfinite(nearest_obstacle_dist) and nearest_obstacle_dist < 1.9:
                need_replan, replan_reason = True, "obstacle_too_close"
            replan_success = False
            emergency_brake = False
            if need_replan:
                new_plan = planner.plan(
                    sim_time=sim_time,
                    current_pos=pos_w,
                    current_vel=vel_w,
                    current_acc=acc_w,
                    current_quat=quat_wxyz,
                    goal_world=goal_world,
                    reason=replan_reason,
                )
                if new_plan is not None:
                    current_plan = new_plan
                    replan_success = True
                elif current_plan is None or replan_reason in {
                    "collision_predicted",
                    "bootstrap",
                    "plan_ending",
                    "obstacle_close",
                    "obstacle_too_close",
                }:
                    current_plan = None
                    emergency_brake = True

            emergency_brake = emergency_brake or (
                need_replan and replan_reason in {"collision_predicted", "obstacle_too_close"}
            )
            safety_speed_scale = compute_safety_speed_scale(
                nearest_obstacle_dist=nearest_obstacle_dist,
                forward_clearance=forward_clearance,
                occ_ratio=latest_stats.occupied_ratio,
                emergency_brake=emergency_brake,
            )

            if current_plan is None:
                hover = build_hover_action(yaw_w)
                pid_debug = {
                    "pos_error_norm": 0.0,
                    "vel_error_norm": 0.0,
                    "target_speed": 0.0,
                    "command_speed": 0.0,
                    "speed_scale": safety_speed_scale,
                    "emergency_brake": float(bool(emergency_brake)),
                    "desired_yaw_deg": np.degrees(yaw_w),
                    "xy_cmd_norm": 0.0,
                    "z_cmd": 0.0,
                    "lin_speed": lin_speed,
                }
                ref = None
                progress_t = 0.0
                target_t = 0.0
                action_np = hover
            else:
                ref, progress_t, target_t = current_plan.sample_tracking_reference(
                    sim_time=sim_time,
                    current_pos=pos_w,
                    lookahead_s=planner.cfg.tracking_lookahead_s,
                    window_back_s=planner.cfg.projection_window_back_s,
                    window_forward_s=planner.cfg.projection_window_forward_s,
                    sample_dt_s=planner.cfg.projection_dt_s,
                )
                action_np, pid_debug = tracker.compute_action(
                    dt=dt,
                    current_pos=pos_w,
                    current_vel=vel_w,
                    current_yaw=yaw_w,
                    ref_pos=ref.position,
                    ref_vel=ref.velocity,
                    ref_acc=ref.acceleration,
                    max_speed_scale=safety_speed_scale,
                    emergency_brake=emergency_brake,
                )
                pid_debug["lin_speed"] = lin_speed

            action = torch.from_numpy(action_np).to(device=device).unsqueeze(0)
            _, _, terminated, truncated, _ = base_env.step(action)

            sim_time += dt
            episode_step += 1

            done = bool(terminated.reshape(-1)[0].item()) or bool(truncated.reshape(-1)[0].item())

            if global_step % max(int(args.print_every), 1) == 0:
                goal_dist = float(np.linalg.norm(goal_world - pos_w))
                ref_pos = ref.position if ref is not None else pos_w
                ref_dist = float(np.linalg.norm(ref_pos - pos_w))
                plan_mode = current_plan.mode if current_plan is not None else "none"
                plan_remaining = current_plan.remaining_time(sim_time) if current_plan is not None else 0.0
                print(
                    "[PLAN] "
                    f"step={global_step:05d} episode_step={episode_step:04d} "
                    f"mode={plan_mode} replan={need_replan}:{replan_reason} "
                    f"goal_dist={goal_dist:.2f} ref_dist={ref_dist:.2f} "
                    f"remaining={plan_remaining:.2f}s "
                    f"track_t=({progress_t:.2f}->{target_t:.2f}) "
                    f"map_occ={latest_stats.inflated_voxels} "
                    f"occ_ratio={latest_stats.occupied_ratio:.4f} "
                    f"pc_used={latest_stats.num_points_used} "
                    f"nearest_obs={nearest_obstacle_dist:.2f} "
                    f"front_obs={forward_clearance:.2f} "
                    f"cmd=({float(action_np[0]):+.2f},{float(action_np[1]):+.2f},{float(action_np[2]):+.2f},{float(action_np[3]):+.2f}) "
                    f"lin_speed={pid_debug['lin_speed']:.2f} "
                    f"target_speed={pid_debug['target_speed']:.2f} "
                    f"cmd_speed={pid_debug['command_speed']:.2f} "
                    f"speed_scale={pid_debug['speed_scale']:.2f} "
                    f"replan_ok={int(replan_success)}",
                    flush=True,
                )

            if done:
                reasons = ",".join(get_episode_end_reasons(base_env))
                goal_dist = float(np.linalg.norm(goal_world - pos_w))
                print(
                    "[EPISODE] Environment signaled terminated/truncated, "
                    f"reasons={reasons}, goal_dist={goal_dist:.2f}, lin_speed={lin_speed:.2f}; resetting...",
                    flush=True,
                )
                goal_world = reset_episode()

    finally:
        try:
            base_env.close()
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()
