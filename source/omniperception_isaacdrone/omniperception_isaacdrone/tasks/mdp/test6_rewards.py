# omniperception_isaacdrone/tasks/mdp/test6_rewards.py

from __future__ import annotations

import math
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .test6_observations import obs_lidar_min_range_grid
from .test6_terminations import (
    termination_reached_goal,
    termination_out_of_workspace,
    termination_collision,
)


# -----------------------------------------------------------------------------
# TensorBoard / debug cache helpers (store per-step tensors on env)
# -----------------------------------------------------------------------------
def _tb_get_dict(env: ManagerBasedRLEnv, attr: str) -> dict:
    d = getattr(env, attr, None)
    if d is None or (not isinstance(d, dict)):
        d = {}
        setattr(env, attr, d)
    return d


def _tb_store_reward(env: ManagerBasedRLEnv, name: str, value: torch.Tensor):
    """Store raw reward term output (shape: [N]) for this step."""
    try:
        d = _tb_get_dict(env, "_tb_reward_terms")
        if isinstance(value, torch.Tensor):
            d[name] = value.detach()
    except Exception:
        pass


def _tb_store_aux(env: ManagerBasedRLEnv, name: str, value: torch.Tensor):
    """Store auxiliary debug metrics (shape: [N]) for this step."""
    try:
        d = _tb_get_dict(env, "_tb_aux_terms")
        if isinstance(value, torch.Tensor):
            d[name] = value.detach()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------
def _safe_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=-1) + eps)


def _broadcast_goal(goal: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Make sure goal has shape (N, 3) on the same device as pos."""
    if goal.device != pos.device:
        goal = goal.to(pos.device)

    if goal.shape[0] == 1 and pos.shape[0] > 1:
        goal = goal.expand(pos.shape[0], 3)
    elif goal.shape[0] != pos.shape[0]:
        goal = goal[:1].expand(pos.shape[0], 3)

    return goal


def _get_goal_pos(env: ManagerBasedRLEnv, pos: torch.Tensor) -> torch.Tensor:
    goal = getattr(env, "goal_pos_w", None)
    if goal is None:
        return torch.zeros_like(pos)
    return _broadcast_goal(goal, pos)


def _get_step_dt(env: ManagerBasedRLEnv) -> float:
    # IsaacLab usually provides env.step_dt; if not, fallback to sim.dt * decimation
    if hasattr(env, "step_dt"):
        try:
            return float(env.step_dt)
        except Exception:
            pass
    try:
        return float(env.cfg.sim.dt) * float(env.cfg.decimation)
    except Exception:
        # last resort: assume 60Hz RL step
        return 1.0 / 60.0


def _get_lidar_max_distance(lidar) -> float:
    try:
        if hasattr(lidar, "cfg") and hasattr(lidar.cfg, "max_distance"):
            return float(lidar.cfg.max_distance)
    except Exception:
        pass
    return 100.0


# -----------------------------------------------------------------------------
# ① goal distance reward (bounded & stable)
# -----------------------------------------------------------------------------
def reward_distance_to_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 6.0) -> torch.Tensor:
    """Bounded goal proximity reward: exp(-0.5*(d/std)^2) in (0, 1]."""
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)  # (N,3)
    goal = _get_goal_pos(env, pos)

    d = _safe_norm(goal - pos)  # (N,)
    std = max(float(std), 1e-6)

    r = d / std
    r2 = torch.clamp(r * r, 0.0, 400.0)
    out = torch.exp(-0.5 * r2)

    # cache
    _tb_store_reward(env, "dist_to_goal", out)
    _tb_store_aux(env, "goal_distance", d)

    return out


# -----------------------------------------------------------------------------
# (keep) height tracking
# -----------------------------------------------------------------------------
def reward_height_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_z: float = 5.0,
    std: float = 2.0,
) -> torch.Tensor:
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    dz = pos[:, 2] - float(target_z)
    std = max(float(std), 1e-6)
    r2 = torch.clamp((dz / std) ** 2, 0.0, 400.0)
    out = torch.exp(-0.5 * r2)

    _tb_store_reward(env, "height", out)
    _tb_store_aux(env, "height_error", dz)

    return out


# -----------------------------------------------------------------------------
# (keep) stability reward
# -----------------------------------------------------------------------------
def reward_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_std: float = 2.0,
    ang_std: float = 6.0,
) -> torch.Tensor:
    lin = mdp.base_lin_vel(env, asset_cfg=asset_cfg)
    ang = mdp.base_ang_vel(env, asset_cfg=asset_cfg)

    lin2 = torch.sum(lin * lin, dim=-1)
    ang2 = torch.sum(ang * ang, dim=-1)

    lin_std = max(float(lin_std), 1e-6)
    ang_std = max(float(ang_std), 1e-6)

    rl = torch.exp(-0.5 * torch.clamp(lin2 / (lin_std * lin_std), 0.0, 400.0))
    ra = torch.exp(-0.5 * torch.clamp(ang2 / (ang_std * ang_std), 0.0, 400.0))
    out = rl * ra

    _tb_store_reward(env, "stability", out)
    _tb_store_aux(env, "lin_speed", _safe_norm(lin))
    _tb_store_aux(env, "ang_speed", _safe_norm(ang))

    return out


# -----------------------------------------------------------------------------
# ④ velocity direction reward (towards goal)
# -----------------------------------------------------------------------------
def reward_velocity_towards_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_speed: float = 0.2,
    speed_ref: float = 3.0,
    use_relu: bool = True,
) -> torch.Tensor:
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    goal = _get_goal_pos(env, pos)

    v = mdp.root_lin_vel_w(env, asset_cfg=asset_cfg)  # (N,3)
    speed = _safe_norm(v)  # (N,)

    dir_vec = goal - pos
    dir_unit = dir_vec / (_safe_norm(dir_vec).unsqueeze(-1) + 1e-6)

    cos = torch.sum(v * dir_unit, dim=-1) / (speed + 1e-6)
    cos = torch.clamp(cos, -1.0, 1.0)

    if use_relu:
        cos = torch.clamp(cos, min=0.0)

    min_speed = float(min_speed)
    if min_speed > 0.0:
        cos = torch.where(speed >= min_speed, cos, torch.zeros_like(cos))

    speed_ref = float(speed_ref)
    if speed_ref > 1e-6:
        speed_fac = torch.clamp(speed / speed_ref, 0.0, 1.0)
        cos = cos * speed_fac

    _tb_store_reward(env, "vel_towards_goal", cos)
    _tb_store_aux(env, "speed", speed)

    return cos


# -----------------------------------------------------------------------------
# ② collision threat penalty from LiDAR (raw or grid)
# -----------------------------------------------------------------------------
def penalty_lidar_threat(
    env: ManagerBasedRLEnv,
    lidar_name: str = "lidar",
    threshold: float = 1.2,
    exp_scale: float = 0.25,
    cap: float = 5.0,
    use_grid: bool = False,
    # grid parameters (only used when use_grid=True)
    theta_min: float = 30.0,
    theta_max: float = 90.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 1.0,
    delta_phi: float = 5.0,
    max_vis_points: int | None = None,
) -> torch.Tensor:
    threshold = float(threshold)
    exp_scale = max(float(exp_scale), 1e-6)
    cap = float(cap)

    out0 = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)

    try:
        lidar = env.scene[lidar_name]
    except Exception:
        _tb_store_reward(env, "lidar_threat", out0)
        return out0

    env_ids = torch.arange(env.num_envs, device=env.device)
    max_d = _get_lidar_max_distance(lidar)

    if not bool(use_grid):
        dist = lidar.get_distances(env_ids)
        if dist is None:
            _tb_store_reward(env, "lidar_threat", out0)
            return out0
        if dist.dim() == 1:
            dist = dist.unsqueeze(0)
        dist = dist.to(dtype=torch.float32)

        dist = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, max_d))
        dist = torch.where(dist > 0.0, dist, torch.full_like(dist, max_d))
        min_dist = dist.min(dim=1).values  # (N,)
    else:
        grid = obs_lidar_min_range_grid(
            env,
            lidar_name=lidar_name,
            theta_min=theta_min,
            theta_max=theta_max,
            phi_min=phi_min,
            phi_max=phi_max,
            delta_theta=delta_theta,
            delta_phi=delta_phi,
            empty_value=max_d,
            max_vis_points=max_vis_points,
        )
        min_dist = grid.min(dim=1).values

    x = (threshold - min_dist) / exp_scale
    x = torch.clamp(x, min=0.0)

    if cap > 0.0:
        x_cap = math.log(cap + 1.0)
        x = torch.clamp(x, max=x_cap)

    pen = torch.expm1(x).to(torch.float32)
    if cap > 0.0:
        pen = torch.clamp(pen, 0.0, cap)

    _tb_store_reward(env, "lidar_threat", pen)
    _tb_store_aux(env, "lidar_min_dist", min_dist)

    return pen


# -----------------------------------------------------------------------------
# ③ energy penalty (v, w, a, alpha) with finite difference
# -----------------------------------------------------------------------------
def penalty_energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_vel_scale: float = 5.0,
    ang_vel_scale: float = 8.0,
    lin_acc_scale: float = 15.0,
    ang_acc_scale: float = 25.0,
    include_acc: bool = True,
    max_penalty: float = 10.0,
) -> torch.Tensor:
    lin_vel_scale = max(float(lin_vel_scale), 1e-6)
    ang_vel_scale = max(float(ang_vel_scale), 1e-6)
    lin_acc_scale = max(float(lin_acc_scale), 1e-6)
    ang_acc_scale = max(float(ang_acc_scale), 1e-6)
    max_penalty = float(max_penalty)

    v = mdp.base_lin_vel(env, asset_cfg=asset_cfg)
    w = mdp.base_ang_vel(env, asset_cfg=asset_cfg)

    v_norm = _safe_norm(v)
    w_norm = _safe_norm(w)

    pv = (v_norm / lin_vel_scale) ** 2
    pw = (w_norm / ang_vel_scale) ** 2

    pa = torch.zeros_like(pv)
    palpha = torch.zeros_like(pv)

    a_norm = torch.zeros_like(pv)
    alpha_norm = torch.zeros_like(pv)

    if bool(include_acc):
        dt = max(_get_step_dt(env), 1e-6)

        prev_v = getattr(env, "_energy_prev_lin_vel_w", None)
        prev_w = getattr(env, "_energy_prev_ang_vel_w", None)

        if (
            prev_v is None
            or prev_w is None
            or (not isinstance(prev_v, torch.Tensor))
            or (not isinstance(prev_w, torch.Tensor))
            or prev_v.shape != v.shape
            or prev_w.shape != w.shape
        ):
            setattr(env, "_energy_prev_lin_vel_w", v.detach().clone())
            setattr(env, "_energy_prev_ang_vel_w", w.detach().clone())
        else:
            a = (v - prev_v) / dt
            alpha = (w - prev_w) / dt

            a_norm = _safe_norm(a)
            alpha_norm = _safe_norm(alpha)

            pa = (a_norm / lin_acc_scale) ** 2
            palpha = (alpha_norm / ang_acc_scale) ** 2

            env._energy_prev_lin_vel_w = v.detach().clone()
            env._energy_prev_ang_vel_w = w.detach().clone()

    pen = pv + pw + pa + palpha
    if max_penalty > 0.0:
        pen = torch.clamp(pen, 0.0, max_penalty)

    _tb_store_reward(env, "energy", pen.to(torch.float32))
    _tb_store_aux(env, "energy_lin_speed", v_norm)
    _tb_store_aux(env, "energy_ang_speed", w_norm)
    _tb_store_aux(env, "energy_lin_acc", a_norm)
    _tb_store_aux(env, "energy_ang_acc", alpha_norm)

    return pen.to(torch.float32)


# -----------------------------------------------------------------------------
# ⑤ termination-related rewards / penalties (wrappers that output float)
# -----------------------------------------------------------------------------
def reward_goal_reached(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    out = termination_reached_goal(env, asset_cfg=asset_cfg, threshold=threshold).to(torch.float32)
    _tb_store_reward(env, "success_bonus", out)
    return out


def penalty_out_of_workspace(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    x_bounds: tuple[float, float] = (-60.0, 60.0),
    y_bounds: tuple[float, float] = (-60.0, 60.0),
    z_bounds: tuple[float, float] = (0.0, 10.0),
) -> torch.Tensor:
    out = termination_out_of_workspace(
        env, asset_cfg=asset_cfg, x_bounds=x_bounds, y_bounds=y_bounds, z_bounds=z_bounds
    ).to(torch.float32)
    _tb_store_reward(env, "oob_penalty", out)
    return out


def penalty_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    out = mdp.time_out(env).to(torch.float32)
    _tb_store_reward(env, "timeout_penalty", out)
    return out


def penalty_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    out = termination_collision(env, sensor_cfg=sensor_cfg, threshold=threshold).to(torch.float32)
    _tb_store_reward(env, "collision_penalty", out)
    return out


# -----------------------------------------------------------------------------
# (keep) action L2 (already used as penalty via negative weight)
# -----------------------------------------------------------------------------
def reward_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    term = env.action_manager.get_term("root_twist")
    a = term.raw_actions
    out = torch.sum(a * a, dim=-1)
    _tb_store_reward(env, "action_l2", out.to(torch.float32))
    return out
