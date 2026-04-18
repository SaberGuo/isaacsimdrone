# omniperception_isaacdrone/tasks/mdp/test6_rewards.py
from __future__ import annotations

import math
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .test6_observations import get_lidar_grid_cached
from .test6_terminations import (
    termination_collision,
    termination_out_of_workspace,
    termination_reached_goal,
)


# =============================================================================
# 日志与调试辅助函数
# =============================================================================

def _tb_get_dict(env: ManagerBasedRLEnv, attr: str) -> dict:
    d = getattr(env, attr, None)
    if d is None or (not isinstance(d, dict)):
        d = {}
        setattr(env, attr, d)
    return d


def _tb_store_reward(env: ManagerBasedRLEnv, name: str, value: torch.Tensor):
    try:
        if not bool(getattr(env, "_enable_tb_reward_terms", True)):
            return
        d = _tb_get_dict(env, "_tb_reward_terms")
        if isinstance(value, torch.Tensor):
            d[name] = value.detach()
    except Exception:
        pass


def _tb_store_aux(env: ManagerBasedRLEnv, name: str, value: torch.Tensor):
    try:
        if not bool(getattr(env, "_enable_tb_aux_terms", False)):
            return
        d = _tb_get_dict(env, "_tb_aux_terms")
        if isinstance(value, torch.Tensor):
            d[name] = value.detach()
    except Exception:
        pass


# =============================================================================
# 基础计算与环境辅助函数
# =============================================================================

def _safe_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=-1) + eps)


def _broadcast_goal(goal: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
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
    if hasattr(env, "step_dt"):
        try:
            return float(env.step_dt)
        except Exception:
            pass
    try:
        return float(env.cfg.sim.dt) * float(env.cfg.decimation)
    except Exception:
        return 1.0 / 60.0


def _get_lidar_max_distance(lidar) -> float:
    try:
        if hasattr(lidar, "cfg") and hasattr(lidar.cfg, "max_distance"):
            return float(lidar.cfg.max_distance)
    except Exception:
        pass
    return 50.0


def _get_exp_closeness(d: float, max_d: float, alpha: float = 3.0) -> float:
    """将物理距离映射为指数型 closeness 值。"""
    norm_dist = min(max(d / max_d, 0.0), 1.0)
    exp_alpha = math.exp(-alpha)
    return (math.exp(-alpha * norm_dist) - exp_alpha) / (1.0 - exp_alpha)


# =============================================================================
# 任务目标与进度奖励
# =============================================================================

def reward_distance_to_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 6.0) -> torch.Tensor:
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    goal = _get_goal_pos(env, pos)

    d = _safe_norm(goal - pos)
    std = max(float(std), 1e-6)

    r = d / std
    r2 = torch.clamp(r * r, 0.0, 400.0)
    out = torch.exp(-0.5 * r2)

    _tb_store_reward(env, "dist_to_goal", out)
    _tb_store_aux(env, "goal_distance", d)
    return out


def reward_progress_to_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    speed_ref: float = 3.0,
    clip: float = 1.0,
) -> torch.Tensor:
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    goal = _get_goal_pos(env, pos)
    d = _safe_norm(goal - pos)

    prev = getattr(env, "_progress_prev_goal_dist", None)
    if prev is None or (not isinstance(prev, torch.Tensor)) or prev.shape != d.shape:
        setattr(env, "_progress_prev_goal_dist", d.detach().clone())
        towards_speed = torch.zeros_like(d)
        out = torch.zeros_like(d)
    else:
        dt = max(_get_step_dt(env), 1e-6)
        delta = prev - d
        towards_speed = delta / dt
        denom = max(float(speed_ref), 1e-6)
        out = torch.clamp(towards_speed / denom, min=-float(clip), max=float(clip))
        env._progress_prev_goal_dist = d.detach().clone()

    out = out.to(torch.float32)
    _tb_store_reward(env, "progress_to_goal", out)
    _tb_store_aux(env, "goal_progress_norm", out)
    _tb_store_aux(env, "goal_progress_speed", towards_speed.to(torch.float32))
    return out


def reward_velocity_towards_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_speed: float = 0.2,
    speed_ref: float = 3.0,
    use_relu: bool = True,
) -> torch.Tensor:
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    goal = _get_goal_pos(env, pos)

    v = mdp.root_lin_vel_w(env, asset_cfg=asset_cfg)
    speed = _safe_norm(v)

    dir_vec = goal - pos
    dir_norm = _safe_norm(dir_vec)
    dir_unit = dir_vec / (dir_norm.unsqueeze(-1) + 1e-6)

    projected_speed = torch.sum(v * dir_unit, dim=-1)
    cos = torch.clamp(projected_speed / (speed + 1e-6), -1.0, 1.0)

    if use_relu:
        projected_speed = torch.clamp(projected_speed, min=0.0)

    min_speed = float(min_speed)
    if min_speed > 0.0:
        projected_speed = torch.where(speed >= min_speed, projected_speed, torch.zeros_like(projected_speed))

    denom = max(float(speed_ref), 1e-6)
    if use_relu:
        out = torch.clamp(projected_speed / denom, 0.0, 1.0)
    else:
        out = torch.clamp(projected_speed / denom, -1.0, 1.0)

    out = out.to(torch.float32)
    _tb_store_reward(env, "vel_towards_goal", out)
    _tb_store_aux(env, "speed", speed)
    _tb_store_aux(env, "goal_direction_cos", cos)
    _tb_store_aux(env, "towards_speed", projected_speed.to(torch.float32))
    return out


# =============================================================================
# 飞行姿态与控制约束奖励
# =============================================================================

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


def reward_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    term = env.action_manager.get_term("root_twist")

    a = getattr(term, "processed_actions", None)
    if not isinstance(a, torch.Tensor):
        a = term.raw_actions

    out = torch.sum(a * a, dim=-1).to(torch.float32)
    _tb_store_reward(env, "action_l2", out)
    _tb_store_aux(env, "action_l2_raw", out)
    return out


# =============================================================================
# 避障与安全惩罚
# =============================================================================

def _compute_min_dist_from_lidar(
    env: ManagerBasedRLEnv,
    lidar,
    lidar_name: str,
    max_d: float,
    use_grid: bool,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    delta_theta: float,
    delta_phi: float,
    max_vis_points: int | None,
) -> torch.Tensor | None:
    """从 LiDAR 数据提取每个环境的最小障碍物距离，返回 (num_envs,) 张量。"""

    if bool(use_grid):
        grid = get_lidar_grid_cached(
            env, lidar_name=lidar_name,
            theta_min=theta_min, theta_max=theta_max,
            phi_min=phi_min, phi_max=phi_max,
            delta_theta=delta_theta, delta_phi=delta_phi,
            empty_value=0.0, max_vis_points=max_vis_points, max_distance=max_d,
        )
        max_close = grid.max(dim=1).values

        # closeness → 物理距离（解析反演）
        alpha = 3.0
        exp_alpha = math.exp(-alpha)
        val = max_close * (1.0 - exp_alpha) + exp_alpha
        val = torch.clamp(val, min=exp_alpha, max=1.0)
        min_dist = -(max_d / alpha) * torch.log(val)
        return min_dist
    else:
        env_ids = torch.arange(env.num_envs, device=env.device)
        dist = lidar.get_distances(env_ids)
        if dist is None:
            return None
        if dist.dim() == 1:
            dist = dist.unsqueeze(0)
        dist = dist.to(dtype=torch.float32)
        dist = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, max_d))
        dist = torch.where(dist > 0.0, dist, torch.full_like(dist, max_d))
        return dist.min(dim=1).values


def penalty_lidar_threat(
    env: ManagerBasedRLEnv,
    lidar_name: str = "lidar",
    safe_dist: float | None = None,
    safe_dist_ratio: float = 0.16,
    speed_ref: float = 6.0,
    clip: float = 1.0,
    proximity_boost: bool = True,
    use_grid: bool = True,
    theta_min: float = 30.0,
    theta_max: float = 90.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 1.0,
    delta_phi: float = 5.0,
    max_vis_points: int | None = None,
) -> torch.Tensor:
    """梯度型激光雷达威胁奖惩。

    在危险范围 (min_dist < safe_dist) 内：
      - 最小距离增大 (远离障碍物) → 返回负值 → ×负权重 = 正奖励 (鼓励)
      - 最小距离减小 (靠近障碍物) → 返回正值 → ×负权重 = 负惩罚 (惩罚)
      - 危险范围外 → 返回 0

    Args:
        safe_dist: 危险距离阈值 (m)。None 则由 safe_dist_ratio × max_distance 计算。
        safe_dist_ratio: safe_dist 为 None 时使用的比例。
        speed_ref: 归一化参考速度 (m/s)，用于将距离变化率映射到 [-1, 1]。
        clip: 输出裁剪范围 [-clip, clip]。
        proximity_boost: 是否按接近程度缩放信号 (越近信号越强)。
    """

    clip_val = max(float(clip), 1e-6)
    speed_ref_val = max(float(speed_ref), 1e-6)
    zeros = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)

    # ── 获取 lidar 传感器 ──
    try:
        lidar = env.scene[lidar_name]
    except Exception:
        _tb_store_reward(env, "lidar_threat", zeros)
        return zeros

    max_d = _get_lidar_max_distance(lidar)

    # ── 确定 safe_dist ──
    if safe_dist is None:
        safe_dist = float(safe_dist_ratio) * float(max_d)
    safe_dist = max(float(safe_dist), 1e-6)

    # ── 计算当前 min_dist ──
    min_dist = _compute_min_dist_from_lidar(
        env, lidar, lidar_name, max_d, use_grid,
        theta_min, theta_max, phi_min, phi_max,
        delta_theta, delta_phi, max_vis_points,
    )
    if min_dist is None:
        _tb_store_reward(env, "lidar_threat", zeros)
        return zeros

    # ── 获取上一步的 min_dist ──
    prev = getattr(env, "_lidar_threat_prev_min_dist", None)
    if prev is None or prev.shape[0] != env.num_envs:
        # 首次调用：初始化，本步不产生梯度信号
        env._lidar_threat_prev_min_dist = min_dist.detach().clone()
        _tb_store_reward(env, "lidar_threat", zeros)
        _tb_store_aux(env, "lidar_min_dist", min_dist)
        _tb_store_aux(env, "lidar_threat_approach_rate", zeros)
        _tb_store_aux(env, "lidar_threat_in_danger_ratio", (min_dist < safe_dist).float())
        return zeros

    # ── 处理刚 reset 的环境（抑制虚假梯度） ──
    reset_mask = getattr(env, "_lidar_threat_reset_mask", None)
    has_reset = reset_mask is not None and reset_mask.any()
    if has_reset:
        prev = prev.clone()
        prev[reset_mask] = min_dist[reset_mask].detach()
        env._lidar_threat_reset_mask[reset_mask] = False

    # ── 计算距离变化梯度 ──
    dt = max(_get_step_dt(env), 1e-6)
    delta = min_dist - prev  # 正 = 远离障碍物，负 = 靠近障碍物

    # 防止 reset 或传感器跳变导致的异常梯度
    max_delta = speed_ref_val * dt * 3.0
    delta = torch.clamp(delta, -max_delta, max_delta)

    # 接近速率：正值 = 正在靠近障碍物
    approach_rate = -delta / dt

    # 归一化到 [-1, 1] 左右
    normalized = approach_rate / speed_ref_val

    # 裁剪
    normalized = torch.clamp(normalized, -clip_val, clip_val)

    # ── 危险区域判定 & 接近度缩放 ──
    in_danger = min_dist < safe_dist  # (num_envs,) bool

    if proximity_boost:
        # 越接近障碍物信号越强，边界处平滑过渡到 0
        proximity_factor = torch.clamp(
            (safe_dist - min_dist) / safe_dist, 0.0, 1.0
        )
        out = normalized * proximity_factor
    else:
        out = normalized * in_danger.float()

    out = out.to(torch.float32)

    # ── 更新 prev 缓存 ──
    env._lidar_threat_prev_min_dist = min_dist.detach().clone()

    # ── TensorBoard 日志 ──
    _tb_store_reward(env, "lidar_threat", out)
    _tb_store_aux(env, "lidar_min_dist", min_dist)
    _tb_store_aux(env, "lidar_safe_dist", torch.full_like(min_dist, safe_dist))
    _tb_store_aux(env, "lidar_threat_approach_rate", approach_rate)
    _tb_store_aux(env, "lidar_threat_in_danger_ratio", in_danger.float())
    return out


def penalty_safe_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lidar_name: str = "lidar",
    safe_dist: float = 8.0,
    margin: float = 2.0,
    theta_min: float = 30.0,
    theta_max: float = 90.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 10.0,
    delta_phi: float = 5.0,
    max_vis_points: int | None = 12000,
) -> torch.Tensor:
    """动态安全速度惩罚 (NavRL 风格)。"""

    v = mdp.base_lin_vel(env, asset_cfg=asset_cfg)
    v_norm = _safe_norm(v)
    v_dir = v / (v_norm.unsqueeze(-1) + 1e-6)

    try:
        lidar = env.scene[lidar_name]
        max_d = _get_lidar_max_distance(lidar)
    except Exception:
        out0 = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        _tb_store_reward(env, "safe_vel_penalty", out0)
        return out0

    # ── 通过缓存接口获取网格 ──
    grid = get_lidar_grid_cached(
        env, lidar_name=lidar_name,
        theta_min=theta_min, theta_max=theta_max,
        phi_min=phi_min, phi_max=phi_max,
        delta_theta=delta_theta, delta_phi=delta_phi,
        empty_value=0.0, max_vis_points=max_vis_points, max_distance=max_d,
    )

    alpha = 3.0
    closeness_threshold = _get_exp_closeness(float(safe_dist), max_d, alpha)
    safe_closeness = _get_exp_closeness(float(safe_dist) + float(margin), max_d, alpha)

    max_closeness = grid.max(dim=1).values
    threat_mask = max_closeness > closeness_threshold

    if not threat_mask.any():
        out0 = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        _tb_store_reward(env, "safe_vel_penalty", out0)
        return out0

    cache_key = f"_safe_vel_bins_{theta_min}_{theta_max}_{phi_min}_{phi_max}_{delta_theta}_{delta_phi}"
    bin_dirs = getattr(env, cache_key, None)
    if bin_dirs is None:
        T = max(int((theta_max - theta_min) / delta_theta), 1)
        Pn = max(int((phi_max - phi_min) / delta_phi), 1)

        theta_idx = torch.arange(T, device=env.device, dtype=torch.float32)
        phi_idx = torch.arange(Pn, device=env.device, dtype=torch.float32)
        theta_centers = theta_min + (theta_idx + 0.5) * delta_theta
        phi_centers = phi_min + (phi_idx + 0.5) * delta_phi

        grid_theta, grid_phi = torch.meshgrid(theta_centers, phi_centers, indexing='ij')
        rad_theta = torch.deg2rad(grid_theta.flatten())
        rad_phi = torch.deg2rad(grid_phi.flatten())

        sin_t = torch.sin(rad_theta)
        bin_x = sin_t * torch.cos(rad_phi)
        bin_y = sin_t * torch.sin(rad_phi)
        bin_z = torch.cos(rad_theta)
        bin_dirs = torch.stack([bin_x, bin_y, bin_z], dim=-1)
        setattr(env, cache_key, bin_dirs)

    cos_sim = torch.einsum('ni,ji->nj', v_dir, bin_dirs)

    current_heading_bin = torch.argmax(cos_sim, dim=1)
    current_heading_closeness = grid.gather(1, current_heading_bin.unsqueeze(1)).squeeze(1)

    is_heading_safe = current_heading_closeness <= safe_closeness
    active_mask = threat_mask & ~is_heading_safe

    if not active_mask.any():
        out0 = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        _tb_store_reward(env, "safe_vel_penalty", out0)
        return out0

    valid_mask = grid <= safe_closeness

    scored_bins = torch.where(valid_mask, cos_sim, torch.full_like(cos_sim, -2.0))
    best_scores, best_idx = torch.max(scored_bins, dim=1)

    has_safe_bin = best_scores > -1.5
    chosen_safe_dirs = bin_dirs[best_idx]

    safe_dir_final = torch.where(has_safe_bin.unsqueeze(-1), chosen_safe_dirs, -v_dir)
    safe_v_norm = torch.where(has_safe_bin, v_norm, torch.zeros_like(v_norm))

    v_safe_cos = torch.sum(v_dir * safe_dir_final, dim=-1)
    dir_penalty = 1.0 - v_safe_cos
    mag_penalty = torch.abs(v_norm - safe_v_norm) / (v_norm + 1e-6)

    penalty = (dir_penalty + mag_penalty) * active_mask.float()

    _tb_store_reward(env, "safe_vel_penalty", penalty)
    _tb_store_aux(env, "safe_vel_trigger_ratio", active_mask.float())
    return penalty


def penalty_energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_vel_scale: float = 6.0,
    ang_vel_scale: float = 10.0,
    lin_acc_scale: float = 50.0,
    ang_acc_scale: float = 80.0,
    include_acc: bool = True,
    acc_weight: float = 0.2,
    max_penalty: float = 10.0,
) -> torch.Tensor:
    lin_vel_scale = max(float(lin_vel_scale), 1e-6)
    ang_vel_scale = max(float(ang_vel_scale), 1e-6)
    lin_acc_scale = max(float(lin_acc_scale), 1e-6)
    ang_acc_scale = max(float(ang_acc_scale), 1e-6)
    acc_weight = max(float(acc_weight), 0.0)
    max_penalty = float(max_penalty)

    v = mdp.root_lin_vel_w(env, asset_cfg=asset_cfg)
    w = mdp.root_ang_vel_w(env, asset_cfg=asset_cfg)

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
            dv = torch.clamp(v - prev_v, min=-10.0 * dt, max=10.0 * dt)
            dw = torch.clamp(w - prev_w, min=-20.0 * dt, max=20.0 * dt)

            a = dv / dt
            alpha_dw = dw / dt

            a_norm = _safe_norm(a)
            alpha_norm = _safe_norm(alpha_dw)

            pa = (a_norm / lin_acc_scale) ** 2
            palpha = (alpha_norm / ang_acc_scale) ** 2

            env._energy_prev_lin_vel_w = v.detach().clone()
            env._energy_prev_ang_vel_w = w.detach().clone()

    pen = pv + pw + acc_weight * (pa + palpha)

    if max_penalty > 0.0:
        pen = torch.clamp(pen, 0.0, max_penalty)

    pen = pen.to(torch.float32)
    _tb_store_reward(env, "energy", pen)
    _tb_store_aux(env, "energy_lin_speed", v_norm)
    _tb_store_aux(env, "energy_ang_speed", w_norm)
    _tb_store_aux(env, "energy_lin_acc", a_norm)
    _tb_store_aux(env, "energy_ang_acc", alpha_norm)
    return pen


# =============================================================================
# 终止条件相关奖惩
# =============================================================================

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
        env,
        asset_cfg=asset_cfg,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
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
