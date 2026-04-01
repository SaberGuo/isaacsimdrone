# omniperception_isaacdrone/tasks/mdp/test6_rewards.py

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .test6_observations import obs_lidar_min_range_grid
from .test6_terminations import (
    termination_collision,
    termination_out_of_workspace,
    termination_reached_goal,
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


# -----------------------------------------------------------------------------
# ① goal distance reward (bounded & stable)
# -----------------------------------------------------------------------------
def reward_distance_to_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 6.0) -> torch.Tensor:
    """Bounded goal proximity reward: exp(-0.5*(d/std)^2) in (0, 1]."""
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


# -----------------------------------------------------------------------------
# dense progress reward (approach goal => positive, go away => negative)
# -----------------------------------------------------------------------------
def reward_progress_to_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    speed_ref: float = 3.0,
    clip: float = 1.0,
) -> torch.Tensor:
    """Dense shaping reward based on distance decrease to the goal.

    delta_d = prev_dist - current_dist
    towards_speed = delta_d / step_dt
    out = clamp(towards_speed / speed_ref, -clip, clip)
    """
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


# -----------------------------------------------------------------------------
# height tracking
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
# stability reward
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
    """Reward the velocity component projected onto the goal direction.

    Compared with the old cos(theta) * speed-factor form, this version directly uses
    the signed projected speed, so the magnitude is easier to interpret and matches
    the physical meaning of "moving towards the goal" more closely.
    """
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


# -----------------------------------------------------------------------------
# ② LiDAR safety penalty
# -----------------------------------------------------------------------------
def penalty_lidar_threat(
    env: ManagerBasedRLEnv,
    lidar_name: str = "lidar",
    safe_dist: float | None = None,
    safe_dist_ratio: float = 0.1,
    exp_scale: float = 1.0,
    cap: float = 5.0,
    use_grid: bool = True,
    threshold: float | None = None,
    theta_min: float = 30.0,
    theta_max: float = 90.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 1.0,
    delta_phi: float = 5.0,
    max_vis_points: int | None = None,
) -> torch.Tensor:
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

    if safe_dist is None and threshold is not None:
        safe_dist = float(threshold)
    if safe_dist is None:
        safe_dist = float(safe_dist_ratio) * float(max_d)
    safe_dist = float(safe_dist)

    if bool(use_grid):
        grid = obs_lidar_min_range_grid(
            env,
            lidar_name=lidar_name,
            theta_min=theta_min,
            theta_max=theta_max,
            phi_min=phi_min,
            phi_max=phi_max,
            delta_theta=delta_theta,
            delta_phi=delta_phi,
            empty_value=0.0,
            max_vis_points=max_vis_points,
            max_distance=max_d,
        )
        max_close = grid.max(dim=1).values
        min_dist = float(max_d) * (1.0 - max_close)
    else:
        dist = lidar.get_distances(env_ids)
        if dist is None:
            _tb_store_reward(env, "lidar_threat", out0)
            return out0
        if dist.dim() == 1:
            dist = dist.unsqueeze(0)
        dist = dist.to(dtype=torch.float32)
        dist = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, max_d))
        dist = torch.where(dist > 0.0, dist, torch.full_like(dist, max_d))
        min_dist = dist.min(dim=1).values

    delta = safe_dist - min_dist
    x = torch.clamp(delta / exp_scale, min=0.0)
    pen = torch.expm1(x).to(torch.float32)

    if cap > 0.0:
        pen = torch.clamp(pen, 0.0, cap)

    _tb_store_reward(env, "lidar_threat", pen)
    _tb_store_aux(env, "lidar_min_dist", min_dist)
    _tb_store_aux(env, "lidar_safe_dist", torch.full_like(min_dist, safe_dist))
    return pen


# -----------------------------------------------------------------------------
# ③ energy penalty
# -----------------------------------------------------------------------------
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
            alpha = dw / dt

            a_norm = _safe_norm(a)
            alpha_norm = _safe_norm(alpha)

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


# -----------------------------------------------------------------------------
# ⑤ termination-related rewards / penalties
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


# -----------------------------------------------------------------------------
# action L2 penalty
# -----------------------------------------------------------------------------
def reward_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    term = env.action_manager.get_term("root_twist")

    a = getattr(term, "processed_actions", None)
    if not isinstance(a, torch.Tensor):
        a = term.raw_actions

    out = torch.sum(a * a, dim=-1).to(torch.float32)
    _tb_store_reward(env, "action_l2", out)
    _tb_store_aux(env, "action_l2_raw", out)
    return out


# -----------------------------------------------------------------------------
# ⑥ Safe Velocity Penalty (NavRL 风格)
# -----------------------------------------------------------------------------
def penalty_safe_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lidar_name: str = "lidar",
    safe_dist: float = 8.0,      # 触发危险的距离阈值
    margin: float = 2.0,         # 安全裕度（寻找新方向时，要求距离 > safe_dist + margin）
    theta_min: float = 30.0,
    theta_max: float = 90.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 10.0,
    delta_phi: float = 5.0,
    max_vis_points: int | None = 12000,
) -> torch.Tensor:
    """
    当进入危险范围时触发，计算当前最优的“安全速度(safe_vel)”，
    并惩罚无人机当前速度与 safe_vel 之间的方向和大小差异。
    """
    # 1. 获取当前速度与方向
    v = mdp.root_lin_vel_w(env, asset_cfg=asset_cfg)
    v_norm = _safe_norm(v)
    v_dir = v / (v_norm.unsqueeze(-1) + 1e-6)

    # 获取雷达最大探测距离
    try:
        lidar = env.scene[lidar_name]
        max_d = _get_lidar_max_distance(lidar)
    except Exception:
        out0 = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        _tb_store_reward(env, "safe_vel_penalty", out0)
        return out0

    # 2. 获取雷达网格点云 closeness (值域 [0,1]，1表示紧贴，0表示在max_d之外)
    grid = obs_lidar_min_range_grid(
        env, lidar_name=lidar_name,
        theta_min=theta_min, theta_max=theta_max,
        phi_min=phi_min, phi_max=phi_max,
        delta_theta=delta_theta, delta_phi=delta_phi,
        empty_value=0.0, max_vis_points=max_vis_points, max_distance=max_d
    ) # shape: (N, num_bins)

    # 定义紧迫度阈值
    # closeness = 1.0 - dist / max_d => dist = max_d * (1 - closeness)
    closeness_threshold = 1.0 - (float(safe_dist) / max_d)
    safe_closeness = 1.0 - ((float(safe_dist) + float(margin)) / max_d)

    # 3. 触发条件：如果有网格点的紧迫度超过 closeness_threshold，说明进入了危险范围
    max_closeness = grid.max(dim=1).values
    threat_mask = max_closeness > closeness_threshold

    # 如果没有任何环境触发危险，直接返回 0
    if not threat_mask.any():
        out0 = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        _tb_store_reward(env, "safe_vel_penalty", out0)
        return out0

    # 4. 动态构建/获取预计算的网格方向向量 (num_bins, 3)
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

        # 球坐标转笛卡尔坐标 (以 Z 为天顶轴)
        sin_t = torch.sin(rad_theta)
        bin_x = sin_t * torch.cos(rad_phi)
        bin_y = sin_t * torch.sin(rad_phi)
        bin_z = torch.cos(rad_theta)
        bin_dirs = torch.stack([bin_x, bin_y, bin_z], dim=-1) # (num_bins, 3)
        setattr(env, cache_key, bin_dirs)

    # 5. 在网格中寻找最接近当前速度方向的“安全方向”
    # 筛选出安全的网格（带裕度）
    valid_mask = grid <= safe_closeness # shape: (N, num_bins)

    # 计算当前速度方向与所有网格方向的余弦相似度
    cos_sim = torch.einsum('ni,ji->nj', v_dir, bin_dirs) # shape: (N, num_bins)

    # 给不安全的网格打上极低的分数，确保选不到它们
    scored_bins = torch.where(valid_mask, cos_sim, torch.full_like(cos_sim, -2.0))

    # 取出每个环境中最优（最顺滑且安全）的网格索引
    best_scores, best_idx = torch.max(scored_bins, dim=1) # (N,)

    # 如果 best_scores <= -1.5，说明所有网格全是不安全的（被障碍物死死包围）
    has_safe_bin = best_scores > -1.5

    # 提取选出的安全方向
    chosen_safe_dirs = bin_dirs[best_idx] # (N, 3)

    # 6. 计算最终的 safe_vel
    # 如果有安全方向，我们希望它朝着那个方向以当前速度大小行驶
    # 如果处于绝境（没有安全方向），最好的策略是减速或反向倒车，这里设定为原方向取反
    safe_dir_final = torch.where(has_safe_bin.unsqueeze(-1), chosen_safe_dirs, -v_dir)
    safe_v_norm = torch.where(has_safe_bin, v_norm, torch.zeros_like(v_norm))
    # safe_vel = safe_dir_final * safe_v_norm.unsqueeze(-1) # 实际惩罚计算可以直接拆解为方向和大小，无需组装

    # 7. 计算奖励惩罚项
    # 方向惩罚: 1.0 - 余弦相似度 (值域 [0, 2]，0表示方向完全一致)
    v_safe_cos = torch.sum(v_dir * safe_dir_final, dim=-1)
    dir_penalty = 1.0 - v_safe_cos

    # 大小惩罚: |实际速度 - 安全速度| / 实际速度 (归一化，防止绝对值过大)
    mag_penalty = torch.abs(v_norm - safe_v_norm) / (v_norm + 1e-6)

    # 仅在触发危险的环境计算惩罚
    penalty = (dir_penalty + mag_penalty) * threat_mask.float()

    _tb_store_reward(env, "safe_vel_penalty", penalty)
    _tb_store_aux(env, "safe_vel_trigger_ratio", threat_mask.float())
    return penalty

