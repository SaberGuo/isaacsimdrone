# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# =============================================================================
# 0) 标准库 & 与 Kit 无关的库
# =============================================================================
import argparse
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import matplotlib.font_manager as fm
# 方案1：使用系统中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 
                                    'DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def parse_args():
    parser = argparse.ArgumentParser(description="ManagerBasedRLEnv Drone (single-file) + minimal training")

    # env basics
    parser.add_argument("--num_envs", type=int, default=4, help="环境数量")
    parser.add_argument("--num_obstacles", type=int, default=50, help="障碍物数量（共享）")
    parser.add_argument("--max_steps", type=int, default=2000, help="非训练模式下最多仿真步数")

    # task ranges
    parser.add_argument("--square_half_size", type=float, default=35.0, help="初始化/目标点采样的半边长")
    parser.add_argument("--z_init_min", type=float, default=3.0)
    parser.add_argument("--z_init_max", type=float, default=7.0)
    parser.add_argument("--goal_z_min", type=float, default=3.0)
    parser.add_argument("--goal_z_max", type=float, default=8.0)

    parser.add_argument("--lin_vel_scale", type=float, default=1.0, help="线速度动作缩放 (m/s)")
    parser.add_argument("--ang_vel_scale", type=float, default=1.0, help="角速度动作缩放 (rad/s)")
    parser.add_argument("--lin_vel_clip", type=float, default=5.0, help="线速度裁剪 (m/s)")
    parser.add_argument("--ang_vel_clip", type=float, default=6.0, help="角速度裁剪 (rad/s)")

    # termination bounds
    parser.add_argument("--min_height", type=float, default=0.25, help="低于该高度判定坠地/终止")
    parser.add_argument("--world_bound", type=float, default=40.0, help="飞出该边界判定终止（|x| or |y|）")

    # action scaling / hover bias
    parser.add_argument("--action_scale", type=float, default=1.0, help="动作缩放（关节 effort）")
    parser.add_argument("--hover_effort_bias", type=float, default=0.0,
                        help="对 4 个电机 effort 增加一个常量偏置（用于更容易起飞/悬停，视资产而定）")

    # optional lidar debug
    parser.add_argument("--enable_lidar", action="store_true", help="启用 LiDAR（在本文件里挂最小 CFG）")
    parser.add_argument("--lidar_print_every", type=int, default=50)
    parser.add_argument("--lidar_max_vis_points", type=int, default=10000)
    parser.add_argument("--lidar_save_every", type=int, default=50)
    parser.add_argument("--lidar_save_max", type=int, default=20)
    parser.add_argument("--lidar_point_size", type=float, default=0.15)
    parser.add_argument("--lidar_save_dir", type=str, default=str(Path.home() / "lidar_pc_images"))

    # lidar polar grid obs params
    parser.add_argument("--lidar_theta_min", type=float, default=30.0)
    parser.add_argument("--lidar_theta_max", type=float, default=90.0)
    parser.add_argument("--lidar_phi_min", type=float, default=0.0)
    parser.add_argument("--lidar_phi_max", type=float, default=360.0)
    parser.add_argument("--lidar_delta_theta", type=float, default=1.0)
    parser.add_argument("--lidar_delta_phi", type=float, default=5.0)
    parser.add_argument("--lidar_empty_value", type=float, default=0.0, help="无点命中时 bin 的填充值（0 或 max_range）")

    # ── 新增：坐标系诊断模式 ──────────────────────────────────────────────────
    parser.add_argument(
        "--diagnose_lidar_frame",
        action="store_true",
        help=(
            "启用 LiDAR 坐标系诊断：\n"
            "  阶段1 无人机保持初始朝向，记录障碍物方位角分布；\n"
            "  阶段2 无人机旋转 90° 偏航，再次记录方位角分布；\n"
            "  若两次分布一致 → 世界系；若偏移 ~90° → 机体系。\n"
            "  结果保存为 PNG 到 --lidar_save_dir。\n"
            "  需要同时指定 --enable_lidar。"
        ),
    )
    parser.add_argument(
        "--diagnose_phase1_steps", type=int, default=30,
        help="诊断阶段1（初始朝向稳定）持续步数",
    )
    parser.add_argument(
        "--diagnose_rotate_steps", type=int, default=60,
        help="诊断旋转阶段（施加偏航角速度）持续步数",
    )
    parser.add_argument(
        "--diagnose_phase2_steps", type=int, default=30,
        help="诊断阶段2（旋转完成后稳定）持续步数",
    )
    parser.add_argument(
        "--diagnose_yaw_rate", type=float, default=math.pi / 2.0,
        help="诊断旋转阶段的偏航角速度 (rad/s)，默认 π/2",
    )

    # training flags
    parser.add_argument("--train", action="store_true", help="开启最小训练流程（A2C风格）")
    parser.add_argument("--train_iters", type=int, default=2000, help="训练迭代次数")
    parser.add_argument("--rollout_len", type=int, default=32, help="每次迭代 rollout 步长")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)

    # Isaac Lab app args（必须在这里加）
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


def save_pointcloud_png(points_np, save_path, title=None, s=1):
    """points_np: (N,3) numpy array"""
    if points_np is None or len(points_np) == 0:
        return False

    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=s)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    if title:
        ax.set_title(title)

    mn = points_np.min(axis=0)
    mx = points_np.max(axis=0)
    c = (mn + mx) / 2.0
    r = (mx - mn).max() / 2.0
    if r < 1e-6:
        r = 1.0

    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=25, azim=45)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


# =============================================================================
# 新增：LiDAR 坐标系诊断工具函数
# =============================================================================

def _compute_phi_distribution(pc_np):
    """
    计算点云在 XY 平面上的方位角分布（phi），用于坐标系诊断。

    参数
    ----
    pc_np : np.ndarray, shape (N, 3)
        点云坐标（相对传感器原点）

    返回
    ----
    phi_deg : np.ndarray, shape (M,)
        有效点的方位角（度），范围 [0, 360)
    phi_mean : float
        加权平均方位角（circular mean，度）
    phi_std : float
        方位角圆形标准差估计（度）
    """
    import numpy as np

    if pc_np is None or len(pc_np) == 0:
        return None, float("nan"), float("nan")

    x = pc_np[:, 0]
    y = pc_np[:, 1]

    # 只保留 XY 平面有效点（r_xy > 0.1m，排除正上/正下方的点）
    r_xy = np.sqrt(x ** 2 + y ** 2)
    mask = r_xy > 0.1
    if mask.sum() == 0:
        return None, float("nan"), float("nan")

    x = x[mask]
    y = y[mask]

    phi_rad = np.arctan2(y, x)                    # [-π, π]
    phi_deg = np.degrees(phi_rad) % 360.0         # [0, 360)

    # Circular mean：避免 0°/360° 边界问题
    sin_mean = np.mean(np.sin(phi_rad))
    cos_mean = np.mean(np.cos(phi_rad))
    circular_mean_deg = math.degrees(math.atan2(sin_mean, cos_mean)) % 360.0

    # Circular std 估计：R̄ = sqrt(sin_mean²+cos_mean²)，std ≈ sqrt(-2*ln(R̄))
    R_bar = math.sqrt(sin_mean ** 2 + cos_mean ** 2)
    R_bar = min(R_bar, 1.0 - 1e-9)
    circular_std_deg = math.degrees(math.sqrt(-2.0 * math.log(R_bar + 1e-9)))

    return phi_deg, circular_mean_deg, circular_std_deg


def _circular_diff_deg(a: float, b: float) -> float:
    """
    计算两个角度（度）之间的有向差 a - b，结果在 (-180, 180]。
    """
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def save_phi_histogram_png(
    phi1_deg,
    phi2_deg,
    save_path: str,
    label1: str = "阶段1（初始朝向）",
    label2: str = "阶段2（旋转后）",
    mean1: float = None,
    mean2: float = None,
    conclusion: str = "",
):
    """
    将两阶段的方位角直方图保存为 PNG，用于人工核查。

    参数
    ----
    phi1_deg, phi2_deg : np.ndarray
        两阶段的方位角数组（度）
    save_path : str
        输出文件路径
    label1, label2 : str
        图例标签
    mean1, mean2 : float
        两阶段圆形均值（度），用于标注竖线
    conclusion : str
        诊断结论文字，显示在标题
    """
    import numpy as np

    bins = np.linspace(0, 360, 73)   # 5° 一格，共 72 格

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle(f"LiDAR 坐标系诊断\n{conclusion}", fontsize=12, y=1.02)

    colors = ["steelblue", "tomato"]
    data_list = [phi1_deg, phi2_deg]
    labels = [label1, label2]
    means = [mean1, mean2]

    for ax, data, label, color, mean in zip(axes, data_list, labels, colors, means):
        if data is not None and len(data) > 0:
            ax.hist(data, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
            if mean is not None and math.isfinite(mean):
                ax.axvline(mean, color="black", linewidth=2.0, linestyle="--",
                           label=f"circular mean={mean:.1f}°")
                ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("方位角 phi (°)", fontsize=9)
        ax.set_ylabel("点数", fontsize=9)
        ax.set_xlim(0, 360)
        ax.set_xticks(range(0, 361, 45))
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[DIAGNOSE] 方位角直方图已保存: {save_path}")


def _get_downsampled_pc_torch(env, lidar, env_ids: torch.Tensor, max_pts: int | None):
    """
    Torch版：读取 pointcloud，做与 _get_downsampled_pc_np 同等的格式规整与过滤（不转numpy）。
    返回:
      pc: (E, P, 3) torch tensor on env.device
      num_raw: (E,) 每个env原始点数（int tensor）
    """
    if lidar is None:
        return None, None

    pc = lidar.get_pointcloud(env_ids)
    if pc is None:
        return None, None

    if pc.dim() == 2:
        pc = pc.unsqueeze(0)

    E, P, _ = pc.shape
    num_raw = torch.full((E,), P, device=pc.device, dtype=torch.int32)

    finite_mask = torch.isfinite(pc).all(dim=-1)
    pc = pc.clone()
    pc[~finite_mask] = float("nan")

    return pc, num_raw


def _get_downsampled_pc_np(env, lidar, env0_ids: torch.Tensor, max_pts: int):
    """
    保持你当前逻辑语义：
    - get_pointcloud
    - 处理 (1,N,3)/(N,3)
    - (可选)下采样（你现在注释掉）
    - 转 numpy
    - 过滤 NaN/Inf
    """
    if lidar is None:
        return None, 0
    try:
        pc_t, num_raw_t = _get_downsampled_pc_torch(env, lidar, env0_ids, max_pts=max_pts)
        if pc_t is None:
            return None, 0

        pc0 = pc_t[0]
        num_raw = int(num_raw_t[0].item()) if num_raw_t is not None else int(pc0.shape[0])

        points_np = pc0.detach().cpu().numpy()
        import numpy as np
        points_np = points_np[np.isfinite(points_np).all(axis=1)]
        if points_np.shape[0] == 0:
            return None, num_raw
        return points_np, num_raw
    except Exception:
        return None, 0


# =============================================================================
# 核心诊断函数
# =============================================================================

def run_lidar_frame_diagnosis(
    env,
    lidar,
    robot_asset,
    save_dir: Path,
    phase1_steps: int = 30,
    rotate_steps: int = 60,
    phase2_steps: int = 30,
    yaw_rate_rad: float = math.pi / 2.0,
    max_pts: int = 10000,
):
    """
    通过控制无人机偏航旋转，判断 LiDAR 点云是世界坐标系还是机体坐标系。

    原理
    ----
    - 世界坐标系（pointcloud_in_world_frame=True）：
        点云方向由世界系固定 XY 轴决定，机体旋转不影响点云 phi 分布。
        → 旋转前后，相同障碍物的 phi 值不变，圆形均值差 ≈ 0°。

    - 机体坐标系（pointcloud_in_world_frame=False）：
        点云方向随机体朝向旋转，偏航 Δyaw 后，所有点的 phi 值
        整体偏移约 -Δyaw（点云随机体转动）。
        → 圆形均值差 ≈ -Δyaw（或 +Δyaw，取决于旋转方向约定）。

    实验设计
    --------
    阶段0 (phase1_steps)：零动作，无人机悬停，记录初始偏航角 yaw0。
    阶段1 (rotate_steps)：施加纯偏航角速度，旋转约 Δyaw = yaw_rate * dt * rotate_steps。
    阶段2 (phase2_steps)：零动作，悬停，记录旋转后偏航角 yaw1，采集点云。

    判断标准
    --------
    实际偏航变化量 Δyaw_actual（由四元数计算）
    点云方位角圆形均值变化量 Δphi

    若 |Δphi| < 20°  → 世界坐标系（点云不随机体转）
    若 ||Δphi| - |Δyaw_actual|| < 30°  → 机体坐标系（点云随机体转）
    否则 → 不确定（可能场景障碍物分布偏置、传感器延迟等原因）

    参数
    ----
    env           : ManagerBasedRLEnv 实例
    lidar         : LidarSensor 实例
    robot_asset   : 无人机 Articulation 实例
    save_dir      : Path，诊断图像保存目录
    phase1_steps  : 阶段0 稳定步数
    rotate_steps  : 旋转步数
    phase2_steps  : 阶段2 稳定步数
    yaw_rate_rad  : 旋转阶段偏航角速度 (rad/s)
    max_pts       : 点云最大采样点数

    返回
    ----
    result : dict，包含：
        "frame"      : "world" | "body" | "unknown"
        "delta_phi"  : float，点云方位角均值变化量（度）
        "delta_yaw"  : float，实际偏航变化量（度）
        "phi_mean1"  : float，阶段0 方位角均值（度）
        "phi_mean2"  : float，阶段2 方位角均值（度）
        "conclusion" : str，诊断结论文字
    """
    import numpy as np

    print("\n" + "=" * 70)
    print("[DIAGNOSE] 开始 LiDAR 坐标系诊断")
    print(f"  阶段0（稳定）: {phase1_steps} 步")
    print(f"  旋转阶段:     {rotate_steps} 步，偏航速度={math.degrees(yaw_rate_rad):.1f}°/s")
    print(f"  阶段2（稳定）: {phase2_steps} 步")
    print("=" * 70)

    env0_ids = torch.tensor([0], device=env.device)

    # ── 辅助：从四元数提取偏航角（世界系，弧度）──────────────────────────────
    def _get_yaw_rad(asset) -> float:
        """从 root_quat_w (wxyz) 提取偏航角（弧度）"""
        try:
            q = asset.data.root_quat_w[0].detach().cpu()   # (4,) wxyz
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            # yaw = atan2(2*(w*z + x*y), 1 - 2*(y²+z²))
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            return yaw
        except Exception:
            return 0.0

    # ── 辅助：构建动作张量 ────────────────────────────────────────────────────
    def _make_action(lin_vel=(0., 0., 0.), ang_vel=(0., 0., 0.)):
        """构建 (num_envs, 6) 的动作张量（世界系，直接写 root velocity）"""
        a = torch.zeros(env.num_envs, 6, device=env.device)
        a[:, 0] = lin_vel[0]
        a[:, 1] = lin_vel[1]
        a[:, 2] = lin_vel[2]
        a[:, 3] = ang_vel[0]
        a[:, 4] = ang_vel[1]
        a[:, 5] = ang_vel[2]
        return a

    # ── 辅助：采集点云并计算 phi 分布 ────────────────────────────────────────
    def _collect_phi(n_steps: int, action_fn):
        """
        运行 n_steps 步，每步执行 action_fn() 返回的动作，
        最后一步采集点云，计算 phi 分布。
        返回 (phi_deg_array, phi_mean, phi_std, yaw_rad_after)
        """
        phi_deg = None
        phi_mean = float("nan")
        phi_std = float("nan")

        for step_i in range(n_steps):
            action = action_fn()
            with torch.inference_mode():
                env.step(action)

        # 采集最后一步的点云（退出 inference_mode 后才能操作 numpy）
        yaw_after = _get_yaw_rad(robot_asset)
        points_np, _ = _get_downsampled_pc_np(env, lidar, env0_ids, max_pts=max_pts)

        if points_np is not None and len(points_np) > 0:
            phi_deg, phi_mean, phi_std = _compute_phi_distribution(points_np)
            print(f"    点云有效点数: {len(points_np)}，"
                  f"phi 均值: {phi_mean:.1f}°，phi std: {phi_std:.1f}°，"
                  f"偏航: {math.degrees(yaw_after):.1f}°")
        else:
            print("    [WARN] 未获取到有效点云！")

        return phi_deg, phi_mean, phi_std, yaw_after

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 阶段0：保持零动作，悬停，记录初始偏航 & 点云 phi 分布
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[DIAGNOSE] 阶段0：悬停稳定，采集初始方位角分布...")
    phi1_deg, phi_mean1, phi_std1, yaw0 = _collect_phi(
        n_steps=phase1_steps,
        action_fn=lambda: _make_action(),   # 零动作
    )
    print(f"  初始偏航 yaw0 = {math.degrees(yaw0):.2f}°")
    print(f"  初始点云 phi_mean1 = {phi_mean1:.2f}°，phi_std1 = {phi_std1:.2f}°")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 旋转阶段：施加纯偏航角速度（世界系 wz）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n[DIAGNOSE] 旋转阶段：施加偏航速度 {math.degrees(yaw_rate_rad):.1f}°/s，共 {rotate_steps} 步...")

    # 注意：RootTwistVelocityActionTerm.apply_actions() 调用
    # write_root_velocity_to_sim，其 ang_vel 是世界系下的角速度
    # wz > 0 → 绕世界 Z+ 轴逆时针旋转（右手系）
    for step_i in range(rotate_steps):
        action = _make_action(ang_vel=(0., 0., yaw_rate_rad))
        with torch.inference_mode():
            env.step(action)
        if (step_i + 1) % 20 == 0:
            yaw_cur = _get_yaw_rad(robot_asset)
            print(f"    旋转步 {step_i+1}/{rotate_steps}，当前偏航: {math.degrees(yaw_cur):.1f}°")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 阶段2：停止旋转，悬停稳定，采集旋转后点云 phi 分布
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n[DIAGNOSE] 阶段2：停止旋转，悬停稳定，采集旋转后方位角分布...")
    phi2_deg, phi_mean2, phi_std2, yaw1 = _collect_phi(
        n_steps=phase2_steps,
        action_fn=lambda: _make_action(),   # 零动作
    )
    print(f"  旋转后偏航 yaw1 = {math.degrees(yaw1):.2f}°")
    print(f"  旋转后点云 phi_mean2 = {phi_mean2:.2f}°，phi_std2 = {phi_std2:.2f}°")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 判断逻辑
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 实际偏航变化量（度，有向）
    delta_yaw_rad = _circular_diff_deg(math.degrees(yaw1), math.degrees(yaw0))
    delta_yaw_deg = delta_yaw_rad   # 已经是度
    # 实际计算：circular diff of yaw angles
    delta_yaw_deg = _circular_diff_deg(math.degrees(yaw1), math.degrees(yaw0))

    # 点云方位角均值变化量（度，有向）
    if math.isfinite(phi_mean1) and math.isfinite(phi_mean2):
        delta_phi_deg = _circular_diff_deg(phi_mean2, phi_mean1)
    else:
        delta_phi_deg = float("nan")

    print("\n" + "─" * 60)
    print(f"[DIAGNOSE] 偏航变化量:    Δyaw  = {delta_yaw_deg:+.1f}°")
    print(f"[DIAGNOSE] phi均值变化量: Δphi  = {delta_phi_deg:+.1f}°")
    print("─" * 60)

    # 判断阈值
    WORLD_THRESHOLD = 20.0    # |Δphi| < 20° → 认为是世界系（点云不随机体转）
    BODY_MATCH_TOL  = 35.0    # ||Δphi| - |Δyaw|| < 35° → 认为是机体系

    frame = "unknown"
    conclusion = ""

    if math.isnan(delta_phi_deg):
        frame = "unknown"
        conclusion = (
            "诊断失败：无法获取有效点云（障碍物可能不在 LiDAR FOV 内）。\n"
            "请检查障碍物位置或调整 LiDAR 参数。"
        )
    elif abs(delta_phi_deg) < WORLD_THRESHOLD:
        frame = "world"
        conclusion = (
            f"✅ 世界坐标系 (World Frame)\n"
            f"无人机偏航 {delta_yaw_deg:+.1f}°，点云 phi 均值仅变化 {delta_phi_deg:+.1f}°（< {WORLD_THRESHOLD}°阈值）。\n"
            f"点云方向固定于世界系，不随机体旋转。\n"
            f"get_pointcloud() 返回世界坐标系下的相对向量。"
        )
    elif abs(abs(delta_phi_deg) - abs(delta_yaw_deg)) < BODY_MATCH_TOL:
        frame = "body"
        conclusion = (
            f"⚠️  机体坐标系 (Body Frame)\n"
            f"无人机偏航 {delta_yaw_deg:+.1f}°，点云 phi 均值变化 {delta_phi_deg:+.1f}°（接近偏航量）。\n"
            f"点云方向随机体旋转，说明是机体坐标系。\n"
            f"get_pointcloud() 返回机体坐标系下的相对向量。\n"
            f"需要将速度转换为机体系，或将点云转换为世界系，才能正确计算 bin 方向。"
        )
    else:
        frame = "unknown"
        conclusion = (
            f"❓ 不确定\n"
            f"无人机偏航 {delta_yaw_deg:+.1f}°，点云 phi 均值变化 {delta_phi_deg:+.1f}°。\n"
            f"变化量既不接近 0（世界系）也不接近偏航量（机体系）。\n"
            f"可能原因：障碍物稀疏/分布不均匀、传感器延迟、episode 中途 reset 等。\n"
            f"建议增加 --num_obstacles、延长各阶段步数后重试。"
        )

    print(f"\n[DIAGNOSE] 判断结果: {frame.upper()}")
    print(f"[DIAGNOSE] {conclusion}")
    print("=" * 70)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 保存诊断图像
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) 方位角直方图对比图
    hist_path = save_dir / "lidar_frame_diagnosis_phi_histogram.png"
    save_phi_histogram_png(
        phi1_deg=phi1_deg,
        phi2_deg=phi2_deg,
        save_path=str(hist_path),
        label1=f"阶段0（初始偏航 {math.degrees(yaw0):.1f}°）",
        label2=f"阶段2（旋转后偏航 {math.degrees(yaw1):.1f}°）",
        mean1=phi_mean1,
        mean2=phi_mean2,
        conclusion=conclusion.replace("\n", " | "),
    )

    # 2) 点云 XY 俯视图对比（用于直观查看点云是否随机体转动）
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    fig.suptitle(f"LiDAR 点云 XY 俯视图对比\n{conclusion.split(chr(10))[0]}", fontsize=11)

    for ax, phi_data, label, yaw_deg in [
        (axes[0], phi1_deg, f"阶段0（偏航={math.degrees(yaw0):.1f}°）", math.degrees(yaw0)),
        (axes[1], phi2_deg, f"阶段2（偏航={math.degrees(yaw1):.1f}°）", math.degrees(yaw1)),
    ]:
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)

        # 绘制机体朝向箭头（单位：3m 长）
        arrow_len = 3.0
        yaw_r = math.radians(yaw_deg)
        ax.annotate(
            "", xy=(arrow_len * math.cos(yaw_r), arrow_len * math.sin(yaw_r)),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.0),
        )
        ax.text(
            arrow_len * math.cos(yaw_r) * 1.1,
            arrow_len * math.sin(yaw_r) * 1.1,
            "机头",
            color="red", fontsize=9,
        )

    # 重新采集点云用于俯视图（此时已在阶段2之后）
    # 用已有的 phi 数据反推 XY 点（仅用于可视化，半径假设均匀）
    # 实际上我们直接用 phi 数组画极坐标散点
    for ax, phi_data in [(axes[0], phi1_deg), (axes[1], phi2_deg)]:
        if phi_data is not None and len(phi_data) > 0:
            phi_r = np.radians(phi_data)
            # 用 r=1 的单位圆散点（只关心方向，不关心距离）
            ax.scatter(
                np.cos(phi_r), np.sin(phi_r),
                s=0.5, alpha=0.3, color="steelblue",
            )
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)

    xy_path = save_dir / "lidar_frame_diagnosis_xy_view.png"
    fig.tight_layout()
    fig.savefig(str(xy_path), bbox_inches="tight")
    plt.close(fig)
    print(f"[DIAGNOSE] XY 俯视图已保存: {xy_path}")

    result = {
        "frame": frame,
        "delta_phi": delta_phi_deg,
        "delta_yaw": delta_yaw_deg,
        "phi_mean1": phi_mean1,
        "phi_mean2": phi_mean2,
        "conclusion": conclusion,
    }
    return result


# =============================================================================
# 1) 启动 Kit（关键：必须在 isaaclab/mdp 等 import 之前）
# =============================================================================
args_cli = parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =============================================================================
# 2) 现在再 import isaaclab/torch/pxr 相关
# =============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
    ActionTermCfg as ActionTermCfg,
)

from isaaclab.managers.action_manager import ActionTerm

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.sensors import LidarSensorCfg

import sys

WORKSPACE_PATH = Path.home() / "hjr_isaacdrone_ws" / "omniperception_isaacdrone" / "source" / "omniperception_isaacdrone"
if str(WORKSPACE_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_PATH))
    print(f"[INFO] 添加路径到 Python path: {WORKSPACE_PATH}")

try:
    from omniperception_isaacdrone.assets.robots.drone_cfg import DRONE_CFG
    print("[INFO] 成功导入自定义无人机配置: drone_cfg.DRONE_CFG")
except ImportError as e:
    print(f"[ERROR] 无法导入 drone_cfg: {e}")
    simulation_app.close()
    raise

try:
    from omniperception_isaacdrone.assets.sensors.lidar_cfg import LIDAR_CFG
    print("[INFO] 成功导入自定义LiDAR配置: lidar_cfg.LIDAR_CFG")
except ImportError as e:
    print(f"[WARN] 无法导入 lidar_cfg: {e}")
    LIDAR_CFG = None


class RootTwistVelocityActionTerm(ActionTerm):
    """6D 动作: [vx, vy, vz, wx, wy, wz] 写入 articulation root velocity。"""

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset = env.scene[cfg.asset_name]
        self._device = env.device
        self._num_envs = env.num_envs

        self._raw_actions = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        self._processed_actions = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)

        params = getattr(cfg, "params", None)
        if params is None:
            params = {}
        if isinstance(params, dict):
            p_get = params.get
        else:
            p_get = lambda k, default=None: getattr(params, k, default)

        self._lin_scale = float(p_get("lin_scale", getattr(cfg, "lin_scale", args_cli.lin_vel_scale)))
        self._ang_scale = float(p_get("ang_scale", getattr(cfg, "ang_scale", args_cli.ang_vel_scale)))
        self._lin_clip  = float(p_get("lin_clip",  getattr(cfg, "lin_clip",  args_cli.lin_vel_clip)))
        self._ang_clip  = float(p_get("ang_clip",  getattr(cfg, "ang_clip",  args_cli.ang_vel_clip)))

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        if actions.device != self._device:
            actions = actions.to(self._device)
        self._raw_actions.copy_(actions)
        lin = actions[:, 0:3] * self._lin_scale
        ang = actions[:, 3:6] * self._ang_scale
        if self._lin_clip > 0:
            lin = torch.clamp(lin, -self._lin_clip, self._lin_clip)
        if self._ang_clip > 0:
            ang = torch.clamp(ang, -self._ang_clip, self._ang_clip)
        self._processed_actions[:, 0:3] = lin
        self._processed_actions[:, 3:6] = ang

    def apply_actions(self):
        self._asset.write_root_velocity_to_sim(self._processed_actions)


# =============================================================================
# 3) reset 逻辑
# =============================================================================
def reset_root_state_on_square_edge(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    square_half_size: float = 35.0,
    z_range: tuple = (3.0, 7.0),
):
    asset = env.scene[asset_cfg.name]
    num_resets = len(env_ids)

    edges = torch.randint(0, 4, (num_resets,), device=env.device)
    positions = torch.zeros((num_resets, 3), device=env.device)
    edge_positions = torch.rand(num_resets, device=env.device) * 2 * square_half_size - square_half_size

    left_mask = edges == 0
    right_mask = edges == 1
    bottom_mask = edges == 2
    top_mask = edges == 3

    positions[left_mask, 0] = -square_half_size
    positions[left_mask, 1] = edge_positions[left_mask]
    positions[right_mask, 0] = square_half_size
    positions[right_mask, 1] = edge_positions[right_mask]
    positions[bottom_mask, 0] = edge_positions[bottom_mask]
    positions[bottom_mask, 1] = -square_half_size
    positions[top_mask, 0] = edge_positions[top_mask]
    positions[top_mask, 1] = square_half_size

    positions[:, 2] = torch.rand(num_resets, device=env.device) * (z_range[1] - z_range[0]) + z_range[0]

    orientations = torch.zeros((num_resets, 4), device=env.device)
    orientations[:, 0] = 1.0

    root_states = torch.cat([positions, orientations], dim=1)
    asset.write_root_pose_to_sim(root_states, env_ids=env_ids)

    velocities = torch.zeros((num_resets, 6), device=env.device)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# =============================================================================
# 4) reward / termination
# =============================================================================
def _get_root_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return mdp.root_pos_w(env, asset_cfg=asset_cfg)


def reward_distance_to_goal(env, asset_cfg: SceneEntityCfg, std: float = 5.0) -> torch.Tensor:
    pos = _get_root_pos(env, asset_cfg)
    diff = pos - env.goal_pos_w
    dist2 = (diff * diff).sum(dim=-1)
    return torch.exp(-dist2 / (2.0 * std * std))


def reward_height_tracking(env, asset_cfg: SceneEntityCfg, target_z: float = 5.0, std: float = 2.0):
    pos = _get_root_pos(env, asset_cfg)
    dz2 = (pos[:, 2] - target_z) ** 2
    return torch.exp(-dz2 / (2.0 * std * std))


def reward_stability(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, lin_std: float = 2.0, ang_std: float = 6.0):
    lin = mdp.base_lin_vel(env, asset_cfg=asset_cfg)
    ang = mdp.base_ang_vel(env, asset_cfg=asset_cfg)
    lin2 = (lin * lin).sum(dim=-1)
    ang2 = (ang * ang).sum(dim=-1)
    return torch.exp(-lin2 / (2.0 * lin_std * lin_std)) * torch.exp(-ang2 / (2.0 * ang_std * ang_std))


def reward_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    term = env.action_manager.get_term("root_twist")
    a = term.raw_actions
    return (a * a).sum(dim=-1)


def termination_crash_or_oob(env, asset_cfg: SceneEntityCfg, min_height: float, world_bound: float):
    pos = _get_root_pos(env, asset_cfg)
    crash = pos[:, 2] < min_height
    oob = (pos[:, 0].abs() > world_bound) | (pos[:, 1].abs() > world_bound)
    return crash | oob


# =============================================================================
# 5) 观测项
# =============================================================================
def obs_goal_delta(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    goal = getattr(env, "goal_pos_w", None)
    if goal is None:
        return torch.zeros_like(pos)
    if goal.device != pos.device:
        goal = goal.to(pos.device)
    if goal.shape[0] == 1 and pos.shape[0] > 1:
        goal = goal.expand(pos.shape[0], 3)
    elif goal.shape[0] != pos.shape[0]:
        goal = goal[:1].expand(pos.shape[0], 3)
    return goal - pos


def obs_lidar_min_range_grid(
    env: ManagerBasedRLEnv,
    lidar_name: str = "lidar",
    theta_min: float = 30.0,
    theta_max: float = 90.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    delta_theta: float = 1.0,
    delta_phi: float = 5.0,
    empty_value: float = 0.0,
    max_vis_points: int | None = None,
) -> torch.Tensor:
    T = int((theta_max - theta_min) / delta_theta)
    Pn = int((phi_max - phi_min) / delta_phi)
    out_shape = (env.num_envs, T * Pn)

    if not hasattr(env, "scene"):
        return torch.zeros(out_shape, device=env.device, dtype=torch.float32)

    try:
        lidar = env.scene[lidar_name]
    except KeyError:
        return torch.zeros(out_shape, device=env.device, dtype=torch.float32)
    except Exception:
        return torch.zeros(out_shape, device=env.device, dtype=torch.float32)

    env_ids = torch.arange(env.num_envs, device=env.device)
    pc, _ = _get_downsampled_pc_torch(env, lidar, env_ids, max_pts=max_vis_points)
    if pc is None:
        T = int((theta_max - theta_min) / delta_theta)
        P = int((phi_max - phi_min) / delta_phi)
        return torch.zeros((env.num_envs, T * P), device=env.device, dtype=torch.float32)

    x = pc[..., 0]
    y = pc[..., 1]
    z = pc[..., 2]

    valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z)
    r = torch.sqrt(x * x + y * y + z * z + 1e-12)
    cos_theta = torch.clamp(z / r, -1.0, 1.0)
    theta = torch.rad2deg(torch.acos(cos_theta))
    phi = torch.rad2deg(torch.atan2(y, x))
    phi = torch.remainder(phi, 360.0)

    in_theta = (theta >= theta_min) & (theta < theta_max)
    in_phi = (phi >= phi_min) & (phi < phi_max)
    m = valid & in_theta & in_phi

    T = int((theta_max - theta_min) / delta_theta)
    Pn = int((phi_max - phi_min) / delta_phi)
    num_bins = T * Pn

    bins = torch.full((env.num_envs, num_bins), float("inf"), device=env.device, dtype=torch.float32)

    if m.any():
        t_idx = torch.floor((theta - theta_min) / delta_theta).to(torch.long)
        p_idx = torch.floor((phi - phi_min) / delta_phi).to(torch.long)
        t_idx = torch.clamp(t_idx, 0, T - 1)
        p_idx = torch.clamp(p_idx, 0, Pn - 1)
        lin_idx = t_idx * Pn + p_idx

        for e in range(env.num_envs):
            me = m[e]
            if me.any():
                idx_e = lin_idx[e, me]
                r_e = r[e, me].to(torch.float32)
                bins[e].scatter_reduce_(0, idx_e, r_e, reduce="amin", include_self=True)

    if empty_value == 0.0:
        bins = torch.where(torch.isfinite(bins), bins, torch.zeros_like(bins))
    else:
        bins = torch.where(torch.isfinite(bins), bins, torch.full_like(bins, float(empty_value)))

    return bins


# =============================================================================
# 6) Scene / Env 配置
# =============================================================================
@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=True,
    )
    robot: ArticulationCfg = DRONE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )
    robot.spawn = DRONE_CFG.spawn.replace(
        scale=(20, 20, 10),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(enable_gyroscopic_forces=True),
    )
    robot.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    )
    if LIDAR_CFG is not None and args_cli.enable_lidar:
        lidar: LidarSensorCfg = LIDAR_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/body",
        )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9), angle=0.53),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )


@configclass
class ActionsCfg:
    root_twist = ActionTermCfg(
        class_type=RootTwistVelocityActionTerm,
        asset_name="robot",
    )
    root_twist.params = {
        "lin_scale": args_cli.lin_vel_scale,
        "ang_scale": args_cli.ang_vel_scale,
        "lin_clip":  args_cli.lin_vel_clip,
        "ang_clip":  args_cli.ang_vel_clip,
    }


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        root_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        root_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        goal_delta = ObsTerm(func=obs_goal_delta, params={"asset_cfg": SceneEntityCfg("robot")})
        lidar_grid = ObsTerm(
            func=obs_lidar_min_range_grid,
            params={
                "lidar_name": "lidar",
                "theta_min": args_cli.lidar_theta_min,
                "theta_max": args_cli.lidar_theta_max,
                "phi_min": args_cli.lidar_phi_min,
                "phi_max": args_cli.lidar_phi_max,
                "delta_theta": args_cli.lidar_delta_theta,
                "delta_phi": args_cli.lidar_delta_phi,
                "empty_value": args_cli.lidar_empty_value,
                "max_vis_points": int(args_cli.lidar_max_vis_points),
            },
        )
        def post_init(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_base = EventTerm(
        func=reset_root_state_on_square_edge,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "square_half_size": args_cli.square_half_size,
            "z_range": (args_cli.z_init_min, args_cli.z_init_max),
        },
    )


@configclass
class RewardsCfg:
    dist_to_goal = RewTerm(
        func=reward_distance_to_goal,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 6.0},
    )
    height = RewTerm(
        func=reward_height_tracking,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_z": 5.0, "std": 2.5},
    )
    stability = RewTerm(
        func=reward_stability,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "lin_std": 2.0, "ang_std": 6.0},
    )
    action_l2 = RewTerm(func=reward_action_l2, weight=-0.01)
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    crash_or_oob = DoneTerm(
        func=termination_crash_or_oob,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": args_cli.min_height,
            "world_bound": args_cli.world_bound,
        },
    )


@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=0.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        try:
            super().__post_init__()
        except Exception:
            pass
        self.decimation = 2
        self.episode_length_s = 20.0
        self.viewer.eye = (60.0, 60.0, 40.0)
        self.viewer.lookat = (0.0, 0.0, 5.0)
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation


# =============================================================================
# 7) 共享障碍物生成
# =============================================================================
class ObstacleSpawner:
    def __init__(
        self,
        num_obstacles: int = 50,
        x_range: tuple = (-33.0, 33.0),
        y_range: tuple = (-33.0, 33.0),
        xy_size_range: tuple = (0.5, 1.5),
        z_height: float = 10.0,
        seed: int = 42,
    ):
        self.num_obstacles = num_obstacles
        self.x_range = x_range
        self.y_range = y_range
        self.xy_size_range = xy_size_range
        self.z_height = z_height
        if seed is not None:
            np.random.seed(seed)

    def spawn_obstacles(self):
        import isaacsim.core.utils.prims as prim_utils
        prim_utils.create_prim("/World/Obstacles", "Xform")
        print(f"\n[INFO]: 正在生成 {self.num_obstacles} 个共享障碍物...")
        for i in range(self.num_obstacles):
            x_pos = np.random.uniform(*self.x_range)
            y_pos = np.random.uniform(*self.y_range)
            z_pos = self.z_height / 2.0
            x_size = np.random.uniform(*self.xy_size_range)
            y_size = np.random.uniform(*self.xy_size_range)
            z_size = self.z_height
            color = (np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7))
            cfg_obstacle = sim_utils.CuboidCfg(
                size=(x_size, y_size, z_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            obstacle_path = f"/World/Obstacles/Obstacle_{i:04d}"
            cfg_obstacle.func(obstacle_path, cfg_obstacle, translation=(x_pos, y_pos, z_pos))
            if (i + 1) % 10 == 0:
                print(f"[INFO]: 已生成 {i + 1}/{self.num_obstacles} 个障碍物")
        print("[INFO]: 共享障碍物生成完成！")


# =============================================================================
# 9) 自定义 RLEnv
# =============================================================================
class MyDroneRLEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: MyEnvCfg):
        self.goal_pos_w = torch.zeros((1, 3), dtype=torch.float32)
        super().__init__(cfg=cfg)
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._sample_goals(torch.arange(self.num_envs, device=self.device))

    def _sample_goals(self, env_ids: torch.Tensor):
        half = float(args_cli.square_half_size)
        n = env_ids.numel()
        gx = (torch.rand(n, device=self.device) * 2 - 1) * half
        gy = (torch.rand(n, device=self.device) * 2 - 1) * half
        gz = torch.rand(n, device=self.device) * (args_cli.goal_z_max - args_cli.goal_z_min) + args_cli.goal_z_min
        self.goal_pos_w[env_ids, 0] = gx
        self.goal_pos_w[env_ids, 1] = gy
        self.goal_pos_w[env_ids, 2] = gz

    def reset_idx(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_goals(env_ids)
        obs, info = super().reset_idx(env_ids)
        try:
            pos = mdp.root_pos_w(self, asset_cfg=SceneEntityCfg("robot"))
            info["goal_delta"] = (self.goal_pos_w - pos).detach()
            info["goal_pos_w"] = self.goal_pos_w.detach()
        except Exception:
            pass
        return obs, info


# =============================================================================
# 10) ActorCritic（保持不变）
# =============================================================================
def gaussian_log_prob(actions, mu, std):
    var = std * std
    log_scale = torch.log(std + 1e-8)
    return -0.5 * (((actions - mu) ** 2) / (var + 1e-8) + 2 * log_scale + math.log(2 * math.pi)).sum(dim=-1)


def gaussian_entropy(std):
    return (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std + 1e-8)).sum(dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mu = self.mu(h)
        v = self.v(h).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, v

    def act(self, obs: torch.Tensor):
        mu, std, v = self.forward(obs)
        eps = torch.randn_like(mu)
        a = mu + eps * std
        logp = gaussian_log_prob(a, mu, std)
        ent = gaussian_entropy(std)
        return a, logp, v, ent

    def value(self, obs: torch.Tensor):
        _, _, v = self.forward(obs)
        return v

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mu, std, v = self.forward(obs)
        logp = gaussian_log_prob(actions, mu, std)
        ent = gaussian_entropy(std)
        return logp, ent, v


# =============================================================================
# 11) main
# =============================================================================
def main():
    print("=" * 80)
    print("ManagerBasedRLEnv Drone - single file + LiDAR 坐标系诊断")
    print("=" * 80)

    # 诊断模式需要 enable_lidar
    if args_cli.diagnose_lidar_frame and not args_cli.enable_lidar:
        print("[ERROR] --diagnose_lidar_frame 需要同时指定 --enable_lidar，退出。")
        simulation_app.close()
        return

    env_cfg = MyEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    try:
        _ = float(env_cfg.decimation)
    except Exception:
        env_cfg.decimation = 2
        env_cfg.episode_length_s = 20.0
        env_cfg.sim.dt = 1.0 / 120.0
        env_cfg.sim.render_interval = env_cfg.decimation

    decim = int(env_cfg.decimation)
    dt = float(env_cfg.sim.dt)
    ctrl_hz = 1.0 / (dt * decim)

    print(f"\n[配置] num_envs={env_cfg.scene.num_envs}, num_obstacles={args_cli.num_obstacles}")
    print(f"[配置] sim_dt={dt:.6f}, decimation={decim}, ctrl_hz={ctrl_hz:.1f}")

    print("\n[状态] 正在生成共享障碍物...")
    ObstacleSpawner(num_obstacles=args_cli.num_obstacles).spawn_obstacles()

    print("\n[状态] 正在创建 RL 环境...")
    env = MyDroneRLEnv(cfg=env_cfg)
    print("[状态] 环境创建成功!")

    lidar = None
    robot_asset = None

    if args_cli.enable_lidar:
        try:
            lidar = env.scene["lidar"]
            print("[INFO] LiDAR 已启用")
        except Exception as e:
            print(f"[WARN] LiDAR 启用失败: {e}")
            lidar = None

    try:
        robot_asset = env.scene["robot"]
    except Exception:
        robot_asset = None

    print("\n[状态] reset...")
    obs, info = env.reset()
    print("[状态] reset 完成!")
    print("policy obs shape:", obs["policy"].shape)

    env0_ids = torch.tensor([0], device=env.device)

    # =========================================================================
    # 诊断模式：运行完即退出
    # =========================================================================
    if args_cli.diagnose_lidar_frame:
        if lidar is None:
            print("[ERROR] LiDAR 未成功初始化，无法进行诊断。")
            env.close()
            simulation_app.close()
            return

        if robot_asset is None:
            print("[ERROR] 无法获取 robot asset，无法进行诊断。")
            env.close()
            simulation_app.close()
            return

        save_dir = Path(args_cli.lidar_save_dir).expanduser()

        result = run_lidar_frame_diagnosis(
            env=env,
            lidar=lidar,
            robot_asset=robot_asset,
            save_dir=save_dir,
            phase1_steps=args_cli.diagnose_phase1_steps,
            rotate_steps=args_cli.diagnose_rotate_steps,
            phase2_steps=args_cli.diagnose_phase2_steps,
            yaw_rate_rad=args_cli.diagnose_yaw_rate,
            max_pts=int(args_cli.lidar_max_vis_points),
        )

        print("\n" + "=" * 70)
        print("[DIAGNOSE] 诊断完成，结果摘要：")
        print(f"  坐标系类型: {result['frame'].upper()}")
        print(f"  Δyaw      : {result['delta_yaw']:+.1f}°")
        print(f"  Δphi      : {result['delta_phi']:+.1f}°")
        print(f"  phi_mean1 : {result['phi_mean1']:.1f}°")
        print(f"  phi_mean2 : {result['phi_mean2']:.1f}°")
        print(f"  图像保存至: {save_dir}")
        print("=" * 70)

        env.close()
        return

    # =========================================================================
    # 非诊断模式：正常运行
    # =========================================================================

    def _maybe_print_lidar_stats(step_i: int):
        if lidar is None:
            return
        if (step_i % args_cli.lidar_print_every) != 0:
            return
        try:
            d0 = lidar.get_distances(env0_ids)[0]
            max_d = float(getattr(lidar.cfg, "max_distance", 0.0))
            hit_mask = (d0 < max_d) if (max_d and max_d > 0) else torch.isfinite(d0)
            total = int(d0.numel())
            hit_count = int(hit_mask.sum().item())
            if hit_count > 0:
                d_hit = d0[hit_mask]
                print(
                    f"  [LiDAR] rays={total}, hits={hit_count}, "
                    f"min={d_hit.min().item():.2f}, mean={d_hit.mean().item():.2f}, max={d_hit.max().item():.2f}"
                )
            else:
                print(f"  [LiDAR] rays={total}, hits=0")
        except Exception as e:
            print(f"  [LiDAR] read failed: {e}")

    def _maybe_save_pc_png(step_i: int, points_np, save_dir: Path, saved_count: int):
        if lidar is None:
            return saved_count
        if (step_i % args_cli.lidar_save_every) != 0:
            return saved_count
        if saved_count >= args_cli.lidar_save_max:
            return saved_count
        if points_np is None:
            return saved_count
        try:
            png_path = save_dir / f"lidar_pc_step_{step_i:06d}.png"
            ok = save_pointcloud_png(points_np, str(png_path), title=f"LiDAR pointcloud step={step_i}", s=2)
            if ok:
                saved_count += 1
                print(f"[INFO] 已保存点云PNG: {png_path}")
        except Exception:
            pass
        return saved_count

    if not args_cli.train:
        save_dir = Path(args_cli.lidar_save_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        count = 0

        while simulation_app.is_running() and count < args_cli.max_steps:
            with torch.inference_mode():
                actions = torch.rand(env.num_envs, env.action_manager.total_action_dim, device=env.device) * 2 - 1
                obs, rew, terminated, truncated, info = env.step(actions)

                if (count % 50) == 0:
                    try:
                        pos0 = mdp.root_pos_w(env, asset_cfg=SceneEntityCfg("robot"))[0].detach().cpu().numpy()
                        goal0 = env.goal_pos_w[0].detach().cpu().numpy()
                        done0 = bool((terminated | truncated)[0].item())
                        print(f"[step {count}] pos0={pos0}, goal0={goal0}, rew0={float(rew[0]):.3f}, done0={done0}")
                    except Exception:
                        print(f"[step {count}] rew0={float(rew[0]):.3f}")

                _maybe_print_lidar_stats(count)

                need_pc = (lidar is not None) and (
                    ((count % args_cli.lidar_save_every == 0) and (saved_count < args_cli.lidar_save_max))
                )
                points_np = None
                if need_pc:
                    points_np, _ = _get_downsampled_pc_np(env, lidar, env0_ids, max_pts=int(args_cli.lidar_max_vis_points))

                saved_count = _maybe_save_pc_png(count, points_np, save_dir, saved_count)
                count += 1

        print("\n[INFO] 非训练模式结束，关闭环境...")
        env.close()
        return

    # ── 训练模式（保持不变）────────────────────────────────────────────────────
    obs_tensor = obs["policy"]
    obs_dim = obs_tensor.shape[-1]
    act_dim = env.action_manager.total_action_dim

    print(f"\n[TRAIN] obs_dim={obs_dim}, act_dim={act_dim}, device={env.device}")

    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(env.device)
    optimizer = optim.Adam(model.parameters(), lr=args_cli.lr)

    T = args_cli.rollout_len
    N = env.num_envs
    gamma = float(args_cli.gamma)
    gae_lam = float(args_cli.gae_lambda)
    vf_coef = float(args_cli.value_coef)
    ent_coef = float(args_cli.entropy_coef)
    max_grad_norm = float(args_cli.max_grad_norm)
    log_every = int(args_cli.log_every)

    for it in range(1, args_cli.train_iters + 1):
        obs_buf = torch.zeros((T, N, obs_dim), device=env.device)
        act_buf = torch.zeros((T, N, act_dim), device=env.device)
        logp_buf = torch.zeros((T, N), device=env.device)
        val_buf = torch.zeros((T, N), device=env.device)
        rew_buf = torch.zeros((T, N), device=env.device)
        done_buf = torch.zeros((T, N), device=env.device, dtype=torch.bool)

        for t in range(T):
            obs_t = obs["policy"]
            with torch.no_grad():
                act_t, logp_t, val_t, ent_t = model.act(obs_t)
            next_obs, rew, terminated, truncated, info = env.step(act_t)
            done = (terminated | truncated)
            obs_buf[t].copy_(obs_t)
            act_buf[t].copy_(act_t)
            logp_buf[t].copy_(logp_t)
            val_buf[t].copy_(val_t)
            rew_buf[t].copy_(rew)
            done_buf[t].copy_(done)
            obs = next_obs

        with torch.no_grad():
            last_val = model.value(obs["policy"])

        adv = torch.zeros((T, N), device=env.device)
        gae = torch.zeros((N,), device=env.device)
        for t in reversed(range(T)):
            not_done = (~done_buf[t]).float()
            next_value = last_val if t == (T - 1) else val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_value * not_done - val_buf[t]
            gae = delta + gamma * gae_lam * not_done * gae
            adv[t] = gae

        ret = adv + val_buf
        adv_mean = adv.mean()
        adv_std = adv.std().clamp_min(1e-6)
        adv_n = (adv - adv_mean) / adv_std

        B = T * N
        flat_obs = obs_buf.reshape(B, obs_dim)
        flat_act = act_buf.reshape(B, act_dim)
        flat_adv = adv_n.reshape(B)
        flat_ret = ret.reshape(B)

        new_logp, new_ent, new_val = model.evaluate_actions(flat_obs, flat_act)
        policy_loss = -(flat_adv.detach() * new_logp).mean()
        value_loss = 0.5 * (flat_ret - new_val).pow(2).mean()
        entropy_loss = -new_ent.mean()
        loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if (it % log_every) == 0:
            avg_rew = rew_buf.mean().item()
            done_rate = done_buf.float().mean().item()
            print(
                f"[TRAIN it={it:04d}] loss={loss.item():.4f} "
                f"pi={policy_loss.item():.4f} v={value_loss.item():.4f} ent={new_ent.mean().item():.4f} "
                f"avg_rew={avg_rew:.3f} done_rate={done_rate:.3f}"
            )

    print("\n[TRAIN] 训练结束，关闭环境...")
    env.close()


# =============================================================================
# 12) entrypoint
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[信息] 用户中断")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
        print("[状态] 仿真器已关闭")
