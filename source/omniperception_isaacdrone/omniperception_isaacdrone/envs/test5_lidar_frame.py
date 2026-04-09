# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from pathlib import Path
import math
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message="Glyph.*missing from current font")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--env_spacing", type=float, default=50.0)
    parser.add_argument("--num_obstacles", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--square_half_size", type=float, default=35.0)
    parser.add_argument("--z_init_min", type=float, default=3.0)
    parser.add_argument("--z_init_max", type=float, default=7.0)
    parser.add_argument("--goal_z_min", type=float, default=3.0)
    parser.add_argument("--goal_z_max", type=float, default=8.0)
    parser.add_argument("--lin_vel_scale", type=float, default=1.0)
    parser.add_argument("--ang_vel_scale", type=float, default=1.0)
    parser.add_argument("--lin_vel_clip", type=float, default=5.0)
    parser.add_argument("--ang_vel_clip", type=float, default=6.0)
    parser.add_argument("--min_height", type=float, default=0.25)
    parser.add_argument("--world_bound", type=float, default=40.0)
    parser.add_argument("--enable_lidar", action="store_true")
    parser.add_argument("--lidar_print_every", type=int, default=50)
    parser.add_argument("--lidar_max_vis_points", type=int, default=10000)
    parser.add_argument("--lidar_save_every", type=int, default=50)
    parser.add_argument("--lidar_save_max", type=int, default=20)
    parser.add_argument("--lidar_save_dir", type=str, default=str(Path.home() / "lidar_pc_images"))
    parser.add_argument("--lidar_theta_min", type=float, default=30.0)
    parser.add_argument("--lidar_theta_max", type=float, default=90.0)
    parser.add_argument("--lidar_phi_min", type=float, default=0.0)
    parser.add_argument("--lidar_phi_max", type=float, default=360.0)
    parser.add_argument("--lidar_delta_theta", type=float, default=1.0)
    parser.add_argument("--lidar_delta_phi", type=float, default=5.0)
    parser.add_argument("--lidar_empty_value", type=float, default=0.0)
    
    # 诊断相关参数
    parser.add_argument("--diagnose_lidar_frame", action="store_true")
    parser.add_argument("--flight_test", action="store_true", help="运行自定义飞行与连续可视化测试")
    
    parser.add_argument("--diagnose_phase1_steps", type=int, default=50)
    parser.add_argument("--diagnose_rotate_steps", type=int, default=100)
    parser.add_argument("--diagnose_phase2_steps", type=int, default=50)
    parser.add_argument("--diagnose_yaw_rate", type=float, default=math.pi / 2.0)
    parser.add_argument("--diagnose_collect_every", type=int, default=2)
    parser.add_argument("--diagnose_min_xy_radius", type=float, default=0.5)
    parser.add_argument("--diagnose_hist_bin_deg", type=float, default=5.0)
    parser.add_argument("--diagnose_scope_settle_steps", type=int, default=30)
    parser.add_argument("--diagnose_scope_aabb_margin", type=float, default=2.0)
    
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_iters", type=int, default=2000)
    parser.add_argument("--rollout_len", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


# =============================================================================
# Visualization helpers
# =============================================================================
def save_pointcloud_png(points_np, save_path, title=None, s=1):
    if points_np is None or len(points_np) == 0:
        return False
    try:
        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=s)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        if title:
            ax.set_title(title)
        mn, mx = points_np.min(0), points_np.max(0)
        c = (mn + mx) / 2.0
        r = max((mx - mn).max() / 2.0, 1e-6)
        ax.set_xlim(c[0]-r, c[0]+r)
        ax.set_ylim(c[1]-r, c[1]+r)
        ax.set_zlim(c[2]-r, c[2]+r)
        ax.view_init(elev=25, azim=45)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[WARN] save_pointcloud_png failed: {e}")
        plt.close("all")
        return False


def save_xy_scatter_png(points_np, save_path, title=None, s=2):
    if points_np is None or len(points_np) == 0:
        return False
    try:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
        ax.scatter(points_np[:, 0], points_np[:, 1], s=s, alpha=0.4)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[WARN] save_xy_scatter_png failed: {e}")
        plt.close("all")
        return False

def save_sensor_frame_xy_png(points_np, save_path, title=None):
    """专门为传感器坐标系(机体系)绘制点云与机头指向箭头的函数"""
    if points_np is None or len(points_np) == 0:
        return False
    try:
        fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
        # 画点云
        ax.scatter(points_np[:, 0], points_np[:, 1], s=4, alpha=0.6, color='steelblue')
        
        # 画无人机机体中心点 (0,0)
        ax.plot(0, 0, 'ko', markersize=6, zorder=5, label='Drone Center')
        # 画红色箭头代表 Yaw (在机体坐标系下，机头永远朝向 +X)
        ax.arrow(0, 0, 3.0, 0, head_width=0.8, head_length=1.0, fc='red', ec='red', 
                 linewidth=2.5, zorder=6, label='Yaw Direction (+X)')

        ax.set_xlabel("Sensor X (Forward) [m]")
        ax.set_ylabel("Sensor Y (Left) [m]")
        # 固定坐标轴范围，以清晰观察周围物体的相对运动
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=8)
        if title:
            ax.set_title(title, fontsize=10)
        
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[WARN] plot failed: {e}")
        plt.close("all")
        return False

# ... 其余原有的 helpers 代码 (save_phi_histogram_png 等)
def save_phi_histogram_png(phi1_deg, phi2_deg, save_path, label1="Phase1", label2="Phase2", mean1=None, mean2=None, conclusion="", bin_deg=5.0):
    try:
        bins = np.arange(0.0, 360.0 + bin_deg, bin_deg)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        title_str = conclusion[:120].replace("\n", " | ") if conclusion else ""
        fig.suptitle(f"LiDAR Frame Diagnosis\n{title_str}", fontsize=10, y=1.01)
        for ax, data, label, color, mean in zip(axes, [phi1_deg, phi2_deg], [label1, label2], ["steelblue", "tomato"], [mean1, mean2]):
            if data is not None and len(data) > 0:
                ax.hist(data, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
                if mean is not None and math.isfinite(float(mean)):
                    ax.axvline(float(mean), color="black", linewidth=2.0, linestyle="--", label=f"mean={float(mean):.1f} deg")
                    ax.legend(fontsize=9)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("phi (deg)"); ax.set_ylabel("count")
            ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 45))
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[DIAGNOSE] phi histogram saved: {save_path}")
    except Exception as e:
        print(f"[WARN] save_phi_histogram_png failed: {e}")
        plt.close("all")

def _save_phi_vs_yaw_png(trajectory, save_path, slope=float("nan"), frame="unknown"):
    if not trajectory: return
    try:
        yaws = [t[0] for t in trajectory]; phis = [t[1] for t in trajectory]
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.scatter(yaws, phis, s=20, alpha=0.7, color="steelblue", label="samples")
        if phis and math.isfinite(phis[0]):
            phi0 = phis[0]; yaw_arr = np.linspace(min(yaws), max(yaws), 100)
            ax.plot(yaw_arr, [phi0]*len(yaw_arr), "g--", lw=1.5, label="world-frame (phi=const)")
            wrapped = [(phi0 - (y - yaws[0])) % 360.0 for y in yaw_arr]
            ax.plot(yaw_arr, wrapped, "r--", lw=1.5, label="body-frame (phi tracks -yaw)")
            if math.isfinite(slope): ax.plot(yaw_arr, slope*(yaw_arr - yaws[0]) + phi0, "k-", lw=2, label=f"fit slope={slope:+.3f}")
        ax.set_xlabel("yaw (deg)"); ax.set_ylabel("phi_mean (deg)")
        ax.set_title(f"phi vs yaw  [{frame.upper()}]  slope={slope:+.3f}" if math.isfinite(slope) else f"phi vs yaw  [{frame.upper()}]")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(save_path); plt.close(fig)
        print(f"[DIAGNOSE] phi-vs-yaw plot saved: {save_path}")
    except Exception as e:
        print(f"[WARN] _save_phi_vs_yaw_png failed: {e}")
        plt.close("all")

def _save_scope_comparison_png(results, expected, origin0, origin1, marker_global, save_path, scope_frame, conclusion):
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
        title_str = conclusion[:100].replace("\n", " | ")
        fig.suptitle(f"Scope Diagnosis [{scope_frame.upper()}]\n{title_str}", fontsize=10, y=1.01)
        subplot_cfg = [("env0_global", axes[0, 0], "env0 global hypothesis"), ("env1_global", axes[0, 1], "env1 global hypothesis [KEY]"),
                       ("env0_local",  axes[1, 0], "env0 local hypothesis"), ("env1_local",  axes[1, 1], "env1 local hypothesis [KEY]")]
        for key, ax, title in subplot_cfg:
            r = results.get(key, {}); exp = expected.get(key, np.zeros(3)); pts = r.get("pts"); c = r.get("centroid", np.full(3, np.nan))
            if pts is not None and len(pts) > 0: ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.6, color="steelblue", label=f"hits={r.get('n_hits',0)}")
            ax.plot(exp[0], exp[1], "r+", ms=18, mew=3, label=f"expect({exp[0]:.1f},{exp[1]:.1f})")
            if np.all(np.isfinite(c)): ax.plot(c[0], c[1], "go", ms=10, label=f"actual({c[0]:.1f},{c[1]:.1f})\nres={r.get('residual',float('inf')):.2f}")
            ax.set_title(title, fontsize=9); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3); ax.set_aspect("equal")
        fig.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
        print(f"[DIAGNOSE] scope comparison plot saved: {save_path}")
    except Exception as e:
        print(f"[WARN] _save_scope_comparison_png failed: {e}")
        plt.close("all")


# =============================================================================
# LiDAR utilities
# =============================================================================
def _circular_diff_deg(a, b):
    diff = (a - b) % 360.0
    return diff - 360.0 if diff > 180.0 else diff

def _compute_phi_mean(pc_np, min_xy_radius=0.5):
    if pc_np is None or len(pc_np) == 0: return None, float("nan"), float("nan")
    x, y = pc_np[:, 0], pc_np[:, 1]
    mask = np.sqrt(x**2 + y**2) > min_xy_radius
    if mask.sum() == 0: return None, float("nan"), float("nan")
    phi_rad = np.arctan2(y[mask], x[mask]); phi_deg = np.degrees(phi_rad) % 360.0
    sin_m, cos_m = np.mean(np.sin(phi_rad)), np.mean(np.cos(phi_rad))
    mean_deg = math.degrees(math.atan2(sin_m, cos_m)) % 360.0
    r_bar = min(math.sqrt(sin_m**2 + cos_m**2), 1.0 - 1e-9)
    std_deg = math.degrees(math.sqrt(max(-2.0 * math.log(r_bar + 1e-9), 0.0)))
    return phi_deg, mean_deg, std_deg

def _get_downsampled_pc_torch(env, lidar, env_ids, max_pts=None):
    if lidar is None: return None, None
    import torch
    pc = lidar.get_pointcloud(env_ids)
    if pc is None: return None, None
    if pc.dim() == 2: pc = pc.unsqueeze(0)
    e, p, _ = pc.shape
    num_raw = torch.full((e,), p, device=pc.device, dtype=torch.int32)
    finite_mask = torch.isfinite(pc).all(dim=-1)
    pc = pc.clone(); pc[~finite_mask] = float("nan")
    if max_pts is not None and max_pts > 0 and p > max_pts:
        idx = torch.randperm(p, device=pc.device)[:max_pts]
        pc = pc[:, idx, :]
    return pc, num_raw

def _get_downsampled_pc_np(env, lidar, env_ids, max_pts=10000):
    if lidar is None: return None, 0
    try:
        import torch
        pc_t, num_raw_t = _get_downsampled_pc_torch(env, lidar, env_ids, max_pts=max_pts)
        if pc_t is None or pc_t.shape[0] != 1: return None, 0
        pc0 = pc_t[0]
        num_raw = int(num_raw_t[0].item()) if num_raw_t is not None else int(pc0.shape[0])
        pts = pc0.detach().cpu().numpy()
        pts = pts[np.isfinite(pts).all(axis=1)]
        return (pts if len(pts) > 0 else None), num_raw
    except Exception:
        return None, 0

def _get_env_origins_np(env):
    origins = getattr(env.scene, "env_origins", None)
    return origins.detach().cpu().numpy() if origins is not None else None

def _extract_points_in_aabb(points_np, center, half_size):
    if points_np is None or len(points_np) == 0: return None
    lo = np.asarray(center) - np.asarray(half_size); hi = np.asarray(center) + np.asarray(half_size)
    mask = np.all((points_np >= lo) & (points_np <= hi), axis=1)
    pts = points_np[mask]
    return pts if len(pts) > 0 else None

def _centroid_or_nan(pts):
    if pts is None or len(pts) == 0: return np.full(3, np.nan)
    return pts.mean(axis=0)

def set_same_local_pose_for_envs(env, robot_asset, env_ids, local_xyz=(0., 0., 5.), yaw_deg=0.0):
    import torch
    with torch.inference_mode():
        origins = env.scene.env_origins[env_ids].clone()
        local_t = torch.tensor(local_xyz, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(len(env_ids), 1)
        pos_w = origins + local_t
        yaw = math.radians(float(yaw_deg))
        qw, qz = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        quat = torch.tensor([qw, 0., 0., qz], device=env.device, dtype=torch.float32).unsqueeze(0).repeat(len(env_ids), 1)
        pose = torch.cat([pos_w, quat], dim=1)
        vel  = torch.zeros(len(env_ids), 6, device=env.device)
        robot_asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        robot_asset.write_root_velocity_to_sim(vel, env_ids=env_ids)

# =============================================================================
# Flight Test (连续飞行与可视化测试)
# =============================================================================
def run_flight_test(env, lidar, robot_asset, save_dir, max_steps=200):
    """
    env_0: 沿 x方向飞，一边飞一边转 yaw
    env_1: 沿 -x方向飞，一边飞一边转 yaw
    生成机体坐标系视角下的点云图像，红箭头指示机头
    """
    import torch
    print("\n" + "="*72)
    print("[FLIGHT TEST] Env0: +X flight & yaw; Env1: -X flight & yaw")
    print("="*72)

    env0_id = torch.tensor([0], device=env.device)
    env1_id = torch.tensor([1], device=env.device)

    # 创建专属保存目录
    save_dir = Path(save_dir) / "flight_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 让环境先沉降几步稳定下来
    zero_act = torch.zeros((env.num_envs, 6), device=env.device)
    for _ in range(20):
        with torch.inference_mode():
            env.step(zero_act)

    # 2. 强制将无人机拉回到指定高度，避免撞地
    set_same_local_pose_for_envs(env, robot_asset, torch.tensor([0, 1], device=env.device), local_xyz=(0., 0., 5.), yaw_deg=0.0)

    yaw_rate = float(args_cli.diagnose_yaw_rate) # 例如 1.57 rad/s

    for step in range(max_steps):
        actions = torch.zeros((env.num_envs, 6), device=env.device)
        
        # Env 0: 沿机体系 x 方向 1.0m/s，偏航旋转 yaw_rate
        actions[0, 0] = 1.0
        actions[0, 5] = yaw_rate
        
        # Env 1: 沿机体系 -x 方向 -1.0m/s，偏航旋转 yaw_rate
        actions[1, 0] = -1.0
        actions[1, 5] = yaw_rate

        with torch.inference_mode():
            env.step(actions)

        # 每 5 步保存一次截图（大约 24 帧/秒的渲染速度，抽帧保存）
        if step % 5 == 0:
            quats = robot_asset.data.root_quat_w
            pos = robot_asset.data.root_pos_w

            for env_idx, env_ids_tensor in [(0, env0_id), (1, env1_id)]:
                # 提取降采样点云
                pts, _ = _get_downsampled_pc_np(env, lidar, env_ids_tensor, max_pts=5000)
                
                # 获取真实世界的姿态信息，仅用于在图片标题上提示当前状态
                q = quats[env_idx].cpu().numpy()
                w, x, y, z = q[0], q[1], q[2], q[3]
                yaw_deg = math.degrees(math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)))
                p = pos[env_idx].cpu().numpy()
                # p 是全局坐标，为了直观，我们减去原点得到相对 env 的坐标
                origins = _get_env_origins_np(env)
                local_p = p - origins[env_idx]

                title = (f"Env {env_idx} | Step {step:03d} | "
                         f"Pos_local:({local_p[0]:.1f}, {local_p[1]:.1f}) | Yaw:{yaw_deg:+.1f}°")
                
                filename = str(save_dir / f"env{env_idx}_step_{step:04d}.png")
                save_sensor_frame_xy_png(pts, filename, title=title)

            print(f"  Flight Test: Step {step:03d} visualizations saved.")

    print(f"\n[FLIGHT TEST] Finished. Sequence images saved to: {save_dir}")

# =============================================================================
# Diagnosis (Rotation & Scope)
# =============================================================================
def run_rotation_frame_diagnosis(env, lidar, robot_asset, save_dir, rot_wall_local_xy=(0.0, 10.0), phase1_steps=50, rotate_steps=100, phase2_steps=50, yaw_rate_rad=math.pi / 2.0, max_pts=10000, collect_every=2, min_xy_radius=0.5, hist_bin_deg=5.0):
    import torch; print("\n" + "="*72); print("[DIAGNOSE-ROT] Rotation diagnosis"); print("="*72)
    env0_ids = torch.tensor([0], device=env.device)
    def _get_yaw_deg(): q = robot_asset.data.root_quat_w[0].detach().cpu(); return math.degrees(math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3])))
    def _make_act(yaw_r=0.0): a = torch.zeros(env.num_envs, 6, device=env.device); a[:, 5] = yaw_r; return a
    def _collect_phase(n_steps, yaw_r, tag):
        phi_all, yaw_list, sample_pc = [], [], None
        for i in range(n_steps):
            with torch.inference_mode(): env.step(_make_act(yaw_r))
            if i % max(collect_every, 1) == 0:
                pts, _ = _get_downsampled_pc_np(env, lidar, env0_ids, max_pts); phi_deg, mean_phi, _ = _compute_phi_mean(pts, min_xy_radius); yaw = _get_yaw_deg()
                if phi_deg is not None:
                    phi_all.append(phi_deg); yaw_list.append(yaw); sample_pc = pts.copy() if sample_pc is None else sample_pc
                    print(f"  {tag} step={i:3d}: yaw={yaw:+.1f} deg, phi_mean={mean_phi:.1f} deg")
        phi_cat = np.concatenate(phi_all) if phi_all else None
        phi_mean = math.degrees(math.atan2(np.mean(np.sin(np.deg2rad(phi_cat))), np.mean(np.cos(np.deg2rad(phi_cat))))) % 360.0 if phi_cat is not None else float("nan")
        return phi_cat, phi_mean, float(np.mean(yaw_list)) if yaw_list else 0.0, sample_pc
    def _collect_rotate_traj(n_steps, yaw_r):
        traj, sample_pc = [], None
        for i in range(n_steps):
            with torch.inference_mode(): env.step(_make_act(yaw_r))
            if i % max(collect_every, 1) == 0:
                pts, _ = _get_downsampled_pc_np(env, lidar, env0_ids, max_pts); _, mean_phi, _ = _compute_phi_mean(pts, min_xy_radius); yaw = _get_yaw_deg()
                traj.append((yaw, mean_phi)); sample_pc = pts.copy() if sample_pc is None and pts is not None else sample_pc
                if i % 20 == 0: print(f"  rotate step={i:3d}: yaw={yaw:+.1f} deg, phi_mean={mean_phi:.1f} deg")
        return traj, sample_pc

    phi1_cat, phi_mean1, yaw_mean1, pc1 = _collect_phase(phase1_steps, 0.0, "phase1")
    rotate_traj, pc_mid = _collect_rotate_traj(rotate_steps, yaw_rate_rad)
    phi2_cat, phi_mean2, yaw_mean2, pc2 = _collect_phase(phase2_steps, 0.0, "phase2")
    delta_yaw = _circular_diff_deg(yaw_mean2, yaw_mean1)
    delta_phi = _circular_diff_deg(phi_mean2, phi_mean1) if math.isfinite(phi_mean1) and math.isfinite(phi_mean2) else float("nan")
    slope = float("nan"); valid_traj = [(y, p) for y, p in rotate_traj if math.isfinite(p)]
    if len(valid_traj) >= 4: res = np.linalg.lstsq(np.column_stack([np.array([t[0] for t in valid_traj]), np.ones(len(valid_traj))]), np.array([t[1] for t in valid_traj]), rcond=None); slope = float(res[0][0])
    frame = "unknown"; conclusion = "FAILED" if not math.isfinite(delta_phi) else "world-like" if (abs(delta_phi)<20 and abs(slope)<0.3) else "body-like" if (abs(delta_phi - (-delta_yaw))<20 and abs(slope+1)<0.4) else "ambiguous"
    if conclusion != "FAILED": frame = "world_like" if "world" in conclusion else "body_like" if "body" in conclusion else "unknown"
    print(f"[DIAGNOSE-ROT] Result: {frame.upper()} | delta_yaw={delta_yaw:+.1f}, delta_phi={delta_phi:+.1f}, slope={slope:+.3f}")
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    save_phi_histogram_png(phi1_cat, phi2_cat, str(save_dir / "rot_phi_histogram.png"), mean1=phi_mean1, mean2=phi_mean2, conclusion=conclusion, bin_deg=hist_bin_deg)
    _save_phi_vs_yaw_png(rotate_traj, str(save_dir / "rot_phi_vs_yaw.png"), slope=slope, frame=frame)
    for pc, tag in [(pc1, "phase1"), (pc_mid, "mid_rotate"), (pc2, "phase2")]:
        if pc is not None: save_xy_scatter_png(pc, str(save_dir / f"rot_{tag}_xy.png")); save_pointcloud_png(pc, str(save_dir / f"rot_{tag}_3d.png"))
    return {"frame": frame, "delta_yaw": delta_yaw, "delta_phi": delta_phi, "slope": slope}

def run_world_scope_diagnosis(env, lidar, robot_asset, save_dir, marker_global_xyz=(12., 6., 5.), marker_size_xyz=(3., 3., 10.), drone_local_xyz=(0., 0., 5.), settle_steps=30, max_pts=10000, aabb_margin=2.0):
    import torch; print("\n"+"="*72); print("[DIAGNOSE-SCOPE] global world vs local world"); print("="*72)
    if env.num_envs < 2: return {"scope_frame": "unknown"}
    o0 = _get_env_origins_np(env)[0].astype(np.float64); o1 = _get_env_origins_np(env)[1].astype(np.float64)
    expected = {"env0_global": np.asarray(marker_global_xyz), "env0_local": np.asarray(marker_global_xyz) - o0, "env1_global": np.asarray(marker_global_xyz), "env1_local": np.asarray(marker_global_xyz) - o1}
    set_same_local_pose_for_envs(env, robot_asset, torch.tensor([0, 1], device=env.device), drone_local_xyz, yaw_deg=0.0)
    for _ in range(settle_steps):
        with torch.inference_mode(): env.step(torch.zeros(env.num_envs, 6, device=env.device))
    pc0, _ = _get_downsampled_pc_np(env, lidar, torch.tensor([0], device=env.device), max_pts)
    pc1, _ = _get_downsampled_pc_np(env, lidar, torch.tensor([1], device=env.device), max_pts)
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True); results = {}
    half = np.array(marker_size_xyz)*0.5 + aabb_margin
    for env_i, pc in {"env0": pc0, "env1": pc1}.items():
        for hyp in ("global", "local"):
            key = f"{env_i}_{hyp}"; pts = _extract_points_in_aabb(pc, expected[key], half); c = _centroid_or_nan(pts)
            res = float(np.linalg.norm(c - expected[key])) if np.all(np.isfinite(c)) else float("inf")
            results[key] = {"pts": pts, "centroid": c, "residual": res, "n_hits": 0 if pts is None else len(pts)}
            print(f"  {key:20s}: hits={results[key]['n_hits']:5d}, res={res:.3f}")
    r1g, n1g = results["env1_global"]["residual"], results["env1_global"]["n_hits"]
    r1l, n1l = results["env1_local"]["residual"],  results["env1_local"]["n_hits"]
    scope_frame = "global_world" if n1g>=10 and r1g<2.0 and (n1l<10 or r1g<r1l) else "local_world" if n1l>=10 and r1l<2.0 else "unknown"
    print(f"[DIAGNOSE-SCOPE] Result: {scope_frame.upper()}")
    _save_scope_comparison_png(results, expected, o0, o1, np.asarray(marker_global_xyz), str(save_dir / "scope_comparison.png"), scope_frame, f"Result: {scope_frame}")
    return {"scope_frame": scope_frame}


# =============================================================================
# Launch Kit
# =============================================================================
args_cli = parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.optim as optim
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm, ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, SceneEntityCfg, TerminationTermCfg as DoneTerm, ActionTermCfg as ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import LidarSensorCfg
import sys

WORKSPACE_PATH = (Path.home() / "hjr_isaacdrone_ws" / "omniperception_isaacdrone" / "source" / "omniperception_isaacdrone")
if str(WORKSPACE_PATH) not in sys.path: sys.path.insert(0, str(WORKSPACE_PATH))

try:
    from omniperception_isaacdrone.assets.robots.drone_cfg import DRONE_CFG
    print("[INFO] DRONE_CFG imported successfully")
except ImportError as e:
    print(f"[ERROR] Cannot import drone_cfg: {e}"); simulation_app.close(); raise

try:
    from omniperception_isaacdrone.assets.sensors.lidar_cfg import LIDAR_CFG
    print("[INFO] LIDAR_CFG imported successfully")
except ImportError as e:
    print(f"[WARN] Cannot import lidar_cfg: {e}"); LIDAR_CFG = None

# =============================================================================
# Action / Managers / Envs
# =============================================================================
class RootTwistVelocityActionTerm(ActionTerm):
    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]; self._device = env.device; self._num_envs = env.num_envs
        self._raw_actions = torch.zeros((self._num_envs, 6), device=self._device)
        self._processed_actions = torch.zeros((self._num_envs, 6), device=self._device)
        p = getattr(cfg, "params", {}) or {}; pg = p.get if isinstance(p, dict) else lambda k, d=None: getattr(p, k, d)
        self._lin_scale, self._ang_scale = float(pg("lin_scale", args_cli.lin_vel_scale)), float(pg("ang_scale", args_cli.ang_vel_scale))
        self._lin_clip, self._ang_clip = float(pg("lin_clip", args_cli.lin_vel_clip)), float(pg("ang_clip", args_cli.ang_vel_clip))
    @property
    def action_dim(self): return 6
    @property
    def raw_actions(self): return self._raw_actions
    @property
    def processed_actions(self): return self._processed_actions
    def reset(self, env_ids=None):
        if env_ids is None: self._raw_actions.zero_(); self._processed_actions.zero_()
        else: self._raw_actions[env_ids] = 0.; self._processed_actions[env_ids] = 0.
    def process_actions(self, actions):
        if actions.device != self._device: actions = actions.to(self._device)
        self._raw_actions.copy_(actions)
        self._processed_actions[:, 0:3] = torch.clamp(actions[:, 0:3] * self._lin_scale, -self._lin_clip, self._lin_clip)
        self._processed_actions[:, 3:6] = torch.clamp(actions[:, 3:6] * self._ang_scale, -self._ang_clip, self._ang_clip)
    def apply_actions(self):
        self._asset.write_root_velocity_to_sim(self._processed_actions)

def reset_root_state_on_square_edge(env, env_ids, asset_cfg, square_half_size=35.0, z_range=(3., 7.)):
    asset = env.scene[asset_cfg.name]; n = len(env_ids); edges = torch.randint(0, 4, (n,), device=env.device)
    pos = torch.zeros((n, 3), device=env.device); ep = torch.rand(n, device=env.device) * 2 * square_half_size - square_half_size
    pos[edges==0, 0] = -square_half_size; pos[edges==0, 1] = ep[edges==0]; pos[edges==1, 0] = square_half_size; pos[edges==1, 1] = ep[edges==1]
    pos[edges==2, 0] = ep[edges==2]; pos[edges==2, 1] = -square_half_size; pos[edges==3, 0] = ep[edges==3]; pos[edges==3, 1] = square_half_size
    pos[:, 2] = torch.rand(n, device=env.device) * (z_range[1] - z_range[0]) + z_range[0]
    ori = torch.zeros((n, 4), device=env.device); ori[:, 0] = 1.0
    asset.write_root_pose_to_sim(torch.cat([pos, ori], dim=1), env_ids=env_ids); asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)

def reward_distance_to_goal(env, asset_cfg, std=5.0): return torch.exp(-((mdp.root_pos_w(env, asset_cfg=asset_cfg) - env.goal_pos_w)**2).sum(-1)/(2.*std*std))
def reward_height_tracking(env, asset_cfg, target_z=5.0, std=2.0): return torch.exp(-((mdp.root_pos_w(env, asset_cfg=asset_cfg)[:,2]-target_z)**2) / (2.*std*std))
def reward_stability(env, asset_cfg, lin_std=2., ang_std=6.): return torch.exp(-(mdp.base_lin_vel(env, asset_cfg=asset_cfg)**2).sum(-1)/(2.*lin_std**2)) * torch.exp(-(mdp.base_ang_vel(env, asset_cfg=asset_cfg)**2).sum(-1)/(2.*ang_std**2))
def reward_action_l2(env): a = env.action_manager.get_term("root_twist").raw_actions; return (a*a).sum(-1)
def termination_crash_or_oob(env, asset_cfg, min_height, world_bound): pos = mdp.root_pos_w(env, asset_cfg=asset_cfg); return (pos[:,2] < min_height) | (pos[:,0].abs() > world_bound) | (pos[:,1].abs() > world_bound)
def obs_goal_delta(env, asset_cfg):
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg); goal = getattr(env, "goal_pos_w", None)
    if goal is None: return torch.zeros_like(pos)
    goal = goal.to(pos.device); return (goal[:1].expand(pos.shape[0], 3) if goal.shape[0] != pos.shape[0] else goal) - pos
def obs_lidar_min_range_grid(env, lidar_name="lidar", theta_min=30., theta_max=90., phi_min=0., phi_max=360., delta_theta=1., delta_phi=5., empty_value=0., max_vis_points=None):
    t_bins, p_bins = max(int((theta_max - theta_min)/delta_theta), 1), max(int((phi_max - phi_min)/delta_phi), 1)
    out_shape = (env.num_envs, t_bins * p_bins)
    try: lidar = env.scene[lidar_name]
    except Exception: return torch.zeros(out_shape, device=env.device)
    pc, _ = _get_downsampled_pc_torch(env, lidar, torch.arange(env.num_envs, device=env.device), max_vis_points)
    if pc is None: return torch.zeros(out_shape, device=env.device)
    x, y, z = pc[...,0], pc[...,1], pc[...,2]
    valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z)
    r = torch.sqrt(x*x + y*y + z*z + 1e-12)
    theta = torch.rad2deg(torch.acos(torch.clamp(z/r, -1., 1.))); phi = torch.remainder(torch.rad2deg(torch.atan2(y, x)), 360.)
    m = valid & (theta >= theta_min) & (theta < theta_max) & (phi >= phi_min) & (phi < phi_max)
    bins = torch.full((env.num_envs, t_bins*p_bins), float("inf"), device=env.device)
    if m.any():
        t_idx = torch.clamp(torch.floor((theta - theta_min)/delta_theta).long(), 0, t_bins-1)
        p_idx = torch.clamp(torch.floor((phi - phi_min)/delta_phi).long(), 0, p_bins-1)
        lin = t_idx * p_bins + p_idx
        for e in range(env.num_envs):
            if m[e].any(): bins[e].scatter_reduce_(0, lin[e,m[e]], r[e,m[e]].float(), reduce="amin", include_self=True)
    return torch.where(torch.isfinite(bins), bins, torch.full_like(bins, float(empty_value)))

@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1, 
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply", 
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        ), debug_vis=True
    )
    
    # 1. 机器人的总体配置，必须指向根节点 /Robot
    robot: ArticulationCfg = DRONE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn = DRONE_CFG.spawn.replace(scale=(20, 20, 10), rigid_props=sim_utils.RigidBodyPropertiesCfg(enable_gyroscopic_forces=True))
    robot.init_state = ArticulationCfg.InitialStateCfg(pos=(0., 0., 5.), rot=(1., 0., 0., 0.), joint_pos={".*": 0.})
    
    # 2. LiDAR 的配置，附加在机器人的身体连杆 /Robot/body 上
    if LIDAR_CFG is not None and args_cli.enable_lidar: 
        lidar: LidarSensorCfg = LIDAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/body")
        
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=3000., color=(0.75, 0.75, 0.75)))
    distant_light = AssetBaseCfg(prim_path="/World/DistantLight", spawn=sim_utils.DistantLightCfg(intensity=3000., color=(0.9, 0.9, 0.9), angle=0.53), init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)))

@configclass
class ActionsCfg:
    root_twist = ActionTermCfg(class_type=RootTwistVelocityActionTerm, asset_name="robot")
    root_twist.params = {"lin_scale": args_cli.lin_vel_scale, "ang_scale": args_cli.ang_vel_scale, "lin_clip": args_cli.lin_vel_clip, "ang_clip": args_cli.ang_vel_clip}

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
        lidar_grid = ObsTerm(func=obs_lidar_min_range_grid, params={"lidar_name": "lidar", "theta_min": args_cli.lidar_theta_min, "theta_max": args_cli.lidar_theta_max, "phi_min": args_cli.lidar_phi_min, "phi_max": args_cli.lidar_phi_max, "delta_theta": args_cli.lidar_delta_theta, "delta_phi": args_cli.lidar_delta_phi, "empty_value": args_cli.lidar_empty_value, "max_vis_points": int(args_cli.lidar_max_vis_points)})
        def __post_init__(self): self.enable_corruption = False; self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg: reset_robot_base = EventTerm(func=reset_root_state_on_square_edge, mode="reset", params={"asset_cfg": SceneEntityCfg("robot"), "square_half_size": args_cli.square_half_size, "z_range": (args_cli.z_init_min, args_cli.z_init_max)})

@configclass
class RewardsCfg:
    dist_to_goal = RewTerm(func=reward_distance_to_goal, weight=10.0, params={"asset_cfg": SceneEntityCfg("robot"), "std": 6.0})
    height = RewTerm(func=reward_height_tracking, weight=2.0, params={"asset_cfg": SceneEntityCfg("robot"), "target_z": 5.0, "std": 2.5})
    stability = RewTerm(func=reward_stability, weight=1.5, params={"asset_cfg": SceneEntityCfg("robot"), "lin_std": 2.0, "ang_std": 6.0})
    action_l2 = RewTerm(func=reward_action_l2, weight=-0.01)
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    crash_or_oob = DoneTerm(func=termination_crash_or_oob, params={"asset_cfg": SceneEntityCfg("robot"), "min_height": args_cli.min_height, "world_bound": args_cli.world_bound})

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=0.0)
    observations: ObservationsCfg = ObservationsCfg(); actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg(); rewards: RewardsCfg = RewardsCfg(); terminations: TerminationsCfg = TerminationsCfg()
    def __post_init__(self):
        try: super().__post_init__()
        except Exception: pass
        self.decimation = 2; self.episode_length_s = 20.0
        self.viewer.eye = (60., 60., 40.); self.viewer.lookat = (0., 0., 5.)
        self.sim.dt = 1.0 / 120.0; self.sim.render_interval = self.decimation

class DiagnosticObstacleSpawner:
    ROT_WALL_LOCAL_X, ROT_WALL_LOCAL_Y, ROT_WALL_LOCAL_Z = 0.0, 10.0, 5.0
    ROT_WALL_SIZE_X, ROT_WALL_SIZE_Y, ROT_WALL_SIZE_Z = 12.0, 0.5, 10.0
    SCOPE_MARKER_GLOBAL_X, SCOPE_MARKER_GLOBAL_Y, SCOPE_MARKER_GLOBAL_Z = 12.0, 6.0, 5.0
    SCOPE_MARKER_SIZE_X, SCOPE_MARKER_SIZE_Y, SCOPE_MARKER_SIZE_Z = 3.0, 3.0, 10.0

    def __init__(self, num_background=15, x_range=(-30., 30.), y_range=(-30., 30.), xy_size_range=(0.5, 1.5), z_height=10.0, env_origins=None, seed=42):
        self.num_bg = num_background; self.x_range = x_range; self.y_range = y_range
        self.xy_size_range = xy_size_range; self.z_height = z_height
        self.env_origins = env_origins or [[0., 0., 0.]]
        if seed is not None: np.random.seed(seed)

    @staticmethod
    def _spawn_single(path, translation, size, color):
        cfg = sim_utils.CuboidCfg(size=size, rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True), collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
        cfg.func(path, cfg, translation=translation)

    def spawn_obstacles(self):
        import isaacsim.core.utils.prims as prim_utils
        prim_utils.create_prim("/World/Obstacles", "Xform")
        o0 = np.asarray(self.env_origins[0]); rw = (float(o0[0] + self.ROT_WALL_LOCAL_X), float(o0[1] + self.ROT_WALL_LOCAL_Y), float(self.ROT_WALL_LOCAL_Z))
        self._spawn_single("/World/Obstacles/RotDiag_Wall", rw, (self.ROT_WALL_SIZE_X, self.ROT_WALL_SIZE_Y, self.ROT_WALL_SIZE_Z), (1.0, 0.2, 0.2))
        sc = (self.SCOPE_MARKER_GLOBAL_X, self.SCOPE_MARKER_GLOBAL_Y, self.SCOPE_MARKER_GLOBAL_Z)
        self._spawn_single("/World/Obstacles/ScopeDiag_Marker", sc, (self.SCOPE_MARKER_SIZE_X, self.SCOPE_MARKER_SIZE_Y, self.SCOPE_MARKER_SIZE_Z), (0.1, 0.2, 1.0))
        for i in range(self.num_bg):
            for _ in range(50):
                x, y = np.random.uniform(*self.x_range), np.random.uniform(*self.y_range)
                if not (abs(x - rw[0]) < 8. and abs(y - rw[1]) < 6.) and not (abs(x - sc[0]) < 5. and abs(y - sc[1]) < 5.): break
            sx, sy = np.random.uniform(*self.xy_size_range), np.random.uniform(*self.xy_size_range)
            self._spawn_single(f"/World/Obstacles/BG_{i:04d}", (float(x), float(y), self.z_height / 2.), (float(sx), float(sy), self.z_height), tuple(np.random.uniform(0.3, 0.7, 3).tolist()))

class MyDroneRLEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: MyEnvCfg):
        self.goal_pos_w = torch.zeros((1, 3), dtype=torch.float32)
        super().__init__(cfg=cfg)
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._sample_goals(torch.arange(self.num_envs, device=self.device))
    def _sample_goals(self, env_ids):
        half = float(args_cli.square_half_size); n = env_ids.numel()
        self.goal_pos_w[env_ids, 0] = (torch.rand(n, device=self.device)*2-1)*half
        self.goal_pos_w[env_ids, 1] = (torch.rand(n, device=self.device)*2-1)*half
        self.goal_pos_w[env_ids, 2] = torch.rand(n, device=self.device) * (args_cli.goal_z_max - args_cli.goal_z_min) + args_cli.goal_z_min
    def reset_idx(self, env_ids=None):
        if env_ids is None: env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_goals(env_ids); obs, info = super().reset_idx(env_ids)
        try:
            pos = mdp.root_pos_w(self, asset_cfg=SceneEntityCfg("robot"))
            info["goal_delta"] = (self.goal_pos_w - pos).detach(); info["goal_pos_w"] = self.goal_pos_w.detach()
        except Exception: pass
        return obs, info

# =============================================================================
# main
# =============================================================================
def main():
    print("="*80)
    print("ManagerBasedRLEnv Drone -- LiDAR Frame Diagnosis & Flight Test")
    print("="*80)

    if (args_cli.diagnose_lidar_frame or args_cli.flight_test) and not args_cli.enable_lidar:
        print("[ERROR] Diagnostic or flight test requires --enable_lidar")
        simulation_app.close(); return

    env_cfg = MyEnvCfg()
    env_cfg.scene.num_envs = max(args_cli.num_envs, 2)  # 至少需要2个环境来跑 flight_test
    env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    env_cfg.sim.device = args_cli.device

    try: _ = float(env_cfg.decimation)
    except Exception:
        env_cfg.decimation = 2; env_cfg.episode_length_s = 20.
        env_cfg.sim.dt = 1./120.; env_cfg.sim.render_interval = 2

    # 1. 提前计算 Isaac Lab 的网格原点
    num_envs = env_cfg.scene.num_envs; spacing = env_cfg.scene.env_spacing
    num_cols = math.ceil(math.sqrt(num_envs)); num_rows = math.ceil(num_envs / num_cols)
    precalculated_origins = []
    for i in range(num_envs):
        row, col = i // num_cols, i % num_cols
        x, y = ((num_rows - 1) / 2.0 - row) * spacing, (col - (num_cols - 1) / 2.0) * spacing
        precalculated_origins.append([x, y, 0.0])

    # 2. 生成障碍物
    print("\n[Status] Spawning diagnostic obstacles...")
    DiagnosticObstacleSpawner(num_background=args_cli.num_obstacles, env_origins=precalculated_origins).spawn_obstacles()

    # 3. 初始化 RL 环境
    print("\n[Status] Creating RL env...")
    env = MyDroneRLEnv(cfg=env_cfg)
    print("[Status] Env created!")

    lidar = env.scene["lidar"] if args_cli.enable_lidar else None
    robot_asset = env.scene["robot"]

    print("\n[Status] Resetting env...")
    obs, info = env.reset()
    save_dir = Path(args_cli.lidar_save_dir).expanduser(); save_dir.mkdir(parents=True, exist_ok=True)

    # ── Flight Test Mode ──────────────────────────────────────────────
    if args_cli.flight_test:
        run_flight_test(env, lidar, robot_asset, save_dir, max_steps=200)
        env.close(); simulation_app.close(); return

    # ── Diagnosis mode ────────────────────────────────────────────────
    if args_cli.diagnose_lidar_frame:
        rot_result = run_rotation_frame_diagnosis(
            env=env, lidar=lidar, robot_asset=robot_asset, save_dir=save_dir,
            rot_wall_local_xy=(DiagnosticObstacleSpawner.ROT_WALL_LOCAL_X, DiagnosticObstacleSpawner.ROT_WALL_LOCAL_Y),
            phase1_steps=args_cli.diagnose_phase1_steps, rotate_steps=args_cli.diagnose_rotate_steps, phase2_steps=args_cli.diagnose_phase2_steps,
            yaw_rate_rad=args_cli.diagnose_yaw_rate, max_pts=int(args_cli.lidar_max_vis_points), collect_every=int(args_cli.diagnose_collect_every),
            min_xy_radius=float(args_cli.diagnose_min_xy_radius), hist_bin_deg=float(args_cli.diagnose_hist_bin_deg),
        )
        scope_result = run_world_scope_diagnosis(
            env=env, lidar=lidar, robot_asset=robot_asset, save_dir=save_dir,
            marker_global_xyz=(DiagnosticObstacleSpawner.SCOPE_MARKER_GLOBAL_X, DiagnosticObstacleSpawner.SCOPE_MARKER_GLOBAL_Y, DiagnosticObstacleSpawner.SCOPE_MARKER_GLOBAL_Z),
            marker_size_xyz=(DiagnosticObstacleSpawner.SCOPE_MARKER_SIZE_X, DiagnosticObstacleSpawner.SCOPE_MARKER_SIZE_Y, DiagnosticObstacleSpawner.SCOPE_MARKER_SIZE_Z),
            drone_local_xyz=(0., 0., 5.), settle_steps=args_cli.diagnose_scope_settle_steps, max_pts=int(args_cli.lidar_max_vis_points), aabb_margin=args_cli.diagnose_scope_aabb_margin,
        )
        env.close(); return

    # ── Free-run mode ─────────────────────────────────────────────────
    if not args_cli.train:
        saved_count = 0
        for count in range(args_cli.max_steps):
            if not simulation_app.is_running(): break
            with torch.inference_mode():
                actions = (torch.rand(env.num_envs, env.action_manager.total_action_dim, device=env.device) * 2 - 1)
                obs, rew, terminated, truncated, info = env.step(actions)
            if count % 50 == 0:
                pos0 = mdp.root_pos_w(env, asset_cfg=SceneEntityCfg("robot"))[0].cpu().numpy()
                print(f"[step {count}] pos={pos0}, rew={float(rew[0]):.3f}")
            if lidar is not None and count % args_cli.lidar_save_every == 0 and saved_count < args_cli.lidar_save_max:
                pts, _ = _get_downsampled_pc_np(env, lidar, torch.tensor([0], device=env.device), int(args_cli.lidar_max_vis_points))
                if save_pointcloud_png(pts, str(save_dir / f"lidar_pc_step_{count:06d}.png"), title=f"step={count}", s=2): saved_count += 1
        env.close(); return

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n[INFO] Interrupted by user")
    finally: simulation_app.close()
