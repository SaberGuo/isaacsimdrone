# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
测试无人机环境 - 在障碍物环境中飞行的无人机（共享障碍物）
无人机在正方形边界上初始化
使用外部配置文件 drone_cfg.py

修复点：
- 关键：先启动 AppLauncher/Kit，再 import 会触发 pxr 的 isaaclab 模块（否则 conda python 下会报 No module named 'pxr'）
- 梳理 LiDAR 点云刷新/保存逻辑，去掉重复嵌套
"""

# ============ 0) 标准库 & 仅与 Kit 无关的库 ============
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无GUI也能保存PNG
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="测试无人机环境")
    parser.add_argument("--num_envs", type=int, default=4, help="环境数量")
    parser.add_argument("--num_obstacles", type=int, default=50, help="障碍物数量（所有环境共享）")

    parser.add_argument("--enable_lidar", action="store_true", help="启用激光雷达")
    parser.add_argument("--lidar_debug_vis", action="store_true", help="显示LiDAR射线(调试)")
    parser.add_argument("--lidar_print_every", type=int, default=50, help="每隔多少步打印一次LiDAR")
    parser.add_argument("--lidar_vis_every", type=int, default=5, help="每隔多少步刷新一次点云可视化")
    parser.add_argument("--lidar_max_vis_points", type=int, default=2000, help="点云可视化最大点数")

    parser.add_argument("--lidar_save_every", type=int, default=50, help="每隔多少步保存一次点云PNG")
    parser.add_argument("--lidar_save_max", type=int, default=20, help="最多保存多少张PNG")
    parser.add_argument("--lidar_point_size", type=float, default=0.15, help="USD点云可视化点大小(米)")
    parser.add_argument("--lidar_save_dir", type=str, default=str(Path.home() / "lidar_pc_images"),
                        help="点云PNG保存目录(默认~/lidar_pc_images)")

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


# ============ 1) 启动 Kit（关键：必须在 isaaclab/mdp 等 import 之前） ============
args_cli = parse_args()

from isaaclab.app import AppLauncher  # 这里再 import 一次没关系
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ============ 2) 现在再 import isaaclab/torch/pxr 相关 ============
import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import LidarSensorCfg

# 你的外部项目路径（保持你原来的逻辑）
WORKSPACE_PATH = Path.home() / "hjr_isaacdrone_ws" / "omniperception_isaacdrone" / "source" / "omniperception_isaacdrone"
if str(WORKSPACE_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_PATH))
    print(f"[INFO] 添加路径到 Python path: {WORKSPACE_PATH}")

# 导入自定义配置
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


# ============ 3) Reset 逻辑 ============
def reset_root_state_on_square_edge(
    env: ManagerBasedEnv,
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
    orientations[:, 0] = 1.0  # w

    root_states = torch.cat([positions, orientations], dim=1)
    asset.write_root_pose_to_sim(root_states, env_ids=env_ids)

    velocities = torch.zeros((num_resets, 6), device=env.device)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# ============ 4) Scene / Env 配置 ============
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

    if LIDAR_CFG is not None:
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
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=None)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=None)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=None)

        def __post_init__(self):
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
            "square_half_size": 35.0,
            "z_range": (3.0, 7.0),
        },
    )


@configclass
class MyEnvCfg(ManagerBasedEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=0.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.sim
        self.decimation = 2
        self.episode_length_s = 20.0
        self.viewer.eye = (60.0, 60.0, 40.0)
        self.viewer.lookat = (0.0, 0.0, 5.0)


# ============ 5) 障碍物生成 ============
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

            cfg_obstacle = sim_utils.MeshCuboidCfg(
                size=(x_size, y_size, z_size),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

            obstacle_path = f"/World/Obstacles/Obstacle_{i:04d}"
            cfg_obstacle.func(obstacle_path, cfg_obstacle, translation=(x_pos, y_pos, z_pos))

            if (i + 1) % 10 == 0:
                print(f"[INFO]: 已生成 {i + 1}/{self.num_obstacles} 个障碍物")

        print("[INFO]: 共享障碍物生成完成！")


# ============ 6) LiDAR 点云 USD prim 初始化/刷新 ============
def init_usd_points_prim():
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    path = "/World/LidarPointCloud"
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        prim = stage.DefinePrim(path, "Points")

    usd_points = UsdGeom.Points(prim)
    usd_points.CreateDisplayColorAttr().Set([(0.1, 1.0, 0.1)])
    return usd_points


def update_usd_points(usd_points, points_np, point_size_m: float):
    from pxr import Gf, UsdGeom

    if points_np is None:
        return
    points_np = points_np[np.isfinite(points_np).all(axis=1)]
    pts = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points_np]
    usd_points.GetPointsAttr().Set(pts)
    usd_points.GetWidthsAttr().Set([float(point_size_m)] * len(pts))
    UsdGeom.Imageable(usd_points.GetPrim()).MakeVisible()


def main():
    print("=" * 80)
    print("无人机障碍物环境测试 - 使用外部配置文件")
    print("=" * 80)

    env_cfg = MyEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    print(f"\n[配置] 环境数量: {env_cfg.scene.num_envs}")
    print(f"[配置] 环境间距: {env_cfg.scene.env_spacing} m (共享空间)")
    print(f"[配置] 仿真频率: {1 / env_cfg.sim.dt:.0f} Hz")
    print(f"[配置] 控制频率: {1 / (env_cfg.sim.dt * env_cfg.decimation):.0f} Hz")
    print(f"[配置] Episode长度: {env_cfg.episode_length_s} 秒")
    print(f"[配置] 共享障碍物数量: {args_cli.num_obstacles}")
    print(f"[配置] USD路径: {DRONE_CFG.spawn.usd_path}")

    print("\n[状态] 正在生成共享障碍物...")
    ObstacleSpawner(num_obstacles=args_cli.num_obstacles).spawn_obstacles()

    print("\n[状态] 正在创建环境...")
    env = ManagerBasedEnv(cfg=env_cfg)
    print("[状态] 环境创建成功!")

    lidar = None
    usd_points = None

    if args_cli.enable_lidar:
        try:
            lidar = env.scene["lidar"]
            lidar.cfg.debug_vis = bool(args_cli.lidar_debug_vis)

            print("\n" + "=" * 80)
            print("LiDAR 初始化信息")
            print("=" * 80)
            print(f"LiDAR prim_path: {lidar.cfg.prim_path}")
            print(f"LiDAR max_distance: {lidar.cfg.max_distance} m")
            print(f"LiDAR min_range: {lidar.cfg.min_range} m")
            print(f"Return pointcloud: {lidar.cfg.return_pointcloud}")
            print(f"Pointcloud world frame: {lidar.cfg.pointcloud_in_world_frame}")
            print(f"Mesh prim paths: {lidar.cfg.mesh_prim_paths}")
            print(f"Debug rays: {lidar.cfg.debug_vis}")
            print("=" * 80 + "\n")

            if lidar.cfg.return_pointcloud:
                usd_points = init_usd_points_prim()
                print("[INFO] 已创建 /World/LidarPointCloud 用于点云可视化")

        except KeyError:
            print("[WARN] 你启用了 --enable_lidar，但 SceneCfg 里没有 lidar 字段或名称不匹配")
            lidar = None

    print("\n[状态] 正在重置环境...")
    obs, _ = env.reset()
    print("[状态] 环境重置完成!")

    save_dir = Path(args_cli.lidar_save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    print(f"[INFO] LiDAR PNG 将保存到: {save_dir}")

    count = 0
    max_iterations = 2000

    while simulation_app.is_running() and count < max_iterations:
        with torch.inference_mode():
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            obs, extras = env.step(actions)

            if lidar is not None and (count % args_cli.lidar_print_every == 0):
                env_ids = torch.tensor([0], device=env.device)
                d0 = lidar.get_distances(env_ids)[0]
                max_d = float(lidar.cfg.max_distance)
                hit_mask = d0 < max_d
                hit_count = int(hit_mask.sum().item())
                total = int(d0.numel())
                if hit_count > 0:
                    d_hit = d0[hit_mask]
                    print(f"  [LiDAR] rays={total}, hits={hit_count}, "
                          f"min={d_hit.min().item():.2f}m, mean={d_hit.mean().item():.2f}m, max={d_hit.max().item():.2f}m")
                else:
                    print(f"  [LiDAR] rays={total}, hits=0")

            # ---- 点云刷新（USD）----
            if lidar is not None and usd_points is not None and (count % args_cli.lidar_vis_every == 0):
                env_ids = torch.tensor([0], device=env.device)
                pc = lidar.get_pointcloud(env_ids)
                pc0 = pc[0] if pc.dim() == 3 else pc
                # 注意：这里不再强制过滤 hits，保持更“密”的显示（与你在视窗看到的更接近）
                max_pts = int(args_cli.lidar_max_vis_points)
                if pc0.shape[0] > max_pts:
                    step = max(1, pc0.shape[0] // max_pts)
                    pc0 = pc0[::step]
                points_np = pc0.detach().cpu().numpy()
                update_usd_points(usd_points, points_np, args_cli.lidar_point_size)

            # ---- 保存 PNG ----
            if lidar is not None and (count % args_cli.lidar_save_every == 0) and (saved_count < args_cli.lidar_save_max):
                env_ids = torch.tensor([0], device=env.device)
                pc = lidar.get_pointcloud(env_ids)
                pc0 = pc[0] if pc.dim() == 3 else pc

                max_pts = int(args_cli.lidar_max_vis_points)
                if pc0.shape[0] > max_pts:
                    step = max(1, pc0.shape[0] // max_pts)
                    pc0 = pc0[::step]

                points_np = pc0.detach().cpu().numpy()
                points_np = points_np[np.isfinite(points_np).all(axis=1)]

                png_path = save_dir / f"lidar_pc_step_{count:06d}.png"
                ok = save_pointcloud_png(points_np, str(png_path), title=f"LiDAR pointcloud step={count}", s=2)
                if ok:
                    saved_count += 1
                    print(f"[INFO] 已保存点云PNG: {png_path}")

            count += 1

    print("\n" + "=" * 80)
    print("仿真结束，正在关闭...")
    print("=" * 80)
    env.close()


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
