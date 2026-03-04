# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
测试无人机环境 - 在障碍物环境中飞行的无人机（共享障碍物）
无人机在正方形边界上初始化
使用外部配置文件 drone_cfg.py
"""

# ============ 第一步：启动仿真器 ============
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="测试无人机环境")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
parser.add_argument("--num_obstacles", type=int, default=50, help="障碍物数量（所有环境共享）")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动仿真应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============ 第二步：添加自定义包路径并导入模块 ============
# 添加自定义包到 Python 路径
WORKSPACE_PATH = Path.home() / "hjr_isaacdrone_ws" / "omniperception_isaacdrone" / "source" / "omniperception_isaacdrone"
if str(WORKSPACE_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_PATH))
    print(f"[INFO] 添加路径到 Python path: {WORKSPACE_PATH}")

import torch
import numpy as np
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

# ✅ 导入自定义无人机配置
try:
    from omniperception_isaacdrone.assets.robots.drone_cfg import DRONE_CFG
    print("[INFO] 成功导入自定义无人机配置: drone_cfg.DRONE_CFG")
except ImportError as e:
    print(f"[ERROR] 无法导入 drone_cfg: {e}")
    print(f"[ERROR] 请检查路径: {WORKSPACE_PATH}")
    simulation_app.close()
    sys.exit(1)


# ============ 自定义重置函数：在正方形边界上初始化 ============
def reset_root_state_on_square_edge(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    square_half_size: float = 35.0,
    z_range: tuple = (3.0, 7.0),
):
    """
    在正方形边界上随机重置机器人位置
    
    参数:
        env: 环境实例
        env_ids: 需要重置的环境ID
        asset_cfg: 资产配置
        square_half_size: 正方形半边长（默认35米，边长70米）
        z_range: 高度范围
    """
    # 获取资产
    asset: ArticulationCfg = env.scene[asset_cfg.name]
    num_resets = len(env_ids)
    
    # 为每个robot随机选择一条边
    # 0: 左边 (x = -35)
    # 1: 右边 (x = 35)
    # 2: 下边 (y = -35)
    # 3: 上边 (y = 35)
    edges = torch.randint(0, 4, (num_resets,), device=env.device)
    
    # 初始化位置张量 [num_resets, 3]
    positions = torch.zeros((num_resets, 3), device=env.device)
    
    # 随机生成沿边的位置参数 [-square_half_size, square_half_size]
    edge_positions = torch.rand(num_resets, device=env.device) * 2 * square_half_size - square_half_size
    
    # 根据选择的边设置x, y坐标
    left_mask = edges == 0
    right_mask = edges == 1
    bottom_mask = edges == 2
    top_mask = edges == 3
    
    # 左边: x = -square_half_size
    positions[left_mask, 0] = -square_half_size
    positions[left_mask, 1] = edge_positions[left_mask]
    
    # 右边: x = square_half_size
    positions[right_mask, 0] = square_half_size
    positions[right_mask, 1] = edge_positions[right_mask]
    
    # 下边: y = -square_half_size
    positions[bottom_mask, 0] = edge_positions[bottom_mask]
    positions[bottom_mask, 1] = -square_half_size
    
    # 上边: y = square_half_size
    positions[top_mask, 0] = edge_positions[top_mask]
    positions[top_mask, 1] = square_half_size
    
    # 随机z高度
    positions[:, 2] = torch.rand(num_resets, device=env.device) * (z_range[1] - z_range[0]) + z_range[0]
    
    # 创建四元数 (w, x, y, z) = (1, 0, 0, 0) 表示无旋转
    orientations = torch.zeros((num_resets, 4), device=env.device)
    orientations[:, 0] = 1.0  # w = 1
    
    # 合并位置和姿态 [num_resets, 7]
    root_states = torch.cat([positions, orientations], dim=1)
    
    # 设置根状态
    asset.write_root_pose_to_sim(root_states, env_ids=env_ids)
    
    # 重置速度为0 [num_resets, 6] (线速度3 + 角速度3)
    velocities = torch.zeros((num_resets, 6), device=env.device)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# ============ 第三步：定义环境配置 ============
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """无人机场景配置"""

    # 地面
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
        debug_vis=False,
    )

    # ========== ✅ 使用导入的无人机配置 ==========
    # 从 drone_cfg.py 导入基础配置
    robot: ArticulationCfg = DRONE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",  # 覆盖路径以支持多环境
    )
    
    # 如果需要修改其他参数，可以继续链式调用 replace()
    # 例如修改缩放、初始状态等：
    robot.spawn = DRONE_CFG.spawn.replace(
        scale=(20,20,10),
        # 可以在这里覆盖 spawn 参数
        # 例如修改刚体属性
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            enable_gyroscopic_forces=True,
        ),
    )
    
    # 覆盖初始状态（会被 reset 事件覆盖）
    robot.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    )
    # ==========================================

    # 光源
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75)
        ),
    )

    # 远距离光源
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(0.9, 0.9, 0.9),
            angle=0.53,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.738, 0.477, 0.477, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """动作配置"""
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
    )


@configclass
class ObservationsCfg:
    """观测配置"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测"""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=None)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=None)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=None)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件配置"""
    # ✅ 使用自定义函数在正方形边界上初始化
    reset_robot_base = EventTerm(
        func=reset_root_state_on_square_edge,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "square_half_size": 35.0,  # 半边长35米，总边长70米
            "z_range": (3.0, 7.0),     # 高度范围
        },
    )


@configclass
class MyEnvCfg(ManagerBasedEnvCfg):
    """无人机环境配置"""
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=0.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """后初始化"""
        self.sim
        self.decimation = 2  # 50Hz 控制频率
        self.episode_length_s = 20.0

        # 调整查看器位置
        self.viewer.eye = (60.0, 60.0, 40.0)
        self.viewer.lookat = (0.0, 0.0, 5.0)


# ============ 第四步：障碍物生成器（共享） ============
class ObstacleSpawner:
    """共享障碍物生成器类"""
    
    def __init__(
        self,
        num_obstacles: int = 50,
        x_range: tuple = (-33.0, 33.0),
        y_range: tuple = (-33.0, 33.0),
        xy_size_range: tuple = (0.5, 1.5),
        z_height: float = 10.0,
        seed: int = 42
    ):
        self.num_obstacles = num_obstacles
        self.x_range = x_range
        self.y_range = y_range
        self.xy_size_range = xy_size_range
        self.z_height = z_height
        
        if seed is not None:
            np.random.seed(seed)
    
    def spawn_obstacles(self):
        """生成共享障碍物（所有环境共用）"""
        import isaacsim.core.utils.prims as prim_utils
        
        # 创建障碍物父节点
        prim_utils.create_prim("/World/Obstacles", "Xform")
        
        print(f"\n[INFO]: 正在生成 {self.num_obstacles} 个共享障碍物...")
        
        for i in range(self.num_obstacles):
            # 在全局坐标系中随机位置
            x_pos = np.random.uniform(self.x_range[0], self.x_range[1])
            y_pos = np.random.uniform(self.y_range[0], self.y_range[1])
            z_pos = self.z_height / 2.0
            
            # 随机尺寸
            x_size = np.random.uniform(self.xy_size_range[0], self.xy_size_range[1])
            y_size = np.random.uniform(self.xy_size_range[0], self.xy_size_range[1])
            z_size = self.z_height
            
            # 随机颜色
            color = (
                np.random.uniform(0.3, 0.7),
                np.random.uniform(0.3, 0.7),
                np.random.uniform(0.3, 0.7)
            )
            
            # 创建障碍物配置
            cfg_obstacle = sim_utils.MeshCuboidCfg(
                size=(x_size, y_size, z_size),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color
                ),
            )
            
            # 生成障碍物
            obstacle_path = f"/World/Obstacles/Obstacle_{i:04d}"
            cfg_obstacle.func(obstacle_path, cfg_obstacle, translation=(x_pos, y_pos, z_pos))
            
            if (i + 1) % 10 == 0:
                print(f"[INFO]: 已生成 {i + 1}/{self.num_obstacles} 个障碍物")
        
        print(f"[INFO]: 共享障碍物生成完成！")


# ============ 第五步：主函数 ============
def main():
    """主函数"""
    print("=" * 80)
    print("无人机障碍物环境测试 - 使用外部配置文件")
    print("=" * 80)
    
    # 创建环境配置
    env_cfg = MyEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    print(f"\n[配置] 环境数量: {env_cfg.scene.num_envs}")
    print(f"[配置] 环境间距: {env_cfg.scene.env_spacing} m (共享空间)")
    print(f"[配置] 仿真频率: {1/env_cfg.sim.dt:.0f} Hz")
    print(f"[配置] 控制频率: {1/(env_cfg.sim.dt * env_cfg.decimation):.0f} Hz")
    print(f"[配置] Episode长度: {env_cfg.episode_length_s} 秒")
    print(f"[配置] 共享障碍物数量: {args_cli.num_obstacles}")
    print(f"[配置] 初始化位置: 70m×70m正方形边界上 (边界: x,y ∈ {{-35, 35}})")
    print(f"[配置] 初始化高度: Z[3m, 7m]")
    print(f"[配置] 无人机配置来源: drone_cfg.DRONE_CFG")
    print(f"[配置] USD路径: {DRONE_CFG.spawn.usd_path}")

    # 创建环境
    print("\n[状态] 正在创建环境...")
    env = ManagerBasedEnv(cfg=env_cfg)
    print("[状态] 环境创建成功!")
    
    # 验证robot数量
    robot = env.scene["robot"]
    print(f"\n[验证] Robot实例数: {robot.num_instances}")
    print(f"[验证] Robot根状态形状: {robot.data.root_state_w.shape}")
    
    # ========== 打印机器人结构信息 ==========
    print("\n" + "=" * 80)
    print("机器人结构信息")
    print("=" * 80)
    print(f"关节数量: {robot.num_joints}")
    print(f"关节名称: {robot.joint_names}")
    print(f"身体数量: {robot.num_bodies}")
    print(f"身体名称: {robot.body_names}")
    
    # 打印质量信息
    if hasattr(robot, 'root_physx_view'):
        try:
            masses = robot.root_physx_view.get_masses()
            print(f"\n根链接质量: {masses[0].item():.6f} kg")
            
            inertias = robot.root_physx_view.get_inertias()
            print(f"根链接惯性张量 (对角线):")
            print(f"  Ixx={inertias[0, 0].item():.6f}, Iyy={inertias[0, 4].item():.6f}, Izz={inertias[0, 8].item():.6f}")
        except Exception as e:
            print(f"[警告] 无法获取物理属性: {e}")
    
    print("=" * 80 + "\n")
    # =====================================
    
    # 生成共享障碍物
    print("\n[状态] 正在生成共享障碍物...")
    obstacle_spawner = ObstacleSpawner(
        num_obstacles=args_cli.num_obstacles,
        x_range=(-33.0, 33.0),      # 略小于边界范围（避免与边界重叠）
        y_range=(-33.0, 33.0),      # 略小于边界范围
        xy_size_range=(0.5, 1.5),   # 障碍物尺寸
        z_height=10.0,
        seed=42
    )
    obstacle_spawner.spawn_obstacles()

    # 重置环境
    print("\n[状态] 正在重置环境...")
    obs, _ = env.reset()
    print(f"[状态] 环境重置完成!")
    print(f"[信息] 观测形状: {obs['policy'].shape}")
    print(f"[信息] 动作维度: {env.action_manager.total_action_dim}")
    
    # 打印每个robot的初始位置（验证是否在边界上）
    print(f"\n[验证] 各Robot初始位置（全局坐标）:")
    root_pos = robot.data.root_state_w[:, :3].cpu().numpy()
    for i in range(env_cfg.scene.num_envs):
        x, y, z = root_pos[i]
        # 判断在哪条边上
        if abs(x + 35.0) < 0.1:
            edge = "左边 (x=-35)"
        elif abs(x - 35.0) < 0.1:
            edge = "右边 (x=+35)"
        elif abs(y + 35.0) < 0.1:
            edge = "下边 (y=-35)"
        elif abs(y - 35.0) < 0.1:
            edge = "上边 (y=+35)"
        else:
            edge = "未知位置"
        print(f"  Robot {i}: X={x:7.2f}, Y={y:7.2f}, Z={z:6.2f} - {edge}")
    
    # 验证是否真的在边界上
    if env_cfg.scene.num_envs > 0:
        on_boundary_count = 0
        for i in range(env_cfg.scene.num_envs):
            x, y, z = root_pos[i]
            if abs(abs(x) - 35.0) < 0.1 or abs(abs(y) - 35.0) < 0.1:
                on_boundary_count += 1
        print(f"\n[验证] 在边界上的Robot数量: {on_boundary_count}/{env_cfg.scene.num_envs}")
        if on_boundary_count == env_cfg.scene.num_envs:
            print("[验证] ✓ 所有robot都正确初始化在边界上")
        else:
            print("[验证] ✗ 警告：部分robot未在边界上")

    # 运行仿真
    print("\n" + "=" * 80)
    print("开始仿真 (按 ESC 或 Q 键退出)")
    print("=" * 80)

    count = 0
    max_iterations = 2000

    while simulation_app.is_running() and count < max_iterations:
        with torch.inference_mode():
            # 零动作（让无人机受重力影响自然下落）
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

            # 执行动作
            obs, extras = env.step(actions)

            # 每50步打印一次
            if count % 50 == 0:
                print(f"\n步数: {count}/{max_iterations}")
                
                # 打印第一个环境的观测
                if obs['policy'].shape[1] >= 9:
                    base_lin_vel = obs['policy'][0, 0:3].cpu().numpy()
                    base_ang_vel = obs['policy'][0, 3:6].cpu().numpy()
                    proj_gravity = obs['policy'][0, 6:9].cpu().numpy()
                    print(f"  [Robot 0] 线速度:   [{base_lin_vel[0]:6.2f}, {base_lin_vel[1]:6.2f}, {base_lin_vel[2]:6.2f}]")
                    print(f"  [Robot 0] 角速度:   [{base_ang_vel[0]:6.2f}, {base_ang_vel[1]:6.2f}, {base_ang_vel[2]:6.2f}]")
                    print(f"  [Robot 0] 重力投影: [{proj_gravity[0]:6.2f}, {proj_gravity[1]:6.2f}, {proj_gravity[2]:6.2f}]")
                
                # 每100步打印所有robot位置
                if count % 100 == 0 and env.num_envs > 1:
                    root_pos = robot.data.root_state_w[:, :3].cpu().numpy()
                    print(f"\n  所有Robot当前位置:")
                    for i in range(min(env.num_envs, 5)):  # 最多显示5个
                        x, y, z = root_pos[i]
                        print(f"    Robot {i}: X={x:7.2f}, Y={y:7.2f}, Z={z:6.2f}")
                
                # 打印超时信息
                if "time_outs" in extras:
                    num_timeouts = extras['time_outs'].sum().item()
                    if num_timeouts > 0:
                        print(f"  超时: {num_timeouts}/{env.num_envs}")

            count += 1

            # 检查是否需要重置
            if "time_outs" in extras and extras["time_outs"].any():
                print("\n[状态] 有Robot超时，正在重置...")
                timeout_ids = extras["time_outs"].nonzero(as_tuple=True)[0].cpu().numpy()
                print(f"[状态] 超时的Robot ID: {timeout_ids}")
                
                # 打印重置后的位置（验证是否在边界上）
                if count % 200 == 0:
                    root_pos = robot.data.root_state_w[:, :3].cpu().numpy()
                    print(f"\n[验证] 重置后的位置:")
                    for robot_id in timeout_ids[:3]:  # 最多显示3个
                        x, y, z = root_pos[robot_id]
                        if abs(x + 35.0) < 0.1:
                            edge = "左边"
                        elif abs(x - 35.0) < 0.1:
                            edge = "右边"
                        elif abs(y + 35.0) < 0.1:
                            edge = "下边"
                        elif abs(y - 35.0) < 0.1:
                            edge = "上边"
                        else:
                            edge = "?"
                        print(f"  Robot {robot_id}: X={x:7.2f}, Y={y:7.2f}, Z={z:6.2f} - {edge}")
                
                count = 0

    # 关闭环境
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
