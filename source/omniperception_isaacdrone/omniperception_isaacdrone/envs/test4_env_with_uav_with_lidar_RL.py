# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# =============================================================================
# 0) 标准库 & 与 Kit 无关的库
# =============================================================================
import argparse
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)
    # ax.set_zlim(-10, 10)
    ax.view_init(elev=25, azim=45)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


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
    ActionTermCfg as ActionTermCfg,   # <<< 新增
)

from isaaclab.managers.action_manager import ActionTerm  # <<< 新增：ActionTerm 基类

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# 可选 LiDAR（本文件内最小挂载）
from isaaclab.sensors import LidarSensorCfg

# 使用 Isaac Lab 自带 Crazyflie 配置（单文件不依赖外部 cfg）
import sys

# 你的外部项目路径（保持参考代码逻辑）
WORKSPACE_PATH = Path.home() / "hjr_isaacdrone_ws" / "omniperception_isaacdrone" / "source" / "omniperception_isaacdrone"
if str(WORKSPACE_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_PATH))
    print(f"[INFO] 添加路径到 Python path: {WORKSPACE_PATH}")

# 导入自定义无人机配置
try:
    from omniperception_isaacdrone.assets.robots.drone_cfg import DRONE_CFG
    print("[INFO] 成功导入自定义无人机配置: drone_cfg.DRONE_CFG")
except ImportError as e:
    print(f"[ERROR] 无法导入 drone_cfg: {e}")
    simulation_app.close()
    raise

# 导入自定义 LiDAR 配置（可选）
try:
    from omniperception_isaacdrone.assets.sensors.lidar_cfg import LIDAR_CFG
    print("[INFO] 成功导入自定义LiDAR配置: lidar_cfg.LIDAR_CFG")
except ImportError as e:
    print(f"[WARN] 无法导入 lidar_cfg: {e}")
    LIDAR_CFG = None


class RootTwistVelocityActionTerm(ActionTerm):
    """6D 动作: [vx, vy, vz, wx, wy, wz] 写入 articulation root velocity。

    raw_actions:      policy 原始输出（通常 [-1, 1]）
    processed_actions: 缩放+裁剪后的物理量（m/s, rad/s）
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset = env.scene[cfg.asset_name]
        self._device = env.device
        self._num_envs = env.num_envs

        # --------- 必须提供这两个缓存（满足 abstract property）---------
        self._raw_actions = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        self._processed_actions = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)

        # --------- 从 cfg.params 读取参数（推荐做法）---------
        # --------- 多路兼容读参数（适配不同 IsaacLab 版本）---------
        params = getattr(cfg, "params", None)
        if params is None:
            params = {}
        # dict-like
        if isinstance(params, dict):
            p_get = params.get
        else:
            # 有些版本 params 可能是 omegaconf 对象/Namespace
            p_get = lambda k, default=None: getattr(params, k, default)

        # 优先顺序：
        # 1) cfg.params['xxx']（如果存在）
        # 2) cfg.xxx（如果存在）
        # 3) args_cli.xxx（fallback）
        self._lin_scale = float(p_get("lin_scale", getattr(cfg, "lin_scale", args_cli.lin_vel_scale)))
        self._ang_scale = float(p_get("ang_scale", getattr(cfg, "ang_scale", args_cli.ang_vel_scale)))
        self._lin_clip  = float(p_get("lin_clip",  getattr(cfg, "lin_clip",  args_cli.lin_vel_clip)))
        self._ang_clip  = float(p_get("ang_clip",  getattr(cfg, "ang_clip",  args_cli.ang_vel_clip)))


    # IsaacLab ActionTerm 要求的动作维度
    @property
    def action_dim(self) -> int:
        return 6

    # --------- 这两个 property 是你当前缺失导致报错的关键 ---------
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
        """actions: (N,6)"""
        if actions.device != self._device:
            actions = actions.to(self._device)

        # 保存 raw（不做任何处理）
        # （注意：这里用 copy_ 避免张量引用问题）
        self._raw_actions.copy_(actions)

        # 缩放
        lin = actions[:, 0:3] * self._lin_scale
        ang = actions[:, 3:6] * self._ang_scale

        # 裁剪
        if self._lin_clip > 0:
            lin = torch.clamp(lin, -self._lin_clip, self._lin_clip)
        if self._ang_clip > 0:
            ang = torch.clamp(ang, -self._ang_clip, self._ang_clip)

        # 写入 processed_actions
        self._processed_actions[:, 0:3] = lin
        self._processed_actions[:, 3:6] = ang

    def apply_actions(self):
        # 直接写 root velocity: [vx,vy,vz, wx,wy,wz]
        self._asset.write_root_velocity_to_sim(self._processed_actions)


# =============================================================================
# 3) reset 逻辑（初始化在正方形边界）
# =============================================================================
def reset_root_state_on_square_edge(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    square_half_size: float = 35.0,
    z_range: tuple = (3.0, 7.0),
):
    """将无人机 root pose 随机初始化到 square 边界上，速度清零。"""
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


# =============================================================================
# 4) 自定义 reward / termination functions（与 RewardTermCfg/DoneTerm 配合）
# =============================================================================
def _get_root_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return mdp.root_pos_w(env, asset_cfg=asset_cfg)  # (N,3)


def reward_distance_to_goal(env: "MyDroneRLEnv", asset_cfg: SceneEntityCfg, std: float = 5.0) -> torch.Tensor:
    """exp(-||p-goal||^2 / (2*std^2))"""
    pos = _get_root_pos(env, asset_cfg)
    diff = pos - env.goal_pos_w
    dist2 = (diff * diff).sum(dim=-1)
    return torch.exp(-dist2 / (2.0 * std * std))


def reward_height_tracking(env: "MyDroneRLEnv", asset_cfg: SceneEntityCfg, target_z: float = 5.0, std: float = 2.0):
    pos = _get_root_pos(env, asset_cfg)
    dz2 = (pos[:, 2] - target_z) ** 2
    return torch.exp(-dz2 / (2.0 * std * std))


def reward_stability(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, lin_std: float = 2.0, ang_std: float = 6.0):
    lin = mdp.base_lin_vel(env, asset_cfg=asset_cfg)  # (N,3)
    ang = mdp.base_ang_vel(env, asset_cfg=asset_cfg)  # (N,3)
    lin2 = (lin * lin).sum(dim=-1)
    ang2 = (ang * ang).sum(dim=-1)
    return torch.exp(-lin2 / (2.0 * lin_std * lin_std)) * torch.exp(-ang2 / (2.0 * ang_std * ang_std))


def reward_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    # 优先惩罚 raw（未缩放、未裁剪）
    term = env.action_manager.get_term("root_twist")
    a = term.raw_actions
    return (a * a).sum(dim=-1)



def termination_crash_or_oob(env: "MyDroneRLEnv", asset_cfg: SceneEntityCfg, min_height: float, world_bound: float):
    pos = _get_root_pos(env, asset_cfg)
    crash = pos[:, 2] < min_height
    oob = (pos[:, 0].abs() > world_bound) | (pos[:, 1].abs() > world_bound)
    return crash | oob


# =============================================================================
# 5) goal 观测项（关键：把 goal 信息提供给 policy）
# =============================================================================
def obs_goal_delta(env: "MyDroneRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)  # (N,3) on env.device (cuda:0)
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




# =============================================================================
# 6) Scene / Env 配置（RL 版）
# =============================================================================
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """共享空间：plane terrain + robot + lights + (optional lidar)"""
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
    # 先构造（不带 params）
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
        # 机器人状态
        root_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        root_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})

        # 关键：目标相对量（让 agent “知道往哪飞”）
        goal_delta = ObsTerm(func=obs_goal_delta, params={"asset_cfg": SceneEntityCfg("robot")})

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
        # 兼容父类（不同版本可能有/没有）
        try:
            super().__post_init__()
        except Exception:
            pass

        # 基本仿真/控制设置
        self.decimation = 2
        self.episode_length_s = 20.0

        # viewer
        self.viewer.eye = (60.0, 60.0, 40.0)
        self.viewer.lookat = (0.0, 0.0, 5.0)

        # sim dt
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
# 9) 自定义 RLEnv：增加 goal buffer，并在 reset 时更新
# =============================================================================
class MyDroneRLEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: MyEnvCfg):
        # 占位：只能先放 CPU，因为 env.device 还没建立
        self.goal_pos_w = torch.zeros((1, 3), dtype=torch.float32)  # cpu

        super().__init__(cfg=cfg)

        # 现在 env.device 已存在（cuda:0），立刻迁移到 cuda:0 并按 num_envs 重新分配
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

        # 先采样 goal，保证 reset 后第一次观测就是新 goal
        self._sample_goals(env_ids)

        obs, info = super().reset_idx(env_ids)

        # debug extras
        try:
            pos = mdp.root_pos_w(self, asset_cfg=SceneEntityCfg("robot"))
            info["goal_delta"] = (self.goal_pos_w - pos).detach()
            info["goal_pos_w"] = self.goal_pos_w.detach()
        except Exception:
            pass
        return obs, info



# =============================================================================
# 10) 最小训练：A2C（on-policy，GAE）
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
    print("ManagerBasedRLEnv Drone - single file (Crazyflie) + minimal training")
    print("=" * 80)

    env_cfg = MyEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # ---- 兜底：如果 decimation 还是 MISSING，就补一次默认值 ----
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
    print(f"[配置] episode_length_s={float(env_cfg.episode_length_s):.1f}")

    print("\n[状态] 正在生成共享障碍物...")
    ObstacleSpawner(num_obstacles=args_cli.num_obstacles).spawn_obstacles()

    print("\n[状态] 正在创建 RL 环境 (ManagerBasedRLEnv)...")
    env = MyDroneRLEnv(cfg=env_cfg)
    print("[状态] 环境创建成功!")

    # -------------------------------------------------------------------------
    # LiDAR 初始化
    # -------------------------------------------------------------------------
    lidar = None

    if args_cli.enable_lidar:
        try:
            lidar = env.scene["lidar"]
            print("[INFO] LiDAR 已启用（scene['lidar']）")

            # 只有在需要可视化点云时才创建 USD points prim
        except Exception as e:
            print(f"[WARN] LiDAR 启用失败（prim_path/资产结构不匹配或未创建）：{e}")
            lidar = None

    print("\n[状态] reset...")
    obs, info = env.reset()
    print("[状态] reset 完成!")
    try:
        print(f"[DEBUG] policy obs shape: {tuple(obs['policy'].shape)}")
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # LiDAR 工具函数（main 内部封装，减少重复）
    # -------------------------------------------------------------------------
    env0_ids = torch.tensor([0], device=env.device)

    def _maybe_print_lidar_stats(step_i: int):
        """按频率打印 LiDAR hits/min/mean/max（只读 distances，便宜）"""
        if lidar is None:
            return
        if (step_i % args_cli.lidar_print_every) != 0:
            return
        try:
            d0 = lidar.get_distances(env0_ids)[0]
            max_d = float(getattr(lidar.cfg, "max_distance", 0.0))
            # max_d<=0 时退化为 finite 判断
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

    def _get_downsampled_pc_np(max_pts: int):
        """
        读取 env0 的 pointcloud，做下采样并转 numpy。
        返回 (points_np, num_raw)
        - points_np: (M,3) float numpy
        - num_raw: 原始点数
        """
        if lidar is None:
            return None, 0
        try:
            pc = lidar.get_pointcloud(env0_ids)  # (1,N,3) 或 (N,3)
            pc0 = pc[0] if pc.dim() == 3 else pc
            num_raw = int(pc0.shape[0])

            if num_raw == 0:
                return None, 0

            # 下采样
            # if max_pts is not None and max_pts > 0 and num_raw > max_pts:
            #     step = max(1, num_raw // max_pts)
            #     pc0 = pc0[::step]

            points_np = pc0.detach().cpu().numpy()
            # 过滤 NaN/Inf
            points_np = points_np[np.isfinite(points_np).all(axis=1)]
            if points_np.shape[0] == 0:
                return None, num_raw
            return points_np, num_raw
        except Exception:
            return None, 0



    def _maybe_save_pc_png(step_i: int, points_np, save_dir: Path, saved_count: int):
        """按频率保存 PNG，返回 updated saved_count"""
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

    # -------------------------------------------------------------------------
    # 非训练模式
    # -------------------------------------------------------------------------
    if not args_cli.train:
        save_dir = Path(args_cli.lidar_save_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_count = 0

        count = 0
        while simulation_app.is_running() and count < args_cli.max_steps:
            with torch.inference_mode():
                # 你当前是随机动作（[-1,1]），保持不变
                actions = torch.rand(env.num_envs, env.action_manager.total_action_dim, device=env.device) * 2 - 1
                obs, rew, terminated, truncated, info = env.step(actions)

                # 打印基础状态
                if (count % 50) == 0:
                    try:
                        pos0 = mdp.root_pos_w(env, asset_cfg=SceneEntityCfg("robot"))[0].detach().cpu().numpy()
                        goal0 = env.goal_pos_w[0].detach().cpu().numpy()
                        done0 = bool((terminated | truncated)[0].item())
                        print(f"[step {count}] pos0={pos0}, goal0={goal0}, rew0={float(rew[0]):.3f}, done0={done0}")
                    except Exception:
                        print(f"[step {count}] rew0={float(rew[0]):.3f}")

                # 1) LiDAR统计（读 distances）
                _maybe_print_lidar_stats(count)

                # 2) 点云（本帧如果要可视化或保存，最多读一次）
                need_pc = (lidar is not None) and (
                    ((count % args_cli.lidar_save_every == 0) and (saved_count < args_cli.lidar_save_max))
                )

                points_np = None
                if need_pc:
                    points_np, _ = _get_downsampled_pc_np(max_pts=int(args_cli.lidar_max_vis_points))


                # 4) 保存 PNG
                saved_count = _maybe_save_pc_png(count, points_np, save_dir, saved_count)

                count += 1

        print("\n[INFO] 非训练模式结束，关闭环境...")
        env.close()
        return

    # -------------------------------------------------------------------------
    # 训练模式（你原来的逻辑保持不变）
    # -------------------------------------------------------------------------
    obs_tensor = obs["policy"]
    obs_dim = obs_tensor.shape[-1]
    act_dim = env.action_manager.total_action_dim

    print("\n" + "=" * 80)
    print(f"[TRAIN] obs_dim={obs_dim}, act_dim={act_dim}, device={env.device}")
    print("=" * 80)

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
        ret = torch.zeros((T, N), device=env.device)

        gae = torch.zeros((N,), device=env.device)
        for t in reversed(range(T)):
            not_done = (~done_buf[t]).float()
            next_value = last_val if t == (T - 1) else val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_value * not_done - val_buf[t]
            gae = delta + gamma * gae_lam * not_done * gae
            adv[t] = gae
            ret[t] = adv[t] + val_buf[t]

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
# 12) entrypoint（关键修复：__main__）
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
