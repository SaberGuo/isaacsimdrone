# omniperception_isaacdrone/envs/test6_env_cfg.py
from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    ActionTermCfg as ActionTermCfg,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import LidarSensorCfg, ContactSensorCfg

from omniperception_isaacdrone.assets.robots.drone_cfg import DRONE_CFG, DRONE_MASS

try:
    from omniperception_isaacdrone.assets.sensors.lidar_cfg import LIDAR_CFG
except Exception:
    LIDAR_CFG = None

from omniperception_isaacdrone.tasks.mdp import (
    RootTwistVelocityActionTerm,
    reset_root_state_on_square_edge,
    obs_goal_delta,
    obs_lidar_min_range_grid,
    reward_distance_to_goal,
    reward_height_tracking,
    reward_stability,
    reward_velocity_towards_goal,
    penalty_lidar_threat,
    penalty_energy,
    reward_goal_reached,
    penalty_out_of_workspace,
    penalty_time_out,
    penalty_collision,
    reward_action_l2,
    termination_reached_goal,
    termination_out_of_workspace,
    termination_collision,
)


# -----------------------------------------------------------------------------
# SceneCfg
# -----------------------------------------------------------------------------
@configclass
class Test6SceneCfg(InteractiveSceneCfg):
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

    robot: ArticulationCfg = DRONE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 关键：不要引入任何临时变量（会被configclass当字段）！！
    robot.spawn = DRONE_CFG.spawn.replace(
        scale=(1, 1, 1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        activate_contact_sensors=True,  # ✅ 必须开启，否则 ContactSensor 读不到碰撞报告
    )

    robot.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    )

    # ✅ 重要修改：
    # 由于你障碍物是全局共享（/World/Obstacles 下 50 个），在 replicate_physics + GPU contact filter 下，
    # ContactSensor 的 filter_prim_paths_expr 会触发 PhysX tensors 的 pattern 数量不匹配错误：
    #   expected num_envs (32) but found 50
    # 因此这里禁用过滤，让 ContactSensor 只提供 net_forces_w（你的 termination_collision 有 fallback）
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[],   # ✅ 禁用过滤（避免 expected 32 found 50）
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
class Test6SceneWithLidarCfg(Test6SceneCfg):
    if LIDAR_CFG is not None:
        lidar: LidarSensorCfg = LIDAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# -----------------------------------------------------------------------------
# Actions / Obs / Events / Rewards / Terminations
# -----------------------------------------------------------------------------
@configclass
class Test6ActionsCfg:
    root_twist = ActionTermCfg(
        class_type=RootTwistVelocityActionTerm,
        asset_name="robot",
    )


@configclass
class Test6ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        root_pos          = ObsTerm(func=mdp.root_pos_w,           params={"asset_cfg": SceneEntityCfg("robot")})
        root_quat         = ObsTerm(func=mdp.root_quat_w,          params={"asset_cfg": SceneEntityCfg("robot")})
        root_lin_vel      = ObsTerm(func=mdp.root_lin_vel_w,       params={"asset_cfg": SceneEntityCfg("robot")})
        root_ang_vel      = ObsTerm(func=mdp.root_ang_vel_w,       params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity,    params={"asset_cfg": SceneEntityCfg("robot")})
        goal_delta        = ObsTerm(func=obs_goal_delta,           params={"asset_cfg": SceneEntityCfg("robot")})
        lidar_grid        = ObsTerm(func=obs_lidar_min_range_grid, params={})

        def post_init(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Test6EventCfg:
    reset_robot_base = EventTerm(func=reset_root_state_on_square_edge, mode="reset", params={})


@configclass
class Test6RewardsCfg:
    dist_to_goal      = RewTerm(func=reward_distance_to_goal,      weight=10.0,  params={})
    vel_towards_goal  = RewTerm(func=reward_velocity_towards_goal, weight=2.0,   params={})
    height            = RewTerm(func=reward_height_tracking,       weight=2.0,   params={})
    stability         = RewTerm(func=reward_stability,             weight=1.5,   params={})

    lidar_threat      = RewTerm(func=penalty_lidar_threat,         weight=-1.0,  params={})
    energy            = RewTerm(func=penalty_energy,               weight=-0.05, params={})
    action_l2         = RewTerm(func=reward_action_l2,             weight=-0.01)

    success_bonus     = RewTerm(func=reward_goal_reached,          weight=200.0,  params={})
    collision_penalty = RewTerm(func=penalty_collision,            weight=-200.0, params={})
    oob_penalty       = RewTerm(func=penalty_out_of_workspace,     weight=-150.0, params={})
    timeout_penalty   = RewTerm(func=penalty_time_out,             weight=-100.0,  params={})


@configclass
class Test6TerminationsCfg:
    time_out     = DoneTerm(func=mdp.time_out, time_out=True)
    reached_goal = DoneTerm(func=termination_reached_goal, params={})
    oob          = DoneTerm(func=termination_out_of_workspace, params={})
    collision    = DoneTerm(func=termination_collision, params={})


@configclass
class Test6DroneEnvCfg(ManagerBasedRLEnvCfg):
    scene:        Test6SceneCfg        = Test6SceneCfg(num_envs=1, env_spacing=0.0)
    observations: Test6ObservationsCfg = Test6ObservationsCfg()
    actions:      Test6ActionsCfg      = Test6ActionsCfg()
    events:       Test6EventCfg        = Test6EventCfg()
    rewards:      Test6RewardsCfg      = Test6RewardsCfg()
    terminations: Test6TerminationsCfg = Test6TerminationsCfg()

    def __post_init__(self):
        try:
            super().__post_init__()
        except Exception:
            pass

        self.decimation = 2
        max_steps = 500
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = float(max_steps) * float(self.sim.dt) * float(self.decimation)

        self.viewer.eye    = (60.0, 60.0, 40.0)
        self.viewer.lookat = (0.0,  0.0,  5.0)

        # Controller params
        self.actions.root_twist.params = {
            "mass":                DRONE_MASS,
            "use_sim_total_mass":  True,
            "prevent_negative_thrust": True,

            "vel_scale":           3.0,
            "vel_clip":            5.0,
            "yaw_rate_scale":      2.0,
            "yaw_rate_clip":       3.0,
            "thrust_sign":         1.0,
            "g":                   9.81,
            "vel_gain":            (3.0, 3.0, 4.0),
            "pos_gain":            (0.0, 0.0, 0.0),
            "attitude_gain":       (6.0, 6.0, 1.5),
            "ang_rate_gain":       (0.25, 0.25, 0.18),
            "thrust_limit_factor": 3,
            "torque_limit":        (200.0, 200.0, 200.0),
            "inertia_diag":        (0.02, 0.02, 0.04),
        }

        self.events.reset_robot_base.params = {
            "asset_cfg":        SceneEntityCfg("robot"),
            "square_half_size": 35.0,
            "z_range":          (3.0, 7.0),
        }

        GOAL_RADIUS = 1.0
        self.terminations.reached_goal.params = {
            "asset_cfg":  SceneEntityCfg("robot"),
            "threshold":  GOAL_RADIUS,
        }
        self.terminations.oob.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "x_bounds":  (-60.0, 60.0),
            "y_bounds":  (-60.0, 60.0),
            "z_bounds":  (1.0,   10.0),
        }
        self.terminations.collision.params = {
            "sensor_cfg": SceneEntityCfg("contact_sensor"),
            "threshold":  1.0,
        }

        self.rewards.dist_to_goal.params = {"asset_cfg": SceneEntityCfg("robot"), "std": 6.0}
        self.rewards.height.params       = {"asset_cfg": SceneEntityCfg("robot"), "target_z": 5.0, "std": 2.5}
        self.rewards.stability.params    = {"asset_cfg": SceneEntityCfg("robot"), "lin_std": 2.0, "ang_std": 6.0}
        self.rewards.vel_towards_goal.params = {
            "asset_cfg":  SceneEntityCfg("robot"),
            "min_speed":  0.2,
            "speed_ref":  3.0,
            "use_relu":   True,
        }

        self.rewards.lidar_threat.params = {
            "lidar_name":   "lidar",
            "threshold":    1.2,
            "exp_scale":    0.25,
            "cap":          5.0,
            "use_grid":     False,
            "theta_min":      30.0,
            "theta_max":      90.0,
            "phi_min":        0.0,
            "phi_max":        360.0,
            "delta_theta":    10.0,
            "delta_phi":      5.0,
            "max_vis_points": 10000,
        }

        self.rewards.energy.params = {
            "asset_cfg":       SceneEntityCfg("robot"),
            "lin_vel_scale":   5.0,
            "ang_vel_scale":   8.0,
            "lin_acc_scale":   15.0,
            "ang_acc_scale":   25.0,
            "include_acc":     True,
            "max_penalty":     10.0,
        }

        self.rewards.success_bonus.params = {"asset_cfg": SceneEntityCfg("robot"), "threshold": GOAL_RADIUS}
        self.rewards.oob_penalty.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "x_bounds":  (-60.0, 60.0),
            "y_bounds":  (-60.0, 60.0),
            "z_bounds":  (1.0,   10.0),
        }
        self.rewards.collision_penalty.params = {"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0}

        self.observations.policy.lidar_grid.params = {
            "lidar_name":     "lidar",
            "theta_min":      30.0,
            "theta_max":      90.0,
            "phi_min":        0.0,
            "phi_max":        360.0,
            "delta_theta":    10.0,
            "delta_phi":      5.0,
            "empty_value":    50.0,
            "max_vis_points": 10000,
        }


@configclass
class Test6DroneLidarEnvCfg(Test6DroneEnvCfg):
    scene: Test6SceneWithLidarCfg = Test6SceneWithLidarCfg(num_envs=1, env_spacing=0.0)
