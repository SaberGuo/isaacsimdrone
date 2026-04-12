# omniperception_isaacdrone/envs/test6_env_cfg.py
from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg,
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, LidarSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg

from omniperception_isaacdrone.assets.robots.drone_cfg import DRONE_CFG, DRONE_MASS

try:
    from omniperception_isaacdrone.assets.sensors.lidar_cfg import LIDAR_CFG
except Exception:
    LIDAR_CFG = None

from omniperception_isaacdrone.tasks import mdp as my_mdp


# -----------------------------------------------------------------------------
# Normalization hyper-params
# -----------------------------------------------------------------------------
@configclass
class NormalizationCfg:
    x_bounds: tuple[float, float] = (-80.0, 80.0)
    y_bounds: tuple[float, float] = (-80.0, 80.0)
    z_bounds: tuple[float, float] = (0.0, 10.0)

    diag_scale: float = 1.1

    lin_vel_max: float = 6.0
    ang_vel_max: float = 31.4

    quat_hemisphere: bool = True
    state_dim: int = 17


@configclass
class ObstacleCurriculumSettingsCfg:
    enabled: bool = True
    levels: tuple[int, ...] = (0, 10, 20, 40, 60, 100)
    initial_level: int = 0

    # Promotion criterion: recent reached_goal ratio
    success_term_name: str = "reached_goal"
    success_threshold: float = 0.8

    # Rolling window over recent terminated episodes
    window_size: int = 200
    min_samples: int = 100

    # Avoid jumping multiple levels using stale statistics from an easier stage
    clear_history_on_promotion: bool = True


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

    robot.spawn = DRONE_CFG.spawn.replace(
        scale=(1,1,1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.05,
            angular_damping=0.05,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        activate_contact_sensors=True,
    )

    robot.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
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
    obstacles: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Obstacles/obj_.*",  # 修改为全局路径，即每个env共享碰撞
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1000.0, 1000.0, -1000.0)
        )
    )

@configclass
class Test6SceneWithLidarCfg(Test6SceneCfg):
    if LIDAR_CFG is not None:
        lidar: LidarSensorCfg = LIDAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# -----------------------------------------------------------------------------
# Actions / Obs / Events / Rewards / Terminations / Curriculum
# -----------------------------------------------------------------------------
@configclass
class Test6ActionsCfg:
    root_twist = ActionTermCfg(
        class_type=my_mdp.RootTwistVelocityActionTerm,
        asset_name="robot",
    )


@configclass
class Test6ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        root_pos_z = ObsTerm(func=my_mdp.obs_root_pos_z_norm, params={"asset_cfg": SceneEntityCfg("robot")})
        root_quat = ObsTerm(func=my_mdp.obs_root_quat_norm, params={"asset_cfg": SceneEntityCfg("robot")})
        root_lin_vel = ObsTerm(func=my_mdp.obs_root_lin_vel_norm, params={"asset_cfg": SceneEntityCfg("robot")})
        root_ang_vel = ObsTerm(func=my_mdp.obs_root_ang_vel_norm, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=my_mdp.obs_projected_gravity_norm, params={"asset_cfg": SceneEntityCfg("robot")})
        goal_delta = ObsTerm(func=my_mdp.obs_goal_delta_norm, params={"asset_cfg": SceneEntityCfg("robot")})

        lidar_grid = ObsTerm(
            func=my_mdp.obs_lidar_min_range_grid,
            params=dict(
                lidar_name="lidar",
                theta_min=30.0,
                theta_max=90.0,
                phi_min=0.0,
                phi_max=360.0,
                delta_theta=10.0,
                delta_phi=5.0,
                empty_value=0.0,
                max_vis_points=12000,
            ),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Test6EventCfg:
    reset_robot_base = EventTerm(func=my_mdp.reset_root_state_on_square_edge, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot"),
        "square_half_size": 35.0,
        "z_range": (3.0, 7.0),
    })
    
    randomize_obstacles = EventTerm(
        func=my_mdp.randomize_obstacles_on_reset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("obstacles"),
            "x_range": (-33.0, 33.0),
            "y_range": (-33.0, 33.0),
            "z_height": 10.0,
        }
    )

@configclass
class Test6RewardsCfg:
    # ---- 密集导航奖励（主信号，per-step 量级 O(0.1~1)） ----
    progress_to_goal = RewTerm(func=my_mdp.reward_progress_to_goal, weight=40.0, params={})
    dist_to_goal = RewTerm(func=my_mdp.reward_distance_to_goal, weight=5.0, params={})
    vel_towards_goal = RewTerm(func=my_mdp.reward_velocity_towards_goal, weight=8.0, params={})
    dijkstra_progress = RewTerm(func=my_mdp.reward_dijkstra_progress, weight=20.0, params={})

    # ---- APF 人工势场奖励（叠加在已有奖励之上，默认关闭） ----
    apf_attractive = RewTerm(
        func=my_mdp.reward_apf_attractive,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    apf_repulsive = RewTerm(
        func=my_mdp.penalty_apf_repulsive,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ---- 稳定性奖励 ----
    height = RewTerm(func=my_mdp.reward_height_tracking, weight=8.0, params={})
    stability = RewTerm(func=my_mdp.reward_stability, weight=0.05, params={})
    attitude_tilt = RewTerm(
        func=my_mdp.penalty_attitude_tilt,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_tilt_deg": 20.0,
            "std_deg": 20.0,
        },
    )

    # ---- 安全惩罚 ----
    lidar_threat = RewTerm(func=my_mdp.penalty_lidar_threat, weight=-200.0, params={})
    energy = RewTerm(func=my_mdp.penalty_energy, weight=-0.02, params={})
    action_l2 = RewTerm(func=my_mdp.reward_action_l2, weight=-0.005)

    # ---- 终端信号（与密集累计量级可比） ----
    success_bonus = RewTerm(func=my_mdp.reward_goal_reached, weight=800.0, params={})
    collision_penalty = RewTerm(func=my_mdp.penalty_collision, weight=-800.0, params={})
    oob_penalty = RewTerm(func=my_mdp.penalty_out_of_workspace, weight=-800.0, params={})
    timeout_penalty = RewTerm(func=my_mdp.penalty_time_out, weight=-500.0, params={})


@configclass
class Test6TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reached_goal = DoneTerm(func=my_mdp.termination_reached_goal, params={})
    oob = DoneTerm(func=my_mdp.termination_out_of_workspace, params={})
    collision = DoneTerm(func=my_mdp.termination_collision, params={})


@configclass
class Test6CurriculumCfg:
    obstacle_count = CurrTerm(
        func=my_mdp.update_obstacle_curriculum,
        params={
            "levels": (0, 10, 20, 40, 60, 100),
            "success_term_name": "reached_goal",
            "success_threshold": 0.8,
            "window_size": 200,
        }
    )



@configclass
class Test6DroneEnvCfg(ManagerBasedRLEnvCfg):
    normalization: NormalizationCfg = NormalizationCfg()
    obstacle_curriculum: ObstacleCurriculumSettingsCfg = ObstacleCurriculumSettingsCfg()

    scene: Test6SceneCfg = Test6SceneCfg(num_envs=1, env_spacing=0.0)
    observations: Test6ObservationsCfg = Test6ObservationsCfg()
    actions: Test6ActionsCfg = Test6ActionsCfg()
    events: Test6EventCfg = Test6EventCfg()
    rewards: Test6RewardsCfg = Test6RewardsCfg()
    terminations: Test6TerminationsCfg = Test6TerminationsCfg()
    curriculum: Test6CurriculumCfg = Test6CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.decimation = 1
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0

        self.viewer.eye = (60.0, 60.0, 40.0)
        self.viewer.lookat = (0.0, 0.0, 5.0)

        WORKSPACE_X = (-80.0, 80.0)
        WORKSPACE_Y = (-80.0, 80.0)
        WORKSPACE_Z = (0.0, 10.0)
        GOAL_RADIUS = 2.5

        self.normalization.x_bounds = WORKSPACE_X
        self.normalization.y_bounds = WORKSPACE_Y
        self.normalization.z_bounds = WORKSPACE_Z

        self.actions.root_twist.params = {
            "mass": DRONE_MASS,
            "use_sim_total_mass": True,
            "prevent_negative_thrust": True,
            "vel_scale": 6.0,
            "vel_clip": 8.0,
            "yaw_rate_scale": 3.14,
            "yaw_rate_clip": 6.28,
            "thrust_sign": 1.0,
            "g": 9.81,
            "vel_gain": (3.0, 3.0, 4.0),
            "pos_gain": (0.0, 0.0, 0.0),
            "attitude_gain": (6.0, 6.0, 1.5),
            "ang_rate_gain": (0.25, 0.25, 0.18),
            "thrust_limit_factor": 3,
            "torque_limit": (0.2, 0.2, 0.1),
            "inertia_diag": (0.02, 0.02, 0.04),
        }

        self.events.reset_robot_base.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "square_half_size": 35.0,
            "z_range": (3.0, 7.0),
        }

        self.terminations.reached_goal.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_RADIUS,
        }
        self.terminations.oob.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "x_bounds": WORKSPACE_X,
            "y_bounds": WORKSPACE_Y,
            "z_bounds": WORKSPACE_Z,
        }
        self.terminations.collision.params = {
            "sensor_cfg": SceneEntityCfg("contact_sensor"),
            "threshold": 1.0,
        }

        self.rewards.progress_to_goal.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "speed_ref": 4.0,
            "clip": 1.0,
        }
        self.rewards.dist_to_goal.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 50.0,
        }
        self.rewards.height.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "target_z": 5.0,
            "std": 2.5,
        }
        self.rewards.stability.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "lin_std": 4.0,
            "ang_std": 6.0,
        }
        self.rewards.vel_towards_goal.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "min_speed": 0.2,
            "speed_ref": 3.0,
            "use_relu": True,
        }
        self.rewards.dijkstra_progress.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "grid_size": 80,  # Reduced from 160 for performance (2m resolution)
            "cell_size": 2.0,  # 2m per cell for faster computation
            "update_interval": 10,  # Update every 10 steps (was 5)
            "speed_ref": 4.0,
            "clip": 1.0,
        }

        self.rewards.lidar_threat.params = {
            "lidar_name": "lidar",
            "safe_dist": None,
            "safe_dist_ratio": 0.1,
            "exp_scale": 2.0,
            "cap": 3.0,
            "use_grid": True,
            "theta_min": 30.0,
            "theta_max": 90.0,
            "phi_min": 0.0,
            "phi_max": 360.0,
            "delta_theta": 10.0,
            "delta_phi": 5.0,
            "max_vis_points": 12000,
        }

        self.rewards.energy.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "lin_vel_scale": 100.0,
            "ang_vel_scale": 100.0,
            "lin_acc_scale": 100.0,
            "ang_acc_scale": 100.0,
            "include_acc": True,
            "acc_weight": 0.2,
            "max_penalty": 1.0,
        }

        self.rewards.success_bonus.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_RADIUS,
        }
        self.rewards.oob_penalty.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "x_bounds": WORKSPACE_X,
            "y_bounds": WORKSPACE_Y,
            "z_bounds": WORKSPACE_Z,
        }
        self.rewards.collision_penalty.params = {
            "sensor_cfg": SceneEntityCfg("contact_sensor"),
            "threshold": 1.0,
        }

        self.scene.replicate_physics = True
        self.scene.filter_collisions = True

@configclass
class Test6DroneLidarEnvCfg(Test6DroneEnvCfg):
    scene: Test6SceneWithLidarCfg = Test6SceneWithLidarCfg(num_envs=1, env_spacing=0.0)
