import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_obstacles", type=int, default=50)
parser.add_argument("--vz", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

WORKSPACE_PATH = (
    Path.home()
    / "hjr_isaacdrone_ws"
    / "omniperception_isaacdrone"
    / "source"
    / "omniperception_isaacdrone"
)
if str(WORKSPACE_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_PATH))

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

try:
    from omniperception_isaacdrone.assets.robots.drone_cfg import DRONE_CFG
except ImportError:
    simulation_app.close()
    raise


def reset_root_state_fixed_pose(env, env_ids, asset_cfg, pos=(30.0, 0.0, 5.0), rot=(1.0, 0.0, 0.0, 0.0)):
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    positions = torch.empty((n, 3), device=env.device, dtype=torch.float32)
    positions[:, 0] = float(pos[0])
    positions[:, 1] = float(pos[1])
    positions[:, 2] = float(pos[2])
    orientations = torch.empty((n, 4), device=env.device, dtype=torch.float32)
    orientations[:, 0] = float(rot[0])
    orientations[:, 1] = float(rot[1])
    orientations[:, 2] = float(rot[2])
    orientations[:, 3] = float(rot[3])
    root_states = torch.cat([positions, orientations], dim=1)
    asset.write_root_pose_to_sim(root_states, env_ids=env_ids)
    velocities = torch.zeros((n, 6), device=env.device, dtype=torch.float32)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


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
        debug_vis=False,
    )

    robot: ArticulationCfg = DRONE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    robot.spawn = DRONE_CFG.spawn.replace(
        scale=(20, 20, 10),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(enable_gyroscopic_forces=True),
    )

    robot.init_state = ArticulationCfg.InitialStateCfg(
        pos=(30.0, 0.0, 5.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
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
        func=reset_root_state_fixed_pose,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "pos": (30.0, 0.0, 5.0), "rot": (1.0, 0.0, 0.0, 0.0)},
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
        self.viewer.eye = (30.0, 30.0, 25.0)
        self.viewer.lookat = (0.0, 0.0, 5.0)


class ObstacleSpawner:
    def __init__(
        self,
        num_obstacles=50,
        area_half_size=10.0,
        xy_size_range=(0.5, 1.5),
        z_height=10.0,
        seed=42,
        obstacle_root_path="/World/Obstacles",
    ):
        self.num_obstacles = num_obstacles
        self.area_half_size = area_half_size
        self.xy_size_range = xy_size_range
        self.z_height = z_height
        self.seed = seed
        self.obstacle_root_path = obstacle_root_path
        if seed is not None:
            np.random.seed(seed)

    def spawn_obstacles(self):
        import isaacsim.core.utils.prims as prim_utils

        if not prim_utils.is_prim_path_valid(self.obstacle_root_path):
            prim_utils.create_prim(self.obstacle_root_path, "Xform")

        for i in range(self.num_obstacles):
            x_pos = np.random.uniform(-self.area_half_size, self.area_half_size)
            y_pos = np.random.uniform(-self.area_half_size, self.area_half_size)
            z_pos = self.z_height / 2.0
            x_size = np.random.uniform(self.xy_size_range[0], self.xy_size_range[1])
            y_size = np.random.uniform(self.xy_size_range[0], self.xy_size_range[1])
            z_size = self.z_height
            color = (np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7))
            cfg_obstacle = sim_utils.MeshCuboidCfg(
                size=(x_size, y_size, z_size),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            obstacle_path = f"{self.obstacle_root_path}/Obstacle_{i:04d}"
            cfg_obstacle.func(obstacle_path, cfg_obstacle, translation=(x_pos, y_pos, z_pos))


def main():
    env_cfg = MyEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedEnv(cfg=env_cfg)

    obstacle_spawner = ObstacleSpawner(
        num_obstacles=args_cli.num_obstacles,
        area_half_size=10.0,
        xy_size_range=(0.5, 1.5),
        z_height=10.0,
        seed=42,
    )
    obstacle_spawner.spawn_obstacles()

    env.reset()

    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    action_dim = env.action_manager.total_action_dim
    actions = torch.zeros((env.num_envs, action_dim), device=env.device, dtype=torch.float32)

    vel_cmd = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
    vel_cmd[:, 2] = float(args_cli.vz)

    count = 0
    max_iterations = 2000

    while simulation_app.is_running() and count < max_iterations:
        with torch.inference_mode():
            robot.write_root_velocity_to_sim(vel_cmd, env_ids=env_ids)
            env.step(actions)
            count += 1

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
