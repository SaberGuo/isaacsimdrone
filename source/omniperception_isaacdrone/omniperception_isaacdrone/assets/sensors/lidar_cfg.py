from isaaclab.sensors import LidarSensorCfg
from isaaclab.sensors.ray_caster.patterns import LivoxPatternCfg

from .lidar_sensor import BodyFrameLidarSensor


LIDAR_CFG = LidarSensorCfg(
    class_type=BodyFrameLidarSensor,
    prim_path="{ENV_REGEX_NS}/Robot/body",
    ray_alignment="base",
    update_frequency=50.0,
    # Keep the nominal scan period explicit in the config snapshot.
    update_period=1.0 / 50.0,

    pattern_cfg=LivoxPatternCfg(
        sensor_type="mid360",
        use_simple_grid=False,
        samples=20000,
        downsample=1,
    ),

    max_distance=50.0,
    min_range=0.2,

    return_pointcloud=True,
    pointcloud_in_world_frame=False,

    enable_sensor_noise=False,
    random_distance_noise=0.02,

    mesh_prim_paths=[
        "/World/ground",
        "/World/Obstacles",
        "/World/Wall",
    ],

    debug_vis=False,
)
