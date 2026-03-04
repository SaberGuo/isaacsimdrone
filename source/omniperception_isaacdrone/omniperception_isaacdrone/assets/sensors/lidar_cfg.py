# source/omniperception_isaacdrone/omniperception_isaacdrone/assets/sensors/lidar_cfg.py

from isaaclab.sensors import LidarSensorCfg
from isaaclab.sensors.ray_caster.patterns import LivoxPatternCfg

# 推荐：先用 mid360（360°）方便验证
LIDAR_CFG = LidarSensorCfg(
    # 注意：这里先写一个默认 link，后面在 SceneCfg 里 replace 成你真实的 link
    prim_path="{ENV_REGEX_NS}/Robot/base",

    pattern_cfg=LivoxPatternCfg(
        sensor_type="mid360",
        use_simple_grid=False,   # True=网格假扫描；False=读 .npy 真实 Livox
        samples=20000,            # 先用 8k，性能/效果比较平衡
        downsample=1,
    ),

    max_distance=50.0,
    min_range=0.2,

    return_pointcloud=True,
    pointcloud_in_world_frame=False,   # 方便直接画到世界坐标系

    enable_sensor_noise=False,
    random_distance_noise=0.02,

    # 关键：把你想让 LiDAR “看见”的物体路径写进来
    mesh_prim_paths=[
        "/World/ground",
        "/World/Obstacles",      # 你的障碍物父节点
    ],

    debug_vis=False,            # 想看射线就改 True（也可在脚本里动态改）
)
