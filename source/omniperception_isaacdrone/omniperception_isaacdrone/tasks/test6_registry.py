from __future__ import annotations

import gymnasium as gym

TASK_ID_NO_LIDAR = "Isaac-OmniPerception-Drone-v0"
TASK_ID_LIDAR = "Isaac-OmniPerception-Drone-Lidar-v0"

# IMPORTANT:
# Use our custom env class (MyDroneRLEnv) so env has goal_pos_w
ENTRY_POINT = "omniperception_isaacdrone.envs.test6_env:MyDroneRLEnv"

gym.register(
    id=TASK_ID_NO_LIDAR,
    entry_point=ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omniperception_isaacdrone.envs.test6_env_cfg:Test6DroneEnvCfg",
    },
)

gym.register(
    id=TASK_ID_LIDAR,
    entry_point=ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omniperception_isaacdrone.envs.test6_env_cfg:Test6DroneLidarEnvCfg",
    },
)
