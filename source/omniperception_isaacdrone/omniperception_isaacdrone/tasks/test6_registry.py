from __future__ import annotations

import gymnasium as gym

# Gym task identifiers used by train/play scripts.
TASK_ID_NO_LIDAR = "Isaac-OmniPerception-Drone-v0"
TASK_ID_LIDAR = "Isaac-OmniPerception-Drone-Lidar-v0"

# Use the custom env class so MDP functions can access goal_pos_w and caches.
ENTRY_POINT = "omniperception_isaacdrone.envs.test_env:MyDroneRLEnv"


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------
gym.register(
    id=TASK_ID_NO_LIDAR,
    entry_point=ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omniperception_isaacdrone.envs.test_env_cfg:Test6DroneEnvCfg",
    },
)

gym.register(
    id=TASK_ID_LIDAR,
    entry_point=ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omniperception_isaacdrone.envs.test_env_cfg:Test6DroneLidarEnvCfg",
    },
)
