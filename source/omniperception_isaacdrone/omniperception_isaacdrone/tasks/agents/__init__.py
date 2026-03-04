# SPDX-License-Identifier: BSD-3-Clause
"""Agent configurations for UAV training."""

import os

# 获取 yaml 配置文件路径
SKRL_PPO_CFG_PATH = os.path.join(os.path.dirname(__file__), "skrl_ppo_cfg.yaml")

__all__ = [
    "SKRL_PPO_CFG_PATH",
]
