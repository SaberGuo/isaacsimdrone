# # SPDX-License-Identifier: BSD-3-Clause
# """Task definitions for OmniPerception IsaacDrone."""

# from .uav_env_cfg import UavEnvCfg  # 根据实际类名调整

# __all__ = [
#     "UavEnvCfg",
# ]
# omniperception_isaacdrone/tasks/__init__.py

"""
Tasks package.

IMPORTANT:
- Keep this file lightweight.
- Do NOT import heavy modules (e.g., env cfgs) here, because importing
  omniperception_isaacdrone.tasks.* will execute this file first.
"""

# If you want to expose mdp submodule symbols, do it via tasks/mdp/__init__.py
# Users should import like:
#   from omniperception_isaacdrone.tasks.mdp import ...
#
# Avoid importing uav_env_cfg here to prevent unwanted side-effects.
