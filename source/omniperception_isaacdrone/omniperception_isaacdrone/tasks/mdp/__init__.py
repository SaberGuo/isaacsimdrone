# SPDX-License-Identifier: BSD-3-Clause
"""MDP components for OmniPerception IsaacDrone (Test6)."""

from .test6_actions import RootTwistVelocityActionTerm
from .test6_events import reset_root_state_on_square_edge
from .test6_observations import obs_goal_delta, obs_lidar_min_range_grid

from .test6_rewards import (
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
)

from .test6_terminations import (
    termination_reached_goal,
    termination_out_of_workspace,
    termination_collision,
)

__all__ = [
    # actions
    "RootTwistVelocityActionTerm",
    # events
    "reset_root_state_on_square_edge",
    # observations
    "obs_goal_delta",
    "obs_lidar_min_range_grid",
    # rewards
    "reward_distance_to_goal",
    "reward_height_tracking",
    "reward_stability",
    "reward_velocity_towards_goal",
    "penalty_lidar_threat",
    "penalty_energy",
    "reward_goal_reached",
    "penalty_out_of_workspace",
    "penalty_time_out",
    "penalty_collision",
    "reward_action_l2",
    # terminations
    "termination_reached_goal",
    "termination_out_of_workspace",
    "termination_collision",
]
