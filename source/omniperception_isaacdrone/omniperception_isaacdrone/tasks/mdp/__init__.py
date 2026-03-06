# SPDX-License-Identifier: BSD-3-Clause
"""MDP components for OmniPerception IsaacDrone (Test6)."""

from .test6_actions import RootTwistVelocityActionTerm
from .test6_events import reset_root_state_on_square_edge

from .test6_observations import (
    obs_goal_delta,
    obs_lidar_min_range_grid,
    # NEW normalized obs
    obs_root_pos_norm,
    obs_root_quat_norm,
    obs_root_lin_vel_norm,
    obs_root_ang_vel_norm,
    obs_projected_gravity_norm,
    obs_goal_delta_norm,
    obs_state_norm,
)

from .test6_rewards import (
    reward_distance_to_goal,
    reward_progress_to_goal,
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
    "obs_root_pos_norm",
    "obs_root_quat_norm",
    "obs_root_lin_vel_norm",
    "obs_root_ang_vel_norm",
    "obs_projected_gravity_norm",
    "obs_goal_delta_norm",
    "obs_state_norm",
    # rewards
    "reward_distance_to_goal",
    "reward_progress_to_goal",
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
