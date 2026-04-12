# SPDX-License-Identifier: BSD-3-Clause
"""MDP components for OmniPerception IsaacDrone (Test6)."""

from .test6_actions import RootTwistVelocityActionTerm
from .test6_curriculums import update_obstacle_curriculum
from .test6_events import reset_root_state_on_square_edge, randomize_obstacles_on_reset
from .test6_observations import (
    obs_goal_delta,
    obs_lidar_min_range_grid,
    obs_root_ang_vel_norm,
    obs_root_lin_vel_norm,
    obs_root_pos_norm,
    obs_root_quat_norm,
    obs_root_pos_z_norm,
    obs_goal_delta_norm,
    obs_projected_gravity_norm,
    obs_state_norm,
)
from .test6_rewards import (
    penalty_apf_repulsive,
    penalty_attitude_tilt,
    penalty_collision,
    penalty_energy,
    penalty_lidar_threat,
    penalty_out_of_workspace,
    penalty_time_out,
    reward_action_l2,
    reward_apf_attractive,
    reward_dijkstra_progress,
    reward_distance_to_goal,
    reward_goal_reached,
    reward_height_tracking,
    reward_progress_to_goal,
    reward_stability,
    reward_velocity_towards_goal,
)
from .test6_terminations import (
    termination_collision,
    termination_out_of_workspace,
    termination_reached_goal,
)

__all__ = [
    # actions
    "RootTwistVelocityActionTerm",
    
    # curriculum
    "update_obstacle_curriculum",
    
    # events
    "reset_root_state_on_square_edge",
    "randomize_obstacles_on_reset",
    
    # observations
    "obs_goal_delta",
    "obs_lidar_min_range_grid",
    "obs_root_pos_norm",
    "obs_root_pos_z_norm",
    "obs_root_quat_norm",
    "obs_root_lin_vel_norm",
    "obs_root_ang_vel_norm",
    "obs_projected_gravity_norm",
    "obs_goal_delta_norm",
    "obs_state_norm",
    
    # rewards (test6)
    "reward_distance_to_goal",
    "reward_progress_to_goal",
    "reward_height_tracking",
    "reward_stability",
    "reward_velocity_towards_goal",
    "reward_dijkstra_progress",
    "reward_apf_attractive",
    "penalty_apf_repulsive",
    "penalty_attitude_tilt",
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
