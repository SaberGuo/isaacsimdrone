# SPDX-License-Identifier: BSD-3-Clause
"""MDP components for OmniPerception IsaacDrone (Test6)."""

# Keep exports grouped by Manager type so env cfg files can import this module as my_mdp.
from .test_actions import RootTwistVelocityActionTerm
from .test_curriculums import update_obstacle_curriculum
from .test_events import reset_root_state_on_square_edge, randomize_obstacles_on_reset
from .test_observations import (
    obs_goal_delta,
    obs_lidar_min_range_grid,
    obs_root_ang_vel_norm,
    obs_root_lin_vel_norm,
    obs_root_pos_norm,
    obs_root_quat_norm,
    obs_root_pos_z_norm,
    obs_goal_delta_norm,
    obs_goal_dir_dist_norm,
    obs_prev_action_norm01,
    obs_projected_gravity_norm,
    obs_state_norm,
)
from .test_rewards import (
    penalty_apf_repulsive,
    penalty_safe_vel,
    penalty_collision,
    penalty_energy,
    penalty_height_error,
    penalty_lidar_threat,
    penalty_out_of_workspace,
    penalty_time_out,
    penalty_time_cost,
    reward_action_l2,
    reward_apf_attractive,
    reward_distance_to_goal,
    reward_goal_reached,
    reward_height_tracking,
    reward_heading_align_velocity,
    reward_progress_to_goal,
    reward_stability,
    reward_velocity_towards_goal,
)
from .test_terminations import (
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
    "obs_goal_dir_dist_norm",
    "obs_prev_action_norm01",
    "obs_state_norm",
    
    # rewards
    "penalty_apf_repulsive",
    "penalty_safe_vel",
    "penalty_height_error",
    "reward_apf_attractive",
    "reward_distance_to_goal",
    "reward_progress_to_goal",
    "reward_height_tracking",
    "reward_heading_align_velocity",
    "reward_stability",
    "reward_velocity_towards_goal",
    "penalty_lidar_threat",
    "penalty_energy",
    "reward_goal_reached",
    "penalty_out_of_workspace",
    "penalty_time_out",
    "penalty_time_cost",
    "penalty_collision",
    "reward_action_l2",
    
    # terminations
    "termination_reached_goal",
    "termination_out_of_workspace",
    "termination_collision",
]
