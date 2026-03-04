# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to visualize the UAV model in an empty scene.

This script spawns a Crazyflie drone in an empty environment with specified dimensions.
The drone size is set to [0.5, 0.5, 0.3] meters.

Usage:
    python scripts/vis_uav_model.py
"""

from __future__ import annotations

import argparse
import torch
import os

# 导入 isaaclab 的 AppLauncher
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Visualize UAV model in empty scene")
# [注意] --headless 参数由 AppLauncher 自动添加，这里必须去掉
# parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--disable_fabric", action="store_true", default=False, 
                    help="Disable fabric and use USD I/O operations.")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# [关键修复] 必须在 app 启动后导入这些模块
from pxr import UsdGeom
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def get_uav_asset_path():
    """Get the path to the UAV USD model.
    
    Returns:
        str: Absolute path to the cf2x.usd file
    """
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Construct path to the USD model
    usd_path = os.path.join(
        project_root,
        "source/omniperception_isaacdrone/omniperception_isaacdrone/assets/robots/cf2x.usd"
    )
    
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"UAV model not found at: {usd_path}")
    
    return usd_path


def design_scene() -> tuple[dict, list[list[float]]]:
    """Design the scene with a UAV model.
    
    Returns:
        tuple: Dictionary of scene entities and list of initial positions
    """
    # Get UAV model path
    uav_model_path = get_uav_asset_path()
    
    # Ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # Lighting
    cfg_dome_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg_dome_light.func("/World/Light", cfg_dome_light)
    
    # Configure UAV articulation
    # Original Crazyflie dimensions: approximately [0.092, 0.092, 0.029] m
    # Target dimensions: [0.5, 0.5, 0.3] m
    # Scale factors: x=5.43, y=5.43, z=10.34
    scale_x = 5.43
    scale_y = 5.43
    scale_z = 10.34
    
    uav_cfg = ArticulationCfg(
        # [修复点 3] 修改 prim_path 为具体路径
        prim_path="/World/UAV",
        spawn=sim_utils.UsdFileCfg(
            usd_path=uav_model_path,
            scale=(scale_x, scale_y, scale_z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),  # Spawn 1 meter above ground
            rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
        ),
        # [保留修复点 2] 必须添加 actuators 字段
        actuators={}
    )
    
    # Create articulation
    uav = Articulation(cfg=uav_cfg)
    
    # Setup scene entities
    scene_entities = {"uav": uav}
    
    # Define spawn positions for multiple instances (if needed)
    spawn_positions = [[0.0, 0.0, 1.0]]
    
    return scene_entities, spawn_positions


def run_simulator(sim: SimulationContext, entities: dict[str, Articulation]):
    """Run the simulation loop.
    
    Args:
        sim: Simulation context
        entities: Dictionary of scene entities
    """
    # Extract UAV
    uav = entities["uav"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # Reset counters
            sim_time = 0.0
            count = 0
            
            # Reset root state
            # 注意：由于 prim_path 是单一路径，root_state 的维度是 [1, 13]
            root_state = uav.data.default_root_state.clone()
            # 这里原本的逻辑是把 default 状态再加高 1米
            root_state[:, :3] += torch.tensor([0.0, 0.0, 1.0], device=uav.device)
            uav.write_root_state_to_sim(root_state)
            
            # Reset joint state
            joint_pos, joint_vel = uav.data.default_joint_pos.clone(), uav.data.default_joint_vel.clone()
            uav.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # Clear internal buffers
            uav.reset()
            print("[INFO]: Resetting UAV state...")
        
        # Apply zero actions (hover mode - you can modify this for active control)
        # For visualization, we just let physics take over
        
        # Write data to sim
        uav.write_data_to_sim()
        
        # Perform step
        sim.step()
        
        # Update buffers
        uav.update(sim_dt)
        
        # Update sim-time
        sim_time += sim_dt
        count += 1
        
        # Print info every 100 steps
        if count % 100 == 0:
            print(f"[INFO]: Sim time: {sim_time:.2f} s | UAV position: {uav.data.root_pos_w[0, :3].cpu().numpy()}")


def main():
    """Main function."""
    # Initialize simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0" if torch.cuda.is_available() else "cpu")
    sim = SimulationContext(sim_cfg)
    
    # [新增代码] 检查并打印当前场景的单位设置
    stage = omni.usd.get_context().get_stage()
    if stage:
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        print(f"--------------------------------------------------")
        print(f"[INFO]: Current Stage Units: 1 Unit = {meters_per_unit} Meters")
        print(f"--------------------------------------------------")
    else:
        print("[WARNING]: Could not retrieve stage to check units.")
    
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.5])
    
    # Design scene
    scene_entities, scene_origins = design_scene()
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete. Starting simulation...")
    
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    try:
        # Run the main function
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        # Close the simulator
        simulation_app.close()
