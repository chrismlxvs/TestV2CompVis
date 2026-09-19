import time
import os
import sys
import numpy as np
import pybullet as pb
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from simulation.simulation_loop import SimulationLoop
from mapping.point_cloud_map import GlobalPointCloudMap

def get_keyboard_command() -> tuple:
    """Reads arrow keys for teleop."""
    keys = pb.getKeyboardEvents()
    v, w = 0.0, 0.0
    
    if pb.B3G_UP_ARROW in keys and keys[pb.B3G_UP_ARROW] & pb.KEY_IS_DOWN:
        v = 1.5
    if pb.B3G_DOWN_ARROW in keys and keys[pb.B3G_DOWN_ARROW] & pb.KEY_IS_DOWN:
        v = -1.5
    if pb.B3G_LEFT_ARROW in keys and keys[pb.B3G_LEFT_ARROW] & pb.KEY_IS_DOWN:
        w = 2.0
    if pb.B3G_RIGHT_ARROW in keys and keys[pb.B3G_RIGHT_ARROW] & pb.KEY_IS_DOWN:
        w = -2.0
        
    return v, w

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    configs = [
        OmegaConf.load(os.path.join(base_dir, 'config', 'simulation.yaml')),
        OmegaConf.load(os.path.join(base_dir, 'config', 'robot.yaml')),
        OmegaConf.load(os.path.join(base_dir, 'config', 'lidar.yaml')),
        OmegaConf.load(os.path.join(base_dir, 'config', 'mapping.yaml')),
        OmegaConf.load(os.path.join(base_dir, 'config', 'camera.yaml'))
    ]
    config = OmegaConf.merge(*configs)
    
    # Turn off live PyBullet LiDAR drawing to speed up mapping
    loop = SimulationLoop(config)
    global_map = GlobalPointCloudMap(config)
    
    print("Mapping started! Drive around to scan the room.")
    print("Press Ctrl+C to finish mapping and visualize the 3D Point Cloud.")
    
    try:
        while True:
            v, w = get_keyboard_command()
            loop.step(v, w)
            
            # Check if a scan occurred on this step
            if loop.steps % loop.lidar_step_interval == 0:
                scan = loop.lidar.scan(timestamp=loop.time, visualize=False)
                
                if scan is not None:
                    # Get Ground Truth pose from robot
                    robot_pos, robot_quat = loop.robot.get_pose()
                    
                    # Add scan to global map
                    global_map.add_scan(
                        local_points=scan.local_points,
                        t_world_base_pos=robot_pos,
                        t_world_base_quat=robot_quat,
                        t_base_lidar_pos=loop.lidar.t_base_lidar_pos,
                        t_base_lidar_quat=loop.lidar.t_base_lidar_quat
                    )
                    
            time.sleep(config.simulation.timestep)
            
    except KeyboardInterrupt:
        print("\nMapping stopped. Generating 3D representation...")
    finally:
        loop.sim.disconnect()
        # Launch Open3D Visualizer
        global_map.visualize()

if __name__ == "__main__":
    main()