import time
import os
import sys
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from simulation.simulation_loop import SimulationLoop
from odometry.lidar_odometry import LidarOdometry
from geometry.transforms import make_transform_matrix

def get_keyboard_command() -> tuple:
    keys = pb.getKeyboardEvents()
    v, w = 0.0, 0.0
    if pb.B3G_UP_ARROW in keys and keys[pb.B3G_UP_ARROW] & pb.KEY_IS_DOWN: v = 1.5
    if pb.B3G_DOWN_ARROW in keys and keys[pb.B3G_DOWN_ARROW] & pb.KEY_IS_DOWN: v = -1.5
    if pb.B3G_LEFT_ARROW in keys and keys[pb.B3G_LEFT_ARROW] & pb.KEY_IS_DOWN: w = 2.0
    if pb.B3G_RIGHT_ARROW in keys and keys[pb.B3G_RIGHT_ARROW] & pb.KEY_IS_DOWN: w = -2.0
    return v, w

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    configs = [
        OmegaConf.load(os.path.join(base_dir, 'config', 'simulation.yaml')),
        OmegaConf.load(os.path.join(base_dir, 'config', 'robot.yaml')),
        OmegaConf.load(os.path.join(base_dir, 'config', 'lidar.yaml')),
        OmegaConf.load(os.path.join(base_dir, 'config', 'camera.yaml'))
    ]
    config = OmegaConf.merge(*configs)
    
    loop = SimulationLoop(config)
    
    # Get initial pose to seed odometry
    init_pos, init_quat = loop.robot.get_pose()
    init_T = make_transform_matrix(init_pos, init_quat)
    odometry = LidarOdometry(initial_pose=init_T)
    
    gt_trajectory_x, gt_trajectory_y = [], []
    est_trajectory_x, est_trajectory_y = [], []
    
    print("Odometry tracking started! Drive around to build the trajectory.")
    print("Press Ctrl+C to finish and view the trajectory comparison.")
    
    try:
        while True:
            v, w = get_keyboard_command()
            loop.step(v, w)
            
            if loop.steps % loop.lidar_step_interval == 0:
                scan = loop.lidar.scan(timestamp=loop.time, visualize=False)
                
                if scan is not None:
                    # 1. Ground Truth
                    gt_pos, _ = loop.robot.get_pose()
                    gt_trajectory_x.append(gt_pos[0])
                    gt_trajectory_y.append(gt_pos[1])
                    
                    # 2. Estimate Pose via ICP
                    est_T = odometry.update(scan.local_points)
                    est_trajectory_x.append(est_T[0, 3])
                    est_trajectory_y.append(est_T[1, 3])
                    
            time.sleep(config.simulation.timestep)
            
    except KeyboardInterrupt:
        print("\nPlotting Trajectories...")
    finally:
        loop.sim.disconnect()
        
        # Plot evaluation
        plt.figure(figsize=(10, 8))
        plt.plot(gt_trajectory_x, gt_trajectory_y, label='Ground Truth', color='black', linewidth=2, linestyle='--')
        plt.plot(est_trajectory_x, est_trajectory_y, label='LiDAR Odometry (ICP)', color='blue', linewidth=2)
        
        plt.title('Trajectory Evaluation: Estimated vs Ground Truth')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    main()