import time
import os
import sys
import cv2
import numpy as np
import pybullet as pb
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from simulation.simulation_loop import SimulationLoop
from perception.camera_lidar import project_lidar_to_camera
from geometry.transforms import make_transform_matrix, transform_points

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
    
    print("Sensor Fusion active! Drive around using arrow keys.")
    
    try:
        while True:
            v, w = get_keyboard_command()
            loop.step(v, w)
            
            if loop.steps % loop.lidar_step_interval == 0:
                scan = loop.lidar.scan(timestamp=loop.time, visualize=False)
                cam_frame = loop.camera.capture(timestamp=loop.time)
                
                if scan is not None:
                    # Convert local LiDAR points to World frame
                    T_world_base = make_transform_matrix(*loop.robot.get_pose())
                    T_base_lidar = make_transform_matrix(
                        loop.lidar.t_base_lidar_pos, loop.lidar.t_base_lidar_quat
                    )
                    T_world_lidar = T_world_base @ T_base_lidar
                    world_points = transform_points(scan.local_points, T_world_lidar)
                    
                    # Project World points onto Camera Image
                    projections = project_lidar_to_camera(
                        world_points, cam_frame.t_world_cam_pos, 
                        cam_frame.t_world_cam_quat, cam_frame.intrinsics
                    )
                    
                    # Draw points on RGB Image
                    rgb = np.asarray(cam_frame.rgb)
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                    display_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    
                    for point in projections:
                        u, v, depth = int(point[0]), int(point[1]), point[2]
                        if 0 <= u < config.camera.width and 0 <= v < config.camera.height:
                            # Color intensity based on depth proximity
                            intensity = max(0, 255 - int(depth * 15))
                            cv2.circle(display_img, (u, v), 2, (0, intensity, 255), -1)
                            
                    cv2.imshow("LiDAR-Camera Fusion", display_img)
                    cv2.waitKey(1)
                    
            time.sleep(config.simulation.timestep)
            
    except KeyboardInterrupt:
        pass
    finally:
        loop.sim.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()