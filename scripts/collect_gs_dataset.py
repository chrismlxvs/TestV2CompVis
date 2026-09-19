import os
import sys
import json
import cv2
import time
import numpy as np
import pybullet as pb
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from simulation.simulation_loop import SimulationLoop
from mapping.point_cloud_map import GlobalPointCloudMap
from splatting.lidar_initialization import LidarGaussianInitializer
from geometry.transforms import make_transform_matrix

def get_keyboard_command() -> tuple:
    keys = pb.getKeyboardEvents()
    v, w = 0.0, 0.0
    if pb.B3G_UP_ARROW in keys and keys[pb.B3G_UP_ARROW] & pb.KEY_IS_DOWN: v = 1.0
    if pb.B3G_DOWN_ARROW in keys and keys[pb.B3G_DOWN_ARROW] & pb.KEY_IS_DOWN: v = -1.0
    if pb.B3G_LEFT_ARROW in keys and keys[pb.B3G_LEFT_ARROW] & pb.KEY_IS_DOWN: w = 1.5
    if pb.B3G_RIGHT_ARROW in keys and keys[pb.B3G_RIGHT_ARROW] & pb.KEY_IS_DOWN: w = -1.5
    return v, w

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    configs = [
        OmegaConf.load(os.path.join(base_dir, 'config', 'simulation.yaml')),
        OmegaConf.load(os.path.join(base_dir, 'config', 'robot.yaml')),
        OmegaConf.load(os.path.join(base_dir, 'config', 'lidar.yaml')),
        OmegaConf.load(os.path.join(base_dir, 'config', 'camera.yaml')),
        OmegaConf.load(os.path.join(base_dir, 'config', 'mapping.yaml')),
        OmegaConf.load(os.path.join(base_dir, 'config', 'gaussian_splatting.yaml'))
    ]
    config = OmegaConf.merge(*configs)
    
    out_dir = os.path.join(base_dir, config.export.output_dir)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    loop = SimulationLoop(config)
    global_map = GlobalPointCloudMap(config)
    initializer = LidarGaussianInitializer(config)
    
    frames_meta = []
    frame_id = 0
    
    print("Dataset collection active! Drive around using arrow keys to scan the room.")
    print("Press Ctrl+C when finished to export the images, transforms, and PLY.")
    
    try:
        while True:
            v, w = get_keyboard_command()
            loop.step(v, w)
            
            if loop.steps % loop.lidar_step_interval == 0:
                scan = loop.lidar.scan(timestamp=loop.time, visualize=False)
                cam_frame = loop.camera.capture(timestamp=loop.time)
                
                if scan is not None:
                    robot_pos, robot_quat = loop.robot.get_pose()
                    
                    # 1. Accumulate Global Map
                    global_map.add_scan(
                        scan.local_points, robot_pos, robot_quat,
                        loop.lidar.t_base_lidar_pos, loop.lidar.t_base_lidar_quat
                    )
                    
                    # 2. Save RGB Frame
                    img_filename = f"frame_{frame_id:05d}.png"
                    img_path = os.path.join(images_dir, img_filename)
                    rgb_image = np.asarray(cam_frame.rgb)
                    if rgb_image.dtype != np.uint8:
                        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
                    cv2.imwrite(img_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                    
                    # 3. Save Camera Extrinsics Matrix (ROS World -> OpenCV Cam Frame)
                    T_world_cam_ros = make_transform_matrix(
                        cam_frame.t_world_cam_pos, cam_frame.t_world_cam_quat
                    )
                    
                    # Coordinate frame correction: ROS (X-fwd, Y-left, Z-up) to OpenCV/Nerfstudio (X-right, Y-down, Z-fwd)
                    R_ros_cv = np.array([
                        [0, -1,  0, 0],
                        [0,  0, -1, 0],
                        [1,  0,  0, 0],
                        [0,  0,  0, 1]
                    ])
                    T_world_cam_cv = T_world_cam_ros @ R_ros_cv
                    
                    frames_meta.append({
                        "file_path": f"images/{img_filename}",
                        "transform_matrix": T_world_cam_cv.tolist()
                    })
                    
                    frame_id += 1
                    
            time.sleep(config.simulation.timestep)
            
    except KeyboardInterrupt:
        print("\nFinalizing dataset export...")
    finally:
        loop.sim.disconnect()
        
        # Write Camera Intrinsics + Extrinsics Json
        if len(frames_meta) > 0:
            transforms_json = {
                "fl_x": float(cam_frame.intrinsics[0, 0]),
                "fl_y": float(cam_frame.intrinsics[1, 1]),
                "cx": float(cam_frame.intrinsics[0, 2]),
                "cy": float(cam_frame.intrinsics[1, 2]),
                "w": int(config.camera.width),
                "h": int(config.camera.height),
                "camera_angle_x": float(np.radians(config.camera.fov)),
                "frames": frames_meta
            }
            
            json_path = os.path.join(out_dir, "transforms.json")
            with open(json_path, "w") as f:
                json.dump(transforms_json, f, indent=4)
            print(f"Exported dataset metadata: {json_path}")

            # Generate and export LiDAR Gaussian Initialization PLY
            gaussians = initializer.generate_initial_gaussians(global_map.global_pcd)
            ply_path = os.path.join(out_dir, "lidar_initialization.ply")
            initializer.save_ply(gaussians, ply_path)
        else:
            print("No frames were captured. Export aborted.")

if __name__ == "__main__":
    main()