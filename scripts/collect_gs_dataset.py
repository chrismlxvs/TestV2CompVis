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

def _key_down(keys, key_code: int) -> bool:
    return key_code in keys and bool(keys[key_code] & pb.KEY_IS_DOWN)


def get_keyboard_command(max_linear: float, max_angular: float) -> tuple:
    """Arrow keys or WASD. Hold Shift for full speed; Space for hard stop."""
    keys = pb.getKeyboardEvents()

    # Space = emergency brake
    if _key_down(keys, ord(" ")) or _key_down(keys, pb.B3G_SPACE):
        return 0.0, 0.0

    boost = _key_down(keys, pb.B3G_SHIFT)
    speed_scale = 1.0 if boost else 0.7
    turn_scale = 1.0 if boost else 0.8

    v = 0.0
    w = 0.0
    if _key_down(keys, pb.B3G_UP_ARROW) or _key_down(keys, ord("w")) or _key_down(keys, ord("W")):
        v += max_linear * speed_scale
    if _key_down(keys, pb.B3G_DOWN_ARROW) or _key_down(keys, ord("s")) or _key_down(keys, ord("S")):
        v -= max_linear * speed_scale
    if _key_down(keys, pb.B3G_LEFT_ARROW) or _key_down(keys, ord("a")) or _key_down(keys, ord("A")):
        w += max_angular * turn_scale
    if _key_down(keys, pb.B3G_RIGHT_ARROW) or _key_down(keys, ord("d")) or _key_down(keys, ord("D")):
        w -= max_angular * turn_scale

    # Keep forward/back dominant when both are held, so strafing isn't cancelled oddly.
    return float(v), float(w)

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
    lidar_dir = os.path.join(out_dir, "lidar")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(lidar_dir, exist_ok=True)
    
    loop = SimulationLoop(config)
    global_map = GlobalPointCloudMap(config)
    initializer = LidarGaussianInitializer(config)
    max_linear = float(config.robot.max_linear_velocity)
    max_angular = float(config.robot.max_angular_velocity)
    
    frames_meta = []
    frame_id = 0
    
    print("Dataset collection active! Click the PyBullet window so keyboard focus works.")
    print("Controls: Arrow keys or WASD to drive, Shift = boost, Space = stop.")
    print("Target: at least 150-300 frames. Aim to cover walls, corners, and obstacles.")
    print("Drive around the room, turn often, and revisit areas from different angles.")
    print("IMPORTANT: Do NOT close the PyBullet window with the X button.")
    print("Press Ctrl+C in this terminal when finished to export safely.")
    
    try:
        while True:
            if not loop.sim.is_connected():
                print("\nPyBullet window/connection closed. Stopping capture...")
                break
            v, w = get_keyboard_command(max_linear, max_angular)
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

                    lidar_filename = f"scan_{frame_id:05d}.npy"
                    np.save(os.path.join(lidar_dir, lidar_filename), scan.local_points)
                    
                    # 3. Save Camera Extrinsics Matrix (ROS World -> OpenCV Cam Frame)
                    T_world_cam_ros = make_transform_matrix(
                        cam_frame.t_world_cam_pos, cam_frame.t_world_cam_quat
                    )
                    
                    # ROS camera axes (X-fwd, Y-left, Z-up) -> OpenCV/gsplat (X-right, Y-down, Z-fwd)
                    # Columns are OpenCV axes expressed in the ROS camera frame.
                    R_ros_cv = np.array([
                        [0,  0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0,  0, 0, 1]
                    ], dtype=float)
                    T_world_cam_cv = T_world_cam_ros @ R_ros_cv
                    
                    frames_meta.append({
                        "file_path": f"images/{img_filename}",
                        "lidar_file_path": f"lidar/{lidar_filename}",
                        "timestamp": float(cam_frame.timestamp),
                        "robot_position": np.asarray(robot_pos).tolist(),
                        "robot_quaternion": np.asarray(robot_quat).tolist(),
                        "transform_matrix": T_world_cam_cv.tolist()
                    })
                    
                    frame_id += 1
                    if frame_id == 1 or frame_id % 10 == 0:
                        print(f"Captured {frame_id} frames...", flush=True)
                    
            time.sleep(config.simulation.timestep)
            
    except KeyboardInterrupt:
        print("\nFinalizing dataset export...")
    except Exception as exc:
        print(f"\nCapture stopped due to error: {exc}")
    finally:
        loop.sim.disconnect()
        
        # Write Camera Intrinsics + Extrinsics Json
        if len(frames_meta) > 0 and "cam_frame" in locals():
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

            split_ranges = {
                "train": (0, int(len(frames_meta) * 0.70)),
                "val": (int(len(frames_meta) * 0.70), int(len(frames_meta) * 0.85)),
                "test": (int(len(frames_meta) * 0.85), len(frames_meta))
            }
            for split_name, (start, end) in split_ranges.items():
                split_data = dict(transforms_json)
                split_data["frames"] = frames_meta[start:end]
                split_path = os.path.join(out_dir, f"transforms_{split_name}.json")
                with open(split_path, "w") as f:
                    json.dump(split_data, f, indent=4)

            dataset_metadata = {
                "num_frames": len(frames_meta),
                "image_resolution": [int(config.camera.width), int(config.camera.height)],
                "camera_fps": float(config.camera.fps),
                "lidar_frequency_hz": float(config.lidar.frequency),
                "lidar_channels": int(config.lidar.channels),
                "splits": {name: end - start for name, (start, end) in split_ranges.items()},
                "color_initialization": "neutral",
                "coordinate_convention": "OpenCV/Nerfstudio camera frame: X-right, Y-down, Z-forward"
            }
            with open(os.path.join(out_dir, "metadata.json"), "w") as f:
                json.dump(dataset_metadata, f, indent=4)
            print(f"Exported dataset summary: {os.path.join(out_dir, 'metadata.json')}")

            # Generate and export LiDAR Gaussian Initialization PLY
            gaussians = initializer.generate_initial_gaussians(global_map.global_pcd)
            ply_path = os.path.join(out_dir, "lidar_initialization.ply")
            initializer.save_ply(gaussians, ply_path)
        else:
            print("No frames were captured. Export aborted.")

if __name__ == "__main__":
    main()