import numpy as np
from geometry.transforms import make_transform_matrix, transform_points

def project_lidar_to_camera(lidar_world_points: np.ndarray, 
                            cam_world_pos: np.ndarray, 
                            cam_world_quat: np.ndarray, 
                            intrinsics: np.ndarray) -> np.ndarray:
    """
    Projects 3D global LiDAR points into 2D camera pixels.
    Returns array of (u, v, depth).
    """
    if len(lidar_world_points) == 0:
        return np.empty((0, 3))
        
    # 1. Transform points from World to Camera ROS frame (X-fwd, Y-left, Z-up)
    T_world_cam = make_transform_matrix(cam_world_pos, cam_world_quat)
    T_cam_world = np.linalg.inv(T_world_cam)
    points_cam_ros = transform_points(lidar_world_points, T_cam_world)
    
    # 2. Filter out points behind the camera (X <= 0 in ROS)
    front_mask = points_cam_ros[:, 0] > 0.1
    points_cam_ros = points_cam_ros[front_mask]
    
    if len(points_cam_ros) == 0:
        return np.empty((0, 3))
        
    # 3. Convert ROS frame to OpenCV frame (Z-fwd, X-right, Y-down)
    x_cv = -points_cam_ros[:, 1]
    y_cv = -points_cam_ros[:, 2]
    z_cv = points_cam_ros[:, 0]
    
    # 4. Perspective Projection
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    u = (fx * x_cv / z_cv) + cx
    v = (fy * y_cv / z_cv) + cy
    
    return np.vstack((u, v, z_cv)).T