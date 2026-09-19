import numpy as np
import open3d as o3d
from geometry.transforms import make_transform_matrix, transform_points

class GlobalPointCloudMap:
    def __init__(self, config):
        self.cfg = config.mapping
        self.global_pcd = o3d.geometry.PointCloud()
        
    def add_scan(self, local_points: np.ndarray, 
                 t_world_base_pos: np.ndarray, t_world_base_quat: np.ndarray, 
                 t_base_lidar_pos: np.ndarray, t_base_lidar_quat: np.ndarray):
        """
        Transforms local LiDAR points to the global frame and accumulates them.
        """
        if len(local_points) == 0:
            return

        # 1. Construct SE(3) Matrices
        T_world_base = make_transform_matrix(t_world_base_pos, t_world_base_quat)
        T_base_lidar = make_transform_matrix(t_base_lidar_pos, t_base_lidar_quat)
        
        # 2. Compose Transforms: T_world_lidar = T_world_base * T_base_lidar
        T_world_lidar = T_world_base @ T_base_lidar
        
        # 3. Transform Points
        world_points = transform_points(local_points, T_world_lidar)
        
        # 4. Create Open3D PointCloud
        scan_pcd = o3d.geometry.PointCloud()
        scan_pcd.points = o3d.utility.Vector3dVector(world_points)
        
        # Add color based on Z-height (makes visualization much clearer)
        z_vals = world_points[:, 2]
        z_min, z_max = 0.0, 2.5  # Approximate wall height
        z_normalized = np.clip((z_vals - z_min) / (z_max - z_min), 0, 1)
        
        colors = np.zeros((len(z_vals), 3))
        colors[:, 0] = z_normalized          # Red increases with height
        colors[:, 1] = 0.2                   # Constant Green
        colors[:, 2] = 1.0 - z_normalized    # Blue decreases with height
        scan_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 5. Accumulate and Downsample
        self.global_pcd += scan_pcd
        self.global_pcd = self.global_pcd.voxel_down_sample(self.cfg.voxel_size)
        
        # Cap point cloud size
        if len(self.global_pcd.points) > self.cfg.max_points:
            print("Warning: Max map size reached. Consider larger voxel size.")

    def visualize(self):
        """Blocks and opens an Open3D visualization window."""
        if len(self.global_pcd.points) == 0:
            print("Map is empty. Nothing to visualize.")
            return
            
        print(f"Visualizing Map with {len(self.global_pcd.points)} points.")
        # Open3D coordinate frame for reference: Red=X, Green=Y, Blue=Z
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([self.global_pcd, axis])