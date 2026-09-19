import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class LidarScan:
    timestamp: float
    local_points: np.ndarray  # Points in LiDAR frame (N, 3)
    ranges: np.ndarray        # Ray ranges (N,)

class LidarSensor:
    def __init__(self, simulator, robot, config):
        self.sim = simulator
        self.robot = robot
        self.cfg = config.lidar
        
        self.t_base_lidar_pos = np.array([self.cfg.offset_x, self.cfg.offset_y, self.cfg.offset_z])
        self.t_base_lidar_quat = np.array([0.0, 0.0, 0.0, 1.0]) # Identity

        self.local_ray_ends = self._generate_ray_patterns()
        
    def _generate_ray_patterns(self) -> np.ndarray:
        """Precomputes ray end points in the LiDAR local coordinate frame."""
        h_res = np.radians(self.cfg.horizontal_resolution)
        h_angles = np.arange(0, np.radians(self.cfg.horizontal_fov), h_res)
        
        if self.cfg.channels > 1:
            v_min, v_max = np.radians(self.cfg.vertical_fov)
            v_angles = np.linspace(v_min, v_max, self.cfg.channels)
        else:
            v_angles = np.array([0.0])
            
        rays = []
        for v in v_angles:
            for h in h_angles:
                # Spherical to Cartesian (ROS convention: X forward, Y left, Z up)
                x = self.cfg.range_max * np.cos(v) * np.cos(h)
                y = self.cfg.range_max * np.cos(v) * np.sin(h)
                z = self.cfg.range_max * np.sin(v)
                rays.append([x, y, z])
                
        return np.array(rays)

    def scan(self, timestamp: float, visualize: bool = False) -> Optional[LidarScan]:
        # 1. Get GT Robot Pose (T_world_base)
        robot_pos, robot_quat = self.robot.get_pose()
        
        # 2. Get LiDAR Pose in World (T_world_lidar = T_world_base * T_base_lidar)
        lidar_world_pos, lidar_world_quat = self.sim.multiply_transforms(
            robot_pos, robot_quat, 
            self.t_base_lidar_pos, self.t_base_lidar_quat
        )
        
        # 3. Transform rays to World frame for raycasting
        num_rays = len(self.local_ray_ends)
        world_ray_starts = np.tile(lidar_world_pos, (num_rays, 1))
        
        world_ray_ends = np.zeros_like(self.local_ray_ends)
        for i in range(num_rays):
            end_pos, _ = self.sim.multiply_transforms(
                lidar_world_pos, lidar_world_quat,
                self.local_ray_ends[i], [0,0,0,1]
            )
            world_ray_ends[i] = end_pos

        # 4. Perform Raycast
        hit_fractions, hit_positions = self.sim.ray_cast_batch(world_ray_starts, world_ray_ends)
        
        # 5. Process hits
        valid_hits = hit_fractions < 1.0 # fraction == 1.0 means no hit
        valid_ranges = hit_fractions[valid_hits] * self.cfg.range_max
        valid_world_points = hit_positions[valid_hits]
        
        # Keep points strictly in LiDAR range limits
        range_mask = valid_ranges >= self.cfg.range_min
        valid_ranges = valid_ranges[range_mask]
        valid_world_points = valid_world_points[range_mask]

        if len(valid_world_points) == 0:
            return None

        # 6. Transform hit points back to local LiDAR frame
        inv_pos, inv_quat = self.sim.invert_transform(lidar_world_pos, lidar_world_quat)
        local_points = np.zeros_like(valid_world_points)
        for i in range(len(valid_world_points)):
            p, _ = self.sim.multiply_transforms(inv_pos, inv_quat, valid_world_points[i], [0,0,0,1])
            local_points[i] = p

        # Optional: Subsample debug visualization so PyBullet doesn't choke
        if visualize and len(valid_world_points) > 0:
            subsample = max(1, len(valid_world_points) // 50)
            for point in valid_world_points[::subsample]:
                self.sim.draw_line(lidar_world_pos, point, [1, 0, 0], time=0.1)

        return LidarScan(timestamp=timestamp, local_points=local_points, ranges=valid_ranges)