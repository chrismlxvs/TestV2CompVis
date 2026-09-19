import numpy as np
from perception.icp import register_scans

class LidarOdometry:
    def __init__(self, initial_pose: np.ndarray):
        """
        Initializes odometry. We pass the very first Ground Truth pose 
        so the estimated trajectory starts in the same world coordinate frame.
        """
        self.current_pose = initial_pose.copy()
        self.previous_points = None
        
    def update(self, current_points: np.ndarray) -> np.ndarray:
        """
        Registers current scan against previous scan to estimate the new global pose.
        """
        if len(current_points) < 50:
            return self.current_pose # Ignore sparse/empty scans

        if self.previous_points is None:
            self.previous_points = current_points
            return self.current_pose
            
        # T_rel transforms points from the Current frame to the Previous frame
        T_rel, fitness = register_scans(
            source_points=current_points,
            target_points=self.previous_points
        )
        
        # Compose transforms: T_world_curr = T_world_prev * T_prev_curr
        self.current_pose = self.current_pose @ T_rel
        self.previous_points = current_points
        
        return self.current_pose