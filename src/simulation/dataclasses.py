# src/simulation/dataclasses.py
from dataclasses import dataclass
import numpy as np

@dataclass
class PoseSE3:
    """Representation of an SE(3) pose in ROS convention (X-forward, Y-left, Z-up)."""
    position: np.ndarray      # [x, y, z]
    orientation: np.ndarray   # Quaternion [x, y, z, w]

    def to_matrix(self) -> np.ndarray:
        from src.geometry.transforms import make_transform_from_quat
        return make_transform_from_quat(self.orientation, self.position)

@dataclass
class LiDARScan:
    """Simulator-agnostic LiDAR scan measurement payload."""
    timestamp: float
    points: np.ndarray        # N x 3 array of valid 3D hit points in local frame
    ranges: np.ndarray        # N array of raw range measurements
    intensities: np.ndarray   # N array of returns/reflectivities
    origin_transform: np.ndarray # 4x4 SE(3) transform T_world_sensor at scan time