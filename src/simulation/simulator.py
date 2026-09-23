import pybullet as pb
import pybullet_data
import numpy as np
from typing import Tuple, List, Optional

class Simulator:
    """
    Abstracts PyBullet. No other file should import pybullet.
    Strictly uses ROS conventions (X-forward, Y-left, Z-up).
    """
    def __init__(self, gui: bool = True, timestep: float = 0.01):
        self.client = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setTimeStep(timestep)
        
        # Configure debug visualizer for ROS coordinate view (Z up, X forward)
        if gui:
            pb.resetDebugVisualizerCamera(cameraDistance=3.0, 
                                          cameraYaw=45, 
                                          cameraPitch=-30, 
                                          cameraTargetPosition=[0, 0, 0])
            
    def set_gravity(self, gravity: List[float]):
        pb.setGravity(gravity[0], gravity[1], gravity[2])

    def step(self):
        pb.stepSimulation()
        
    def is_connected(self) -> bool:
        try:
            return bool(pb.isConnected(self.client))
        except Exception:
            return False

    def disconnect(self):
        """Safely close PyBullet. Closing the GUI window first can already kill the connection."""
        try:
            if self.is_connected():
                pb.disconnect(self.client)
        except Exception:
            # Window already closed / OpenGL context already destroyed.
            pass

    # --- Math Helpers hiding PyBullet functions ---
    def euler_to_quaternion(self, rpy: List[float]) -> np.ndarray:
        return np.array(pb.getQuaternionFromEuler(rpy))

    def multiply_transforms(self, pos1, quat1, pos2, quat2) -> Tuple[np.ndarray, np.ndarray]:
        pos, quat = pb.multiplyTransforms(pos1, quat1, pos2, quat2)
        return np.array(pos), np.array(quat)
        
    def invert_transform(self, pos, quat) -> Tuple[np.ndarray, np.ndarray]:
        inv_pos, inv_quat = pb.invertTransform(pos, quat)
        return np.array(inv_pos), np.array(inv_quat)

    # --- Sensor Abstractions ---
    def ray_cast_batch(self, ray_starts: np.ndarray, ray_ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs batch raycasting.
        Returns:
            hit_fractions: np.ndarray of shape (N,)
            hit_positions: np.ndarray of shape (N, 3)
        """
        results = pb.rayTestBatch(ray_starts, ray_ends)
        hit_fractions = np.array([res[2] for res in results])
        hit_positions = np.array([res[3] for res in results])
        return hit_fractions, hit_positions

    # --- Debugging ---
    def draw_line(self, start: np.ndarray, end: np.ndarray, color: List[float], time: float = 0.1):
        pb.addUserDebugLine(start, end, lineColorRGB=color, lifeTime=time)