import numpy as np
import pybullet as pb
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CameraFrame:
    timestamp: float
    rgb: np.ndarray
    intrinsics: np.ndarray
    t_world_cam_pos: np.ndarray
    t_world_cam_quat: np.ndarray

class CameraSensor:
    def __init__(self, simulator, robot, config):
        self.sim = simulator
        self.robot = robot
        self.cfg = config.camera
        
        self.width = self.cfg.width
        self.height = self.cfg.height
        
        # Calculate Intrinsics (Pinhole Camera Model)
        cx = self.width / 2.0
        cy = self.height / 2.0
        f = (self.width / 2.0) / np.tan(np.radians(self.cfg.fov) / 2.0)
        self.intrinsics = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0,  1]
        ])
        
        self.proj_matrix = pb.computeProjectionMatrixFOV(
            fov=self.cfg.fov, aspect=self.width/self.height, nearVal=0.1, farVal=100.0
        )
        self.t_base_cam_pos = np.array([self.cfg.offset_x, self.cfg.offset_y, self.cfg.offset_z])

    def capture(self, timestamp: float) -> CameraFrame:
        base_pos, base_quat = self.robot.get_pose()
        
        # Camera Pose in World
        cam_pos, cam_quat = self.sim.multiply_transforms(
            base_pos, base_quat, self.t_base_cam_pos, [0,0,0,1]
        )
        
        # Calculate target position (forward +X in ROS) and up vector (+Z in ROS)
        target_pos, _ = self.sim.multiply_transforms(cam_pos, cam_quat, [1,0,0], [0,0,0,1])
        up_world, _ = self.sim.multiply_transforms([0,0,0], cam_quat, [0,0,1], [0,0,0,1])
        
        view_matrix = pb.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_world
        )
        
        _, _, rgb_pixels, _, _ = pb.getCameraImage(
            width=self.width, height=self.height,
            viewMatrix=view_matrix, projectionMatrix=self.proj_matrix,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Reshape and drop alpha channel
        rgb = np.reshape(rgb_pixels, (self.height, self.width, 4))[:, :, :3]
        
        return CameraFrame(
            timestamp=timestamp, rgb=rgb, intrinsics=self.intrinsics,
            t_world_cam_pos=np.array(cam_pos), t_world_cam_quat=np.array(cam_quat)
        )