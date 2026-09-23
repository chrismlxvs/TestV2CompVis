import pybullet as pb
import numpy as np
from typing import Tuple

class Robot:
    """Procedural 4-Wheel Skid-Steer Robot using ROS coordinate convention."""
    def __init__(self, simulator, config):
        self.sim = simulator
        self.cfg = config.robot
        
        # Joint mapping
        self.fl_wheel_idx = 0  # Front-Left
        self.rl_wheel_idx = 1  # Rear-Left
        self.fr_wheel_idx = 2  # Front-Right
        self.rr_wheel_idx = 3  # Rear-Right
        
        # State tracking for velocity smoothing
        self.current_v = 0.0
        self.current_w = 0.0
        self.smoothing_factor = float(getattr(self.cfg, "velocity_smoothing", 0.45))
        self.brake_smoothing = float(getattr(self.cfg, "brake_smoothing", 0.65))
        self.wheel_motor_force = float(getattr(self.cfg, "wheel_motor_force", 80.0))
        self.wheel_lateral_friction = float(getattr(self.cfg, "wheel_lateral_friction", 1.2))
        self.chassis_linear_damping = float(getattr(self.cfg, "chassis_linear_damping", 0.02))
        self.chassis_angular_damping = float(getattr(self.cfg, "chassis_angular_damping", 0.08))
        
        self.robot_id = self._build_robot()
        
    def _build_robot(self):
        # 1. Base Chassis
        base_extents = [e/2 for e in self.cfg.chassis_size]
        base_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=base_extents)
        base_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=base_extents, rgbaColor=[0.8, 0.2, 0.2, 1])

        # 2. Wheels (Cylinders in Y-axis alignment for ROS)
        wheel_radius = self.cfg.wheel_radius
        wheel_length = 0.05
        wheel_quat = pb.getQuaternionFromEuler([np.pi/2, 0, 0])
        
        wheel_col = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=wheel_radius, height=wheel_length)
        wheel_vis = pb.createVisualShape(pb.GEOM_CYLINDER, radius=wheel_radius, length=wheel_length, rgbaColor=[0.1, 0.1, 0.1, 1])

        # Link Definitions for 4 wheels
        y_offset = self.cfg.wheel_base / 2
        x_offset = self.cfg.wheel_offset_x
        z_offset = -self.cfg.chassis_size[2]/2 # Bottom of chassis
        
        linkMasses = [self.cfg.wheel_mass] * 4
        linkCollisionShapeIndices = [wheel_col] * 4
        linkVisualShapeIndices = [wheel_vis] * 4
        
        linkPositions = [
            [x_offset, y_offset, z_offset],    
            [-x_offset, y_offset, z_offset],   
            [x_offset, -y_offset, z_offset],   
            [-x_offset, -y_offset, z_offset]   
        ]
        
        linkOrientations = [wheel_quat] * 4
        linkInertialFramePositions = [[0,0,0]] * 4
        linkInertialFrameOrientations = [[0,0,0,1]] * 4
        linkParentIndices = [0, 0, 0, 0]
        linkJointTypes = [pb.JOINT_REVOLUTE] * 4
        linkJointAxis = [[0, 0, 1]] * 4 

        init_pos = self.cfg.initial_position
        init_quat = self.sim.euler_to_quaternion(self.cfg.initial_orientation)

        robot_id = pb.createMultiBody(
            baseMass=self.cfg.chassis_mass,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=init_pos,
            baseOrientation=init_quat,
            linkMasses=linkMasses,
            linkCollisionShapeIndices=linkCollisionShapeIndices,
            linkVisualShapeIndices=linkVisualShapeIndices,
            linkPositions=linkPositions,
            linkOrientations=linkOrientations,
            linkInertialFramePositions=linkInertialFramePositions,
            linkInertialFrameOrientations=linkInertialFrameOrientations,
            linkParentIndices=linkParentIndices,
            linkJointTypes=linkJointTypes,
            linkJointAxis=linkJointAxis
        )
        
        # Chassis Damping for stability
        pb.changeDynamics(
            robot_id,
            -1,
            linearDamping=self.chassis_linear_damping,
            angularDamping=self.chassis_angular_damping,
        )

        # Tuned Dynamics for 4WD Skid-Steer
        for i in range(4):
            pb.changeDynamics(
                robot_id, 
                i, 
                lateralFriction=self.wheel_lateral_friction,
                spinningFriction=0.02,
                rollingFriction=0.005,
            ) 
            
        return robot_id

    def set_velocity(self, linear: float, angular: float):
        """4-Wheel Skid-Steer kinematics with responsive target blending."""
        target_v = np.clip(linear, -self.cfg.max_linear_velocity, self.cfg.max_linear_velocity)
        target_w = np.clip(angular, -self.cfg.max_angular_velocity, self.cfg.max_angular_velocity)

        # Accelerate with normal smoothing; brake harder when the stick is released.
        alpha_v = self.brake_smoothing if abs(target_v) < 1e-6 else self.smoothing_factor
        alpha_w = self.brake_smoothing if abs(target_w) < 1e-6 else self.smoothing_factor
        self.current_v += (target_v - self.current_v) * alpha_v
        self.current_w += (target_w - self.current_w) * alpha_w
        
        L = self.cfg.wheel_base
        R = self.cfg.wheel_radius
        
        # Side linear velocities using smoothed commands
        v_left = self.current_v - (self.current_w * L / 2.0)
        v_right = self.current_v + (self.current_w * L / 2.0)
        
        # Angular wheel velocities (rad/s)
        omega_left = v_left / R
        omega_right = v_right / R
        
        max_torque = self.wheel_motor_force
        
        # Left wheels
        pb.setJointMotorControlArray(
            self.robot_id, 
            [self.fl_wheel_idx, self.rl_wheel_idx], 
            pb.VELOCITY_CONTROL, 
            targetVelocities=[omega_left, omega_left], 
            forces=[max_torque, max_torque]
        )
        
        # Right wheels
        pb.setJointMotorControlArray(
            self.robot_id, 
            [self.fr_wheel_idx, self.rr_wheel_idx], 
            pb.VELOCITY_CONTROL, 
            targetVelocities=[omega_right, omega_right], 
            forces=[max_torque, max_torque]
        )

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns Ground Truth (GT) Position and Quaternion in World frame"""
        pos, quat = pb.getBasePositionAndOrientation(self.robot_id)
        return np.array(pos), np.array(quat)