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
        self.smoothing_factor = 0.15  # Smoothing rate [0.01 (very smooth) - 1.0 (raw/jerky)]
        
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
        pb.changeDynamics(robot_id, -1, linearDamping=0.05, angularDamping=0.05)

        # Tuned Dynamics for 4WD Skid-Steer
        for i in range(4):
            pb.changeDynamics(
                robot_id, 
                i, 
                lateralFriction=0.35,   
                spinningFriction=0.01,  # Smooths low-speed rotations
                rollingFriction=0.01    # Reduces sudden stops
            ) 
            
        return robot_id

    def set_velocity(self, linear: float, angular: float):
        """4-Wheel Skid-Steer kinematics with smooth target blending"""
        target_v = np.clip(linear, -self.cfg.max_linear_velocity, self.cfg.max_linear_velocity)
        target_w = np.clip(angular, -self.cfg.max_angular_velocity, self.cfg.max_angular_velocity)
        
        # Exponential Moving Average / Low-Pass Filter for smooth acceleration
        self.current_v += (target_v - self.current_v) * self.smoothing_factor
        self.current_w += (target_w - self.current_w) * self.smoothing_factor
        
        L = self.cfg.wheel_base
        R = self.cfg.wheel_radius
        
        # Side linear velocities using smoothed commands
        v_left = self.current_v - (self.current_w * L / 2.0)
        v_right = self.current_v + (self.current_w * L / 2.0)
        
        # Angular wheel velocities (rad/s)
        omega_left = v_left / R
        omega_right = v_right / R
        
        max_torque = 25.0  # Reduced torque cap prevents aggressive jerk on startup
        
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