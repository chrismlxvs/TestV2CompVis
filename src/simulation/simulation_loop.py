from .simulator import Simulator
from .environment import Environment
from .robot import Robot
from .lidar_sensor import LidarSensor
from .camera_sensor import  CameraSensor

class SimulationLoop:
    def __init__(self, config):
        self.cfg = config
        self.sim = Simulator(timestep=self.cfg.simulation.timestep, gui=self.cfg.simulation.gui)
        self.sim.set_gravity(self.cfg.simulation.gravity)
        
        self.env = Environment(self.sim, self.cfg)
        self.env.build()
        
        self.robot = Robot(self.sim, self.cfg)
        self.lidar = LidarSensor(self.sim, self.robot, self.cfg)
        self.camera = CameraSensor(self.sim, self.robot, self.cfg)
        
        self.time = 0.0
        self.steps = 0
        
        # Calculate steps per LiDAR scan
        sim_hz = 1.0 / self.cfg.simulation.timestep
        self.lidar_step_interval = int(sim_hz / self.cfg.lidar.frequency)

    def step(self, v: float, w: float):
        """Advances the simulation by one step."""
        self.robot.set_velocity(v, w)
        self.sim.step()
        
        self.time += self.cfg.simulation.timestep
        self.steps += 1
        
        # Trigger LiDAR at requested frequency
        if self.steps % self.lidar_step_interval == 0:
            scan = self.lidar.scan(timestamp=self.time, visualize=True)
            