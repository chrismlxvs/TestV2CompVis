import time
import argparse
from omegaconf import OmegaConf
import pybullet as pb

# Add src/ to path for script execution
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from simulation.simulation_loop import SimulationLoop

def get_keyboard_command() -> tuple:
    """Reads arrow keys for teleop."""
    keys = pb.getKeyboardEvents()
    v, w = 0.0, 0.0
    
    # Arrow keys
    if pb.B3G_UP_ARROW in keys and keys[pb.B3G_UP_ARROW] & pb.KEY_IS_DOWN:
        v = 1.0
    if pb.B3G_DOWN_ARROW in keys and keys[pb.B3G_DOWN_ARROW] & pb.KEY_IS_DOWN:
        v = -1.0
    if pb.B3G_LEFT_ARROW in keys and keys[pb.B3G_LEFT_ARROW] & pb.KEY_IS_DOWN:
        w = 1.5
    if pb.B3G_RIGHT_ARROW in keys and keys[pb.B3G_RIGHT_ARROW] & pb.KEY_IS_DOWN:
        w = -1.5
        
    return v, w

def main():
    print("Loading configurations...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    sim_cfg = OmegaConf.load(os.path.join(base_dir, 'config', 'simulation.yaml'))
    robot_cfg = OmegaConf.load(os.path.join(base_dir, 'config', 'robot.yaml'))
    lidar_cfg = OmegaConf.load(os.path.join(base_dir, 'config', 'lidar.yaml'))
    camera_cfg = OmegaConf.load(os.path.join(base_dir, 'config', 'camera.yaml'))
    
    # Merge config
    config = OmegaConf.merge(sim_cfg, robot_cfg, lidar_cfg, camera_cfg)
    
    print("Initializing Simulation Environment...")
    loop = SimulationLoop(config)
    
    print("Simulation running! Use Arrow Keys to drive.")
    print("Press Ctrl+C to exit.")
    
    try:
        while True:
            v, w = get_keyboard_command()
            loop.step(v, w)
            
            # Maintain real-time execution roughly
            time.sleep(config.simulation.timestep)
            
    except KeyboardInterrupt:
        print("\nExiting simulation.")
    finally:
        loop.sim.disconnect()

if __name__ == "__main__":
    main()