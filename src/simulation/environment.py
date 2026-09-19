import pybullet as pb
import pybullet_data
import numpy as np

class Environment:
    """Procedural environment generation."""
    def __init__(self, simulator, config):
        self.sim = simulator
        self.config = config

    def build(self):
        # Ground plane
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.loadURDF("plane.urdf")
        
        size_x, size_y = self.config.environment.room_size
        h = self.config.environment.wall_height
        
        # Procedural walls
        wall_color = [0.8, 0.8, 0.8, 1.0]
        self._create_box_obstacle([0, size_y/2, h/2], [size_x, 0.2, h], wall_color)
        self._create_box_obstacle([0, -size_y/2, h/2], [size_x, 0.2, h], wall_color)
        self._create_box_obstacle([size_x/2, 0, h/2], [0.2, size_y, h], wall_color)
        self._create_box_obstacle([-size_x/2, 0, h/2], [0.2, size_y, h], wall_color)

        # Random internal obstacles
        np.random.seed(42) # Deterministic for reproducible maps
        for _ in range(self.config.environment.num_obstacles):
            pos = [np.random.uniform(-size_x/3, size_x/3), 
                   np.random.uniform(-size_y/3, size_y/3), 
                   0.5]
            extents = np.random.uniform(0.5, 1.5, size=3).tolist()
            self._create_box_obstacle(pos, extents, [0.3, 0.5, 0.8, 1.0])

    def _create_box_obstacle(self, position, extents, color):
        visual_id = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[e/2 for e in extents], rgbaColor=color)
        collision_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[e/2 for e in extents])
        pb.createMultiBody(baseMass=0, 
                           baseCollisionShapeIndex=collision_id, 
                           baseVisualShapeIndex=visual_id, 
                           basePosition=position)