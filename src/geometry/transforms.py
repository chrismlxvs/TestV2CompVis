import numpy as np
from scipy.spatial.transform import Rotation as R

def make_transform_matrix(translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Creates a 4x4 SE(3) transformation matrix.
    Args:
        translation: [x, y, z]
        quaternion: [x, y, z, w] (SciPy/ROS standard)
    Returns:
        4x4 numpy array representing the rigid body transform.
    """
    T = np.eye(4)
    # scipy expects scalar-last quaternion format: [x, y, z, w], exactly what PyBullet outputs
    T[:3, :3] = R.from_quat(quaternion).as_matrix()
    T[:3, 3] = translation
    return T

def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Applies a 4x4 transform to an Nx3 array of points.
    """
    if len(points) == 0:
        return points
        
    N = points.shape[0]
    # Convert to homogeneous coordinates (Nx4)
    points_hom = np.hstack((points, np.ones((N, 1))))
    
    # Apply transform and strip the homogeneous coordinate
    transformed = (T @ points_hom.T).T
    return transformed[:, :3]