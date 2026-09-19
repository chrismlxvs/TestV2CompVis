import numpy as np

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion [x, y, z, w] into a 3x3 rotation matrix.
    Assumes Hamilton convention (scalar last) and normalized quaternion.
    """
    q = np.asarray(q, dtype=np.float64)
    # Ensure it's normalized to avoid scaling artifacts in rotation
    q = q / np.linalg.norm(q)
    x, y, z, w = q

    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
    ], dtype=np.float64)
    
    return R

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to a quaternion [x, y, z, w].
    Uses a robust trace-based method to avoid singularities.
    """
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        # Find the largest diagonal element to pivot
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Creates a rotation matrix from Euler angles (in radians).
    Convention: ZYX intrinsic (Yaw -> Pitch -> Roll).
    This maps to: R = R_z(yaw) * R_y(pitch) * R_x(roll)
    """
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)

    R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    return R_z @ R_y @ R_x