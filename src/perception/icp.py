import numpy as np
import open3d as o3d
from typing import Tuple

def register_scans(source_points: np.ndarray, target_points: np.ndarray, 
                   voxel_size: float = 0.1, max_corr_dist: float = 0.5, 
                   init_guess: np.ndarray = np.eye(4)) -> Tuple[np.ndarray, float]:
    """
    Finds the SE(3) transformation that aligns source_points to target_points.
    """
    # 1. Convert to Open3D format
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    
    # 2. Voxel downsample for speed
    source = source.voxel_down_sample(voxel_size)
    target = target.voxel_down_sample(voxel_size)
    
    # 3. Estimate normals for Point-to-Plane ICP (prevents sliding in corridors)
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    source.estimate_normals(search_param)
    target.estimate_normals(search_param)
    
    # 4. Perform ICP
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_corr_dist, init_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    return result.transformation, result.fitness