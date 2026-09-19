import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from typing import Dict, Tuple

class LidarGaussianInitializer:
    def __init__(self, config):
        self.cfg = config.gaussian_splatting
        self.init_cfg = config.initialization

    def compute_knn_scales(self, points: np.ndarray, k: int = 3) -> np.ndarray:
        """
        Computes initial anisotropic scales based on average distance to k-nearest neighbors.
        preventing initial Gaussians from overlapping excessively or leaving empty voids.
        """
        if len(points) < k + 1:
            return np.ones_like(points) * 0.1
            
        tree = KDTree(points)
        distances, _ = tree.query(points, k=k+1)  # First neighbor is point itself
        mean_distances = np.mean(distances[:, 1:], axis=1, keepdims=True)
        
        # Clamp minimum scale to prevent numerical instability in rasterizer
        mean_distances = np.maximum(mean_distances, 1e-4)
        
        # Replicate mean distance along all 3 local axes (isotropic base scale)
        scales = np.tile(mean_distances * self.init_cfg.scale_factor, (1, 3))
        return scales

    def generate_initial_gaussians(self, pcd: o3d.geometry.PointCloud) -> Dict[str, np.ndarray]:
        """
        Converts an Open3D point cloud into 3D Gaussian initialization attributes.
        Returns dictionary containing positions, colors, scales, rotations, and opacities.
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points) * 0.5
        
        num_points = len(points)
        print(f"Initializing {num_points} 3D Gaussians from LiDAR points...")
        
        # 1. Position Mean (\mu)
        positions = points.copy()
        
        # 2. Scale (\mathbf{s}): Log-space scales for optimizer compatibility
        raw_scales = self.compute_knn_scales(points, k=self.init_cfg.k_neighbors)
        log_scales = np.log(raw_scales)
        
        # 3. Rotation (\mathbf{q}): Identity quaternions [w=1, x=0, y=0, z=0]
        rotations = np.zeros((num_points, 4))
        rotations[:, 0] = 1.0  
        
        # 4. Opacity (o): Inverse-logit space initialization
        opacities_linear = np.full((num_points, 1), self.init_cfg.default_opacity)
        opacities_logit = np.log(opacities_linear / (1.0 - opacities_linear))
        
        # 5. Base RGB Features (converted to 0th-order Spherical Harmonics coefficient)
        # C0 coefficient = (RGB - 0.5) / 0.28209479177387814
        sh_c0 = (colors - 0.5) / 0.28209479177387814

        return {
            "positions": positions,
            "sh_c0": sh_c0,
            "log_scales": log_scales,
            "rotations": rotations,
            "opacities_logit": opacities_logit
        }

    def save_ply(self, gaussians: Dict[str, np.ndarray], filepath: str):
        """Saves Gaussian attributes to PLY format for standard 3DGS training pipelines."""
        num_points = len(gaussians["positions"])
        
        # Create structured array matching standard 3DGS PLY format
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ]
        
        elements = np.empty(num_points, dtype=dtype)
        
        elements['x'] = gaussians["positions"][:, 0]
        elements['y'] = gaussians["positions"][:, 1]
        elements['z'] = gaussians["positions"][:, 2]
        
        elements['f_dc_0'] = gaussians["sh_c0"][:, 0]
        elements['f_dc_1'] = gaussians["sh_c0"][:, 1]
        elements['f_dc_2'] = gaussians["sh_c0"][:, 2]
        
        elements['opacity'] = gaussians["opacities_logit"][:, 0]
        
        elements['scale_0'] = gaussians["log_scales"][:, 0]
        elements['scale_1'] = gaussians["log_scales"][:, 1]
        elements['scale_2'] = gaussians["log_scales"][:, 2]
        
        elements['rot_0'] = gaussians["rotations"][:, 0]
        elements['rot_1'] = gaussians["rotations"][:, 1]
        elements['rot_2'] = gaussians["rotations"][:, 2]
        elements['rot_3'] = gaussians["rotations"][:, 3]
        
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
        with open(filepath, 'wb') as f:
            f.write(header.encode('utf-8'))
            f.write(elements.tobytes())
            
        print(f"Successfully exported LiDAR Gaussian initialization to: {filepath}")