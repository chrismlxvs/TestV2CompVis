import torch
import torch.nn as nn
import numpy as np

class GaussianModel(nn.Module):
    def __init__(self, ply_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self._load_ply(ply_path)
        
    def _load_ply(self, ply_path: str):
        """Minimal PLY reader extracting the LiDAR-initialized data."""
        with open(ply_path, 'rb') as f:
            header = []
            while True:
                line = f.readline().decode('utf-8').strip()
                header.append(line)
                if line == "end_header":
                    break
                    
            num_vertices = int([line.split()[-1] for line in header if "element vertex" in line][0])
            
            dtype = [
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                ('opacity', 'f4'),
                ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
            ]
            
            vertex_data = np.frombuffer(f.read(), dtype=dtype)
            
        # Convert to differentiable PyTorch parameters
        self.means = nn.Parameter(torch.tensor(
            np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1), 
            device=self.device
        ))
        
        self.scales = nn.Parameter(torch.tensor(
            np.stack([vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2']], axis=-1), 
            device=self.device
        ))
        
        self.quats = nn.Parameter(torch.tensor(
            np.stack([vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3']], axis=-1), 
            device=self.device
        ))
        
        self.opacities = nn.Parameter(torch.tensor(
            vertex_data['opacity'].reshape(-1, 1), 
            device=self.device
        ))
        
        # We use Spherical Harmonics degree 0 (diffuse color) for V1
        self.sh0 = nn.Parameter(torch.tensor(
            np.stack([vertex_data['f_dc_0'], vertex_data['f_dc_1'], vertex_data['f_dc_2']], axis=-1).reshape(-1, 1, 3), 
            device=self.device
        ))

    def get_render_params(self):
        """Applies necessary activations to parameters before rendering."""
        # Normalize quaternions
        quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        
        # Exponentiate scales to ensure they are positive
        scales = torch.exp(self.scales)
        
        # Sigmoid opacities to bound between 0 and 1, and squeeze to (N,) to match gsplat expectations
        opacities = torch.sigmoid(self.opacities).squeeze(-1)
        
        # Squeeze SH0 to (N, 3) and convert back to base RGB for the rasterizer
        colors = self.sh0.squeeze(1) * 0.28209479177387814 + 0.5
        
        return self.means, scales, quats, opacities, colors