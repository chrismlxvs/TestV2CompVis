import os
import sys
import torch
import torch.optim as optim
from tqdm import tqdm
import cv2
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from splatting.dataset import GSDataset
from splatting.gaussian_model import GaussianModel
from gsplat.rendering import rasterization

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config = OmegaConf.load(os.path.join(base_dir, 'config', 'gaussian_splatting.yaml'))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not detected. Gaussian Splatting requires a GPU.")
        sys.exit(1)
        
    dataset_path = os.path.join(base_dir, config.export.output_dir)
    dataset = GSDataset(dataset_path, device=device)
    
    ply_path = os.path.join(dataset_path, "lidar_initialization.ply")
    model = GaussianModel(ply_path, device=device)
    
    # Configure parameter-specific learning rates
    optimizer = optim.Adam([
        {'params': [model.means], 'lr': config.gaussian_splatting.learning_rate_position},
        {'params': [model.scales], 'lr': config.gaussian_splatting.learning_rate_scaling},
        {'params': [model.quats], 'lr': config.gaussian_splatting.learning_rate_rotation},
        {'params': [model.opacities], 'lr': config.gaussian_splatting.learning_rate_opacity},
        {'params': [model.sh0], 'lr': config.gaussian_splatting.learning_rate_feature}
    ])
    
    iterations = config.gaussian_splatting.iterations
    progress_bar = tqdm(range(iterations), desc="Training 3DGS")
    
    for step in progress_bar:
        # 1. Randomly sample a camera viewpoint from our trajectory
        idx = torch.randint(0, len(dataset), (1,)).item()
        batch = dataset[idx]
        
        gt_image = batch["image"]
        R, T, K = batch["R"], batch["T"], batch["K"]
        
        # 2. Get activated Gaussian parameters
        means, scales, quats, opacities, sh0 = model.get_render_params()
        
        # 3. Rasterize Gaussians to 2D image
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh0,
            viewmats=torch.cat([R, T.unsqueeze(1)], dim=1).unsqueeze(0), 
            Ks=K.unsqueeze(0),
            width=dataset.width,
            height=dataset.height,
            packed=False
        )
        
        # Rearrange render shape from [1, H, W, 3] to [3, H, W] to match GT image
        render_img = render_colors[0].permute(2, 0, 1)
        
        # 4. Compute Loss (L1)
        loss = torch.abs(render_img - gt_image).mean()
        
        # 5. Backpropagate & Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # 6. Periodic Visualization
        if step % 1000 == 0:
            out_img = render_img.detach().permute(1, 2, 0).cpu().numpy()
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("GS Training Progress", out_img)
            cv2.waitKey(1)
            
    print("Training complete!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()