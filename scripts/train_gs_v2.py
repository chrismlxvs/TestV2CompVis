import os
import sys
import torch
import torch.optim as optim
from tqdm import tqdm
import cv2
import numpy as np
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from splatting.dataset import GSDataset
from splatting.gaussian_model import GaussianModel
from gsplat.rendering import rasterization


def save_gaussians_ply(model, filepath):
    """Export trained optimizer-space Gaussian parameters as a binary PLY."""
    attributes = np.empty(model.means.shape[0], dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"),
        ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])

    means = model.means.detach().cpu().numpy()
    sh0 = model.sh0.detach().squeeze(1).cpu().numpy()
    opacities = model.opacities.detach().squeeze(-1).cpu().numpy()
    scales = model.scales.detach().cpu().numpy()
    quats = model.quats.detach().cpu().numpy()

    attributes["x"], attributes["y"], attributes["z"] = means.T
    attributes["f_dc_0"], attributes["f_dc_1"], attributes["f_dc_2"] = sh0.T
    attributes["opacity"] = opacities
    attributes["scale_0"], attributes["scale_1"], attributes["scale_2"] = scales.T
    attributes["rot_0"], attributes["rot_1"], attributes["rot_2"], attributes["rot_3"] = quats.T

    header = f"""ply
format binary_little_endian 1.0
element vertex {len(attributes)}
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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(attributes.tobytes())


def check_gsplat_backend():
    """Fail early with a useful message when gsplat CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. gsplat training requires a CUDA-capable GPU "
            "and a working PyTorch CUDA installation."
        )

    try:
        from gsplat.cuda._backend import _C
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Could not load gsplat's CUDA backend. Install the CUDA toolkit "
            "(including nvcc), then reinstall gsplat in this virtual environment."
        ) from exc

    if _C is None:
        raise RuntimeError(
            "gsplat's CUDA backend is unavailable. PyTorch can see the GPU, "
            "but gsplat could not load its native extension. Install the CUDA "
            "toolkit, then reinstall gsplat."
        )


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config = OmegaConf.load(os.path.join(base_dir, 'config', 'gaussian_splatting.yaml'))
    
    check_gsplat_backend()
    device = "cuda"

    dataset_path = os.path.join(base_dir, config.export.output_dir)
    dataset = GSDataset(dataset_path, device=device, manifest_name="transforms_train.json")
    if len(dataset) == 0:
        raise RuntimeError("The training manifest contains no frames.")

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
        
        # --- SHAPE FIX: Construct the strict (C, 4, 4) and (C, 3, 3) tensors gsplat requires ---
        viewmat = torch.eye(4, device=device)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = T
        viewmats = viewmat.unsqueeze(0)  # Shape: (1, 4, 4)
        Ks = K.unsqueeze(0)              # Shape: (1, 3, 3)
        
        # 2. Get activated Gaussian parameters
        means, scales, quats, opacities, colors = model.get_render_params()
        
        # 3. Rasterize Gaussians to 2D image
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats, 
            Ks=Ks,
            width=dataset.width,
            height=dataset.height,
            packed=False
        )
        
        # Rearrange render shape from [C, H, W, D] to [D, H, W] to match GT image
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
            out_img = np.clip(out_img, 0.0, 1.0) # Clamp to prevent OpenCV artifacting
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("GS Training Progress", out_img)
            cv2.waitKey(1)
            
    print("Training complete!")
    checkpoint_path = os.path.join(base_dir, config.export.checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "steps": iterations}, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")

    ply_output_path = os.path.join(dataset_path, "trained_gaussians.ply")
    save_gaussians_ply(model, ply_output_path)
    print(f"Saved trained Gaussian PLY to: {ply_output_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()