import argparse
import json
import os
import sys

import torch
from omegaconf import OmegaConf
from gsplat.rendering import rasterization

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from splatting.dataset import GSDataset
from splatting.gaussian_model import GaussianModel


def compute_psnr(prediction, target):
    mse = torch.mean((prediction - target) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return float(-10.0 * torch.log10(torch.tensor(mse)).item())


def compute_ssim(prediction, target):
    window_size = 11
    padding = window_size // 2
    prediction = torch.nn.functional.pad(prediction, (padding,) * 4, mode="reflect")
    target = torch.nn.functional.pad(target, (padding,) * 4, mode="reflect")
    mean_prediction = torch.nn.functional.avg_pool2d(prediction, window_size, stride=1)
    mean_target = torch.nn.functional.avg_pool2d(target, window_size, stride=1)
    variance_prediction = torch.nn.functional.avg_pool2d(prediction ** 2, window_size, stride=1) - mean_prediction ** 2
    variance_target = torch.nn.functional.avg_pool2d(target ** 2, window_size, stride=1) - mean_target ** 2
    covariance = torch.nn.functional.avg_pool2d(prediction * target, window_size, stride=1) - mean_prediction * mean_target
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)
    denominator = (mean_prediction ** 2 + mean_target ** 2 + c1) * (variance_prediction + variance_target + c2)
    return float((numerator / (denominator + 1e-8)).mean().item())


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Gaussian Splatting model on held-out frames.")
    parser.add_argument("--dataset", default=None, help="Dataset directory; defaults to config output_dir")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to config checkpoint_path")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    config = OmegaConf.load(os.path.join(base_dir, "config", "gaussian_splatting.yaml"))
    dataset_path = os.path.abspath(args.dataset or os.path.join(base_dir, config.export.output_dir))
    checkpoint_path = os.path.abspath(args.checkpoint or os.path.join(base_dir, config.export.checkpoint_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = GSDataset(dataset_path, device=device, manifest_name="transforms_test.json")
    model = GaussianModel(os.path.join(dataset_path, "lidar_initialization.ply"), device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        means, scales, quats, opacities, colors = model.get_render_params()
        for index in range(len(dataset)):
            batch = dataset[index]
            viewmat = torch.eye(4, device=device)
            viewmat[:3, :3] = batch["R"]
            viewmat[:3, 3] = batch["T"]
            rendered, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat.unsqueeze(0),
                Ks=batch["K"].unsqueeze(0),
                width=dataset.width,
                height=dataset.height,
                packed=False,
            )
            prediction = rendered[0].permute(2, 0, 1).clamp(0.0, 1.0).unsqueeze(0)
            target = batch["image"].unsqueeze(0)
            psnr_values.append(compute_psnr(prediction, target))
            ssim_values.append(compute_ssim(prediction, target))

    results = {
        "frames": len(dataset),
        "psnr": float(sum(psnr_values) / len(psnr_values)),
        "ssim": float(sum(ssim_values) / len(ssim_values)),
        "device": device,
    }
    result_path = os.path.join(dataset_path, "evaluation_test.json")
    with open(result_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved evaluation results to: {result_path}")


if __name__ == "__main__":
    main()
