import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class GSDataset(Dataset):
    def __init__(self, dataset_path: str, device: str = "cuda", manifest_name: str = "transforms.json"):
        self.device = device
        self.dataset_path = dataset_path
        
        with open(os.path.join(dataset_path, manifest_name), "r") as f:
            self.meta = json.load(f)
            
        self.K = torch.tensor([
            [self.meta["fl_x"], 0, self.meta["cx"]],
            [0, self.meta["fl_y"], self.meta["cy"]],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        self.width = self.meta["w"]
        self.height = self.meta["h"]
        self.frames = self.meta["frames"]
        
    def __len__(self):
        return len(self.frames)
        
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # Load image and convert to float tensor [C, H, W]
        img_path = os.path.join(self.dataset_path, frame["file_path"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).to(self.device)
        
        # Load Extrinsics (Cam-to-World)
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=self.device)
        
        # gsplat expects World-to-Cam (w2c) matrices
        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        
        return {"image": img_tensor, "R": R, "T": T, "K": self.K}