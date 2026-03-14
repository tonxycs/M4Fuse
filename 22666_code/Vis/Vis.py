'''
Combine with Models.py doing Vis
'''

import os
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from scipy.ndimage import zoom


import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.cuda.amp import autocast, GradScaler
import nibabel as nib

from timm.layers import trunc_normal_
from mamba_ssm import Mamba
from scipy.ndimage import zoom, rotate
import scipy.stats as stats

# from monai.networks.nets import SegResNet, SwinUNETR  
from Network.Models import M4Fuse-T/S/B/L, UNet3D, LightMUNet, NormalU_Net, Segmamba, TransBTS, SegResNet, SwinUNETR, nnUnet   #10 cases compare models



class Config:
    test_data_dir = ""  
    model_weight = ".pth"
    modal_type = "flair" 
    slice_idx = 80  
    save_path = ".png"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Red(TC)、Green(ET)、Blue(WT)
    cmap = LinearSegmentedColormap.from_list(
        "brats", [(0,0,0,0), (1,0,0,0.6), (0,1,0,0.6), (0,0,1,0.6)], N=4
    )






def load_data(case_path, modal_type):

    flair_file = [f for f in os.listdir(case_path) if modal_type in f.lower() and 'nii.gz' in f][0]
    flair = nib.load(os.path.join(case_path, flair_file)).get_fdata().transpose(2, 0, 1)

    seg_file = [f for f in os.listdir(case_path) if 'seg' in f.lower() and 'nii.gz' in f][0]
    seg_gt = nib.load(os.path.join(case_path, seg_file)).get_fdata().transpose(2, 0, 1).astype(int)
    seg_gt[seg_gt == 4] = 3

    modal_data = []
    for mod in ['t1', 't1ce', 't2', 'flair']:
        mod_file = [f for f in os.listdir(case_path) if mod in f.lower() and 'nii.gz' in f][0]
        img = nib.load(os.path.join(case_path, mod_file)).get_fdata().transpose(2, 0, 1)
        modal_data.append(img)

    processed = []
    target_size = (64, 128, 128)
    for mod in modal_data:
        mod_norm = (mod - mod.mean()) / (mod.std() + 1e-8)

        depth_scale = target_size[0] / mod_norm.shape[0]
        mod_scaled_depth = zoom(mod_norm, zoom=(depth_scale, 1, 1), order=1)
 
        mod_resized = np.zeros((target_size[0], target_size[1], target_size[2]), dtype=np.float32)
        for d in range(target_size[0]):
            mod_resized[d] = cv2.resize(mod_scaled_depth[d], (target_size[2], target_size[1]), interpolation=cv2.INTER_LINEAR)
        processed.append(mod_resized)
    input_tensor = torch.from_numpy(np.stack(processed, axis=0)).unsqueeze(0).float().to(Config.device)
    return flair, seg_gt, input_tensor


case_path = Config.test_data_dir
flair, seg_gt, input_tensor = load_data(case_path, Config.modal_type)


# Seeing Models
model = M4Fuse(
    num_classes=4,
    input_channels=4,
    modalities=1
).to(Config.device)


checkpoint = torch.load(Config.model_weight, map_location=Config.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# # inference  option: 1.Others 2.Ours M4Fuse Class
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# with torch.no_grad():

    # dataset_id = torch.tensor([0], device=Config.device)
    # output = model(input_tensor, dataset_id)  
    # pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()


original_d, original_h, original_w = flair.shape
# 1. deep 64→155
depth_scale = original_d / pred.shape[0]
pred_scaled_depth = zoom(pred, zoom=(depth_scale, 1, 1), order=0).astype(np.int32)
# 2. H,W 128→240
pred_resized = np.zeros((original_d, original_h, original_w), dtype=np.int32)
for d in range(original_d):
    pred_resized[d] = cv2.resize(
        pred_scaled_depth[d].astype(np.float32),
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)


plt.figure(figsize=(8, 8))

plt.imshow(flair[Config.slice_idx], cmap="gray")

plt.imshow(pred_resized[Config.slice_idx], cmap=Config.cmap, alpha=0.6)
plt.title("M4Fuse-Base", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.savefig(Config.save_path, dpi=300, bbox_inches="tight")
plt.show()
