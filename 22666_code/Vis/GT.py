
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2


class Config:
    case_dir = ""  
    output_path = ""  
    slice_idx = 80  

    cmap = LinearSegmentedColormap.from_list(
        "brats", [(0,0,0,0), (1,0,0,0.6), (0,1,0,0.6), (0,0,1,0.6)], N=4
    )


def load_and_visualize_gt(case_dir, slice_idx, output_path, cmap):
  
    seg_files = [f for f in os.listdir(case_dir) if 'seg' in f.lower() and 'nii.gz' in f]
    if not seg_files:
        raise FileNotFoundError(".")
    

    seg_path = os.path.join(case_dir, seg_files[0])
    seg_data = nib.load(seg_path).get_fdata().transpose(2, 0, 1).astype(int)
    

    seg_data[seg_data == 4] = 3
    

    if slice_idx < 0 or slice_idx >= seg_data.shape[0]:
        raise ValueError(f".")
    

    flair_files = [f for f in os.listdir(case_dir) if 'flair' in f.lower() and 'nii.gz' in f]
    flair_data = nib.load(os.path.join(case_dir, flair_files[0])).get_fdata().transpose(2, 0, 1)
    flair_slice = flair_data[slice_idx]
    

    plt.figure(figsize=(8, 8))
    plt.imshow(flair_slice, cmap="gray") 
    plt.imshow(seg_data[slice_idx], cmap=cmap, alpha=0.6)  
    plt.title("(a) Ground Truth", fontsize=16)  
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"{output_path}")


load_and_visualize_gt(
    case_dir=Config.case_dir,
    slice_idx=Config.slice_idx,
    output_path=Config.output_path,
    cmap=Config.cmap
)

