
'''
Original nii.gz----png
'''
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2


class Config:
    case_dir = ""  
    output_root = ""  
    slice_indices = [80, 85, 90]  

    modalities = ["flair", "t1", "t2", "t1ce"]


for mod in Config.modalities:
    mod_dir = os.path.join(Config.output_root, mod)
    os.makedirs(mod_dir, exist_ok=True)


def save_modality_slices(case_dir, mod, slice_indices, output_dir):


    mod_files = [f for f in os.listdir(case_dir) if mod in f.lower() and 'nii.gz' in f]
    if not mod_files:
        print(f"")
        return
    

    mod_path = os.path.join(case_dir, mod_files[0])
    mod_data = nib.load(mod_path).get_fdata().transpose(2, 0, 1)  
    

    for idx in slice_indices:
        if 0 <= idx < mod_data.shape[0]:
            slice_img = mod_data[idx]
  
            slice_norm = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8) * 255).astype(np.uint8)
            
        
            save_path = os.path.join(output_dir, f"{mod}_slice_{idx}.png")
   
            cv2.imwrite(save_path, slice_norm)
            print(f" {save_path}")
        else:
            print(f".")


for mod in Config.modalities:
    print(f".")
    mod_output_dir = os.path.join(Config.output_root, mod)
    save_modality_slices(
        case_dir=Config.case_dir,
        mod=mod,
        slice_indices=Config.slice_indices,
        output_dir=mod_output_dir
    )
