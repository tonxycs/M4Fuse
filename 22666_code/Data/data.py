import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, random_split
from scipy.ndimage import zoom, rotate
from Preprocessing import apply_augmentations, add_gaussian_noise, adjust_contrast


# BraTS2021/BarTS2019   e.g. 2021(64×128×128， IRS:2.09M)

# Train / Valid / Eval = TVE Mode training way(spliting 6:2:2)

class BraTS20213DDataset(Dataset):
    def __init__(self, data_dir,
                 modal_types=['t1', 't1ce', 't2', 'flair'],
                 target_size=(64, 128, 128),
                 normalize=True,
                 augment=False):
        self.data_dir = data_dir
        self.modal_types = modal_types
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment
        self.valid_cases = self._scan_valid_cases()
        if len(self.valid_cases) == 0:
            raise ValueError(f"No valid cases found in {data_dir}")

    def _scan_valid_cases(self):
        valid_cases = []
        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            has_all_modal = True
            for mod in self.modal_types:
                mod_files = [f for f in os.listdir(folder_path) if mod in f and 'nii.gz' in f]
                if len(mod_files) == 0:
                    has_all_modal = False
                    break

            seg_files = [f for f in os.listdir(folder_path) if 'seg' in f and 'nii.gz' in f]
            if has_all_modal and len(seg_files) > 0:
                valid_cases.append(folder)
        return valid_cases

    def __len__(self):
        return len(self.valid_cases)

    def __getitem__(self, idx):
        case_name = self.valid_cases[idx]
        case_path = os.path.join(self.data_dir, case_name)


        modal_data = []
        for mod in self.modal_types:
            mod_file = [f for f in os.listdir(case_path) if mod in f and 'nii.gz' in f][0]
            mod_path = os.path.join(case_path, mod_file)
            img_nii = nib.load(mod_path)
            img_data = img_nii.get_fdata().astype(np.float32).transpose(2, 0, 1)  # (D,H,W)
            self.spacing = img_nii.header.get_zooms()[:3]  # (z, x, y)
            img_resized = self._resize_3d(img_data, is_label=False)
            modal_data.append(img_resized)
        img = np.stack(modal_data, axis=0)  # (4,D,H,W)


        seg_file = [f for f in os.listdir(case_path) if 'seg' in f and 'nii.gz' in f][0]
        seg_path = os.path.join(case_path, seg_file)
        seg_nii = nib.load(seg_path)
        seg_data = seg_nii.get_fdata().astype(np.int64).transpose(2, 0, 1)  # (D,H,W)
        seg_data = np.where(seg_data == 4, 3, seg_data)
        
        if np.max(seg_data) > 3 or np.min(seg_data) < 0:
            raise ValueError(f"Invalid label values in case {case_name}")
        
        seg = self._resize_3d(seg_data, is_label=True)

  
        if self.normalize:
            img = self._zscore_normalize(img)

       
        if self.augment:
            img, seg = apply_augmentations(img, seg, self.target_size)

       
        if img.shape[1:] != self.target_size:
            new_img = np.zeros((img.shape[0],) + self.target_size)
            for c in range(img.shape[0]):
                new_img[c] = zoom(img[c], [self.target_size[i]/img.shape[i+1] for i in range(3)],
                                order=1, mode='constant', cval=0)
            img = new_img
            
        if seg.shape != self.target_size:
            seg = zoom(seg, [self.target_size[i]/seg.shape[i] for i in range(3)],
                    order=0, mode='constant', cval=0)

    
        img = torch.from_numpy(img.copy()).float()
        seg = torch.from_numpy(seg.copy()).long()

        dataset_id = torch.tensor([0], dtype=torch.long)
        spacing = torch.tensor(self.spacing, dtype=torch.float32)

        return img, seg, dataset_id, case_name, spacing

    def _resize_3d(self, data, is_label):
        zoom_factors = [
            self.target_size[0]/data.shape[0],
            self.target_size[1]/data.shape[1],
            self.target_size[2]/data.shape[2]
        ]
        order = 0 if is_label else 1
        return zoom(data, zoom_factors, order=order, mode='constant', cval=0)

    def _zscore_normalize(self, img):
        for c in range(img.shape[0]):
            mean = img[c].mean()
            std = img[c].std()
            if std > 1e-6:
                img[c] = (img[c] - mean) / std
            else:
                img[c] = img[c] - mean
        return img



def split_tve_dataset(full_dataset, train_ratio=0.6, valid_ratio=0.2, eval_ratio=0.2, seed=42):
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    eval_size = total_size - train_size - valid_size

    train_dataset, rest_dataset = random_split(
        full_dataset, [train_size, total_size - train_size],
        generator=torch.Generator().manual_seed(seed)
    )
    valid_dataset, eval_dataset = random_split(
        rest_dataset, [valid_size, eval_size],
        generator=torch.Generator().manual_seed(seed + 1)
    )

    return train_dataset, valid_dataset, eval_dataset


def load_independent_tve_dataset(train_dir, valid_dir, eval_dir, **dataset_kwargs):
    train_dataset = BraTS20213DDataset(data_dir=train_dir,** dataset_kwargs)
    valid_dataset = BraTS20213DDataset(data_dir=valid_dir, **dataset_kwargs)
    eval_dataset = BraTS20213DDataset(data_dir=eval_dir,** dataset_kwargs)
    return train_dataset, valid_dataset, eval_dataset