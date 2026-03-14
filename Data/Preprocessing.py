import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, random_split, Subset
from scipy.ndimage import zoom, rotate
import glob
import random



def random_rotate_3d(image, label, prob=0.5, angle_range=(-10, 10)):
    if np.random.random() < prob:
        angle_x = np.random.uniform(*angle_range)
        angle_y = np.random.uniform(*angle_range)
        angle_z = np.random.uniform(*angle_range)
        original_shape = image.shape[1:]  # (D, H, W)
        
        rotated_image = np.zeros_like(image)
        for c in range(image.shape[0]):
            temp = rotate(image[c], angle_z, axes=(1, 2), reshape=False, mode='constant', cval=0, order=1)
            temp = rotate(temp, angle_y, axes=(0, 2), reshape=False, mode='constant', cval=0, order=1)
            rotated = rotate(temp, angle_x, axes=(0, 1), reshape=False, mode='constant', cval=0, order=1)
            if rotated.shape != original_shape:
                rotated = zoom(rotated, [original_shape[i]/rotated.shape[i] for i in range(3)], 
                               order=1, mode='constant', cval=0)
            rotated_image[c] = rotated
        
        rotated_label = rotate(label, angle_z, axes=(1, 2), reshape=False, mode='constant', cval=0, order=0)
        rotated_label = rotate(rotated_label, angle_y, axes=(0, 2), reshape=False, mode='constant', cval=0, order=0)
        rotated_label = rotate(rotated_label, angle_x, axes=(0, 1), reshape=False, mode='constant', cval=0, order=0)
        if rotated_label.shape != original_shape:
            rotated_label = zoom(rotated_label, [original_shape[i]/rotated_label.shape[i] for i in range(3)], 
                               order=0, mode='constant', cval=0)
        
        return rotated_image, rotated_label
    return image, label


def random_flip_3d(image, label, prob=0.5, axis=0):
    if np.random.random() < prob:
        return np.flip(image, axis=axis+1), np.flip(label, axis=axis)
    return image, label


def random_zoom_3d(image, label, target_size, prob=0.5, zoom_range=(0.95, 1.05)):
    if np.random.random() < prob:
        zoom_factor = np.random.uniform(*zoom_range)
        zf = (zoom_factor, zoom_factor, zoom_factor)
        
        zoomed_image = np.zeros((image.shape[0],) + target_size)
        for c in range(image.shape[0]):
            scaled = zoom(image[c], zf, order=1, mode='constant', cval=0)
            zoomed_image[c] = zoom(scaled, [target_size[i]/scaled.shape[i] for i in range(3)],
                                 order=1, mode='constant', cval=0)
        
        scaled_label = zoom(label, zf, order=0, mode='constant', cval=0)
        zoomed_label = zoom(scaled_label, [target_size[i]/scaled_label.shape[i] for i in range(3)],
                           order=0, mode='constant', cval=0)
        
        return zoomed_image, zoomed_label
    return image, label


def add_gaussian_noise(image, prob=0.3, mean=0, std=0.01):
    if np.random.random() < prob:
        noise = np.random.normal(mean, std, size=image.shape)
        return image + noise
    return image


def adjust_contrast(image, prob=0.3, gamma_range=(0.8, 1.2)):
    if np.random.random() < prob:
        gamma = np.random.uniform(*gamma_range)
        adjusted = []
        for c in range(image.shape[0]):
            img = image[c]
            min_val = img.min()
            max_val = img.max()
            if max_val - min_val > 1e-6:
                img_norm = (img - min_val) / (max_val - min_val)
                img_norm = np.power(img_norm, gamma)
                adjusted.append(img_norm * (max_val - min_val) + min_val)
            else:
                adjusted.append(img)
        return np.stack(adjusted)
    return image


def apply_augmentations(image, label, target_size):
    image, label = random_rotate_3d(image, label, prob=0.5)
    image, label = random_flip_3d(image, label, prob=0.5, axis=0)
    image, label = random_flip_3d(image, label, prob=0.5, axis=1)
    image, label = random_zoom_3d(image, label, target_size, prob=0.5)
    image = add_gaussian_noise(image, prob=0.3)
    image = adjust_contrast(image, prob=0.3)
    return image, label


class RandomCrop:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, data, seg):
        c, d, h, w = data.shape
        crop_d, crop_h, crop_w = self.size
        start_d = random.randint(0, d - crop_d)
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        return (
            data[:, start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w],
            seg[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
        )


class ElasticTransform:
    def __init__(self, alpha=600, sigma=30, p=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        
    def __call__(self, data, seg):
        if random.random() < self.p:
            data_np = data.permute(1, 2, 3, 0).numpy()
            seg_np = seg.numpy()[..., None]
            
            import albumentations as A
            transform = A.ElasticTransform(
                alpha=self.alpha, sigma=self.sigma, alpha_affine=0,
                interpolation=1, border_mode=0, value=0, mask_value=0, p=1.0
            )
            augmented = transform(image=data_np, mask=seg_np)
            data_np = augmented['image']
            seg_np = augmented['mask'][..., 0]
            
            data = torch.from_numpy(data_np).permute(3, 0, 1, 2)
            seg = torch.from_numpy(seg_np)
        return data, seg


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, data, seg):
        for t in self.transforms:
            data, seg = t(data, seg)
        return data, seg

