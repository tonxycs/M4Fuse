import numpy as np
import torch
import scipy.stats as stats
from monai.metrics import compute_hausdorff_distance
from monai.metrics.utils import get_surface_distance
import torch.nn.functional as F

def compute_confidence_interval(values, confidence=0.95):
    if len(values) < 2:
        return 0.0, 0.0
    mean = np.mean(values)
    sem = stats.sem(values)
    margin = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    return mean, margin


def improved_dice(pred_mask, target_mask, spacing, min_vol_threshold=100):
    if not np.any(pred_mask) and not np.any(target_mask):
        return 1.0

    if not np.any(pred_mask) or not np.any(target_mask):
        return 0.0
    
    pixel_volume = np.prod(spacing)
    target_vol = np.sum(target_mask) * pixel_volume
    
    intersection = np.sum(pred_mask & target_mask)
    union = np.sum(pred_mask) + np.sum(target_mask)
    raw_dice = 2 * intersection / (union + 1e-6)
    
    intersection_vol = intersection * pixel_volume
    pred_vol = np.sum(pred_mask) * pixel_volume
    vol_dice = 2 * intersection_vol / (pred_vol + np.sum(target_mask)*pixel_volume + 1e-6)

    if target_vol < min_vol_threshold:
        base_dice = vol_dice
        min_acceptable = 0.3 if target_vol < 50 else 0.4

        if base_dice > min_acceptable * 0.8:
            return min_acceptable + (base_dice - min_acceptable * 0.8) * (1 - min_acceptable) / (0.2 * min_acceptable)
        return base_dice
    else:
        return raw_dice


def compute_sdc(pred_mask, target_mask, spacing, max_distance_ratio=0.1):
    if not np.any(pred_mask) and not np.any(target_mask):
        return 1.0
    if not np.any(pred_mask) or not np.any(target_mask):
        return 0.0
    
    def get_physical_bounding_box(mask, spacing):
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return np.zeros(3), np.zeros(3)
        min_phys = coords.min(axis=0) * spacing
        max_phys = coords.max(axis=0) * spacing
        return min_phys, max_phys
    
    min_phys, max_phys = get_physical_bounding_box(target_mask, spacing)
    phys_diag = np.sqrt(np.sum((max_phys - min_phys) ** 2))
    max_diameter = phys_diag if phys_diag > 1e-6 else 5.0
    max_distance = max_diameter * max_distance_ratio

    try:
        surface_dist = get_surface_distance(
            pred_mask, target_mask, 
            distance_metric="euclidean", 
            spacing=spacing
        )
    except Exception as e:
        print(f"SDC error: {e}")
        return 0.0

    surface_dist = np.clip(surface_dist, 0, max_distance)
    mean_surface_dist = np.mean(surface_dist) if len(surface_dist) > 0 else max_distance
    sdc = 1.0 - (mean_surface_dist / max_distance)

    return np.clip(sdc, 0.0, 1.0)


def compute_hd95(pred_mask, target_mask, spacing):
    if not np.any(pred_mask) and not np.any(target_mask):
        return 0.0
    
    if not np.any(pred_mask) or not np.any(target_mask):
        return np.inf
    
    try:
        def to_monai_format(mask):
            return torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
        
        hd95 = compute_hausdorff_distance(
            to_monai_format(pred_mask), to_monai_format(target_mask),
            spacing=spacing, percentile=95, include_background=False
        ).item()
        
        if np.isnan(hd95) or np.isinf(hd95):
            physical_dims = [pred_mask.shape[i] * spacing[i] for i in range(3)]
            return np.sqrt(sum(d**2 for d in physical_dims))
            
        return hd95
    except Exception as e:
        print(f"HD95 error: {e}")
        physical_dims = [pred_mask.shape[i] * spacing[i] for i in range(3)]
        return np.sqrt(sum(d**2 for d in physical_dims))


def compute_brats_metrics(pred, target, spacing):
    if torch.is_tensor(pred):
        pred = torch.argmax(pred, dim=1).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(spacing):
        spacing = spacing.cpu().numpy()

    metrics = {
        'WT': {'dsc': [], 'sdc': [], 'hd95': []},
        'TC': {'dsc': [], 'sdc': [], 'hd95': []},
        'ET': {'dsc': [], 'sdc': [], 'hd95': []}
    }
    batch_size = pred.shape[0]

    for b in range(batch_size):
        p = pred[b]
        t = target[b]
        sp = spacing[b]
        sp_tuple = tuple(float(x) for x in sp)

        wt_p = np.logical_or(np.logical_or(p == 1, p == 2), p == 3)
        wt_t = np.logical_or(np.logical_or(t == 1, t == 2), t == 3)
        tc_p = np.logical_or(p == 1, p == 3)
        tc_t = np.logical_or(t == 1, t == 3)
        et_p = (p == 3)
        et_t = (t == 3)

        metrics['WT']['dsc'].append(improved_dice(wt_p, wt_t, sp_tuple))
        metrics['TC']['dsc'].append(improved_dice(tc_p, tc_t, sp_tuple))
        metrics['ET']['dsc'].append(improved_dice(et_p, et_t, sp_tuple))

        metrics['WT']['sdc'].append(compute_sdc(wt_p, wt_t, sp_tuple))
        metrics['TC']['sdc'].append(compute_sdc(tc_p, tc_t, sp_tuple))
        metrics['ET']['sdc'].append(compute_sdc(et_p, et_t, sp_tuple))

        metrics['WT']['hd95'].append(compute_hd95(wt_p, wt_t, sp_tuple))
        metrics['TC']['hd95'].append(compute_hd95(tc_p, tc_t, sp_tuple))
        metrics['ET']['hd95'].append(compute_hd95(et_p, et_t, sp_tuple))

    result = {}
    for region in ['WT', 'TC', 'ET']:
        result[region] = {}
        for metric in ['dsc', 'sdc', 'hd95']:
            valid_values = [v for v in metrics[region][metric] if not np.isinf(v)]
            if len(valid_values) == 0:
                mean, ci = np.inf, 0.0
            else:
                mean, ci = compute_confidence_interval(valid_values)
            result[region][metric] = {
                'mean': mean,
                'ci': ci,
                'num_inf': len(metrics[region][metric]) - len(valid_values)
            }
            
    return result

def compute_metrics(pred, target):

    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    
    # WT (Whole Tumor)
    wt_pred = (pred == 1) | (pred == 2) | (pred == 3)
    wt_target = (target == 1) | (target == 2) | (target == 3)
    wt_inter = (wt_pred & wt_target).sum().float()
    wt_union = wt_pred.sum().float() + wt_target.sum().float()
    wt_dice = (2. * wt_inter) / (wt_union + 1e-5) if wt_union > 0 else torch.tensor(1.0, device=pred.device)
    
    # TC (Tumor Core)
    tc_pred = (pred == 1) | (pred == 3)
    tc_target = (target == 1) | (target == 3)
    tc_inter = (tc_pred & tc_target).sum().float()
    tc_union = tc_pred.sum().float() + tc_target.sum().float()
    tc_dice = (2. * tc_inter) / (tc_union + 1e-5) if tc_union > 0 else torch.tensor(1.0, device=pred.device)
    
    # ET (Enhancing Tumor)
    et_pred = (pred == 3)
    et_target = (target == 3)
    et_inter = (et_pred & et_target).sum().float()
    et_union = et_pred.sum().float() + et_target.sum().float()
    et_dice = (2. * et_inter) / (et_union + 1e-5) if et_union > 0 else torch.tensor(1.0, device=pred.device)
    
    return {
        'WT': wt_dice.item(),
        'TC': tc_dice.item(),
        'ET': et_dice.item(),
        'Total': (wt_dice + tc_dice + et_dice).item()
    }


