import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from Supp_metric import compute_brats_metrics, compute_confidence_interval

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    metrics = {'WT': [], 'TC': [], 'ET': [], 'Total': []}
    scaler = GradScaler()
    
    for data, target, dataset_id in tqdm(dataloader, desc="Training", mininterval=30):
        data, target, dataset_id = data.to(device), target.to(device), dataset_id.to(device)
        
        optimizer.zero_grad()
        with autocast():
            output = model(data, dataset_id)  
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        batch_metrics = compute_brats_metrics(output, target)
        for key in metrics:
            metrics[key].append(batch_metrics[key])
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    return avg_loss, avg_metrics


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = {'WT': [], 'TC': [], 'ET': [], 'Total': []}
    
    with torch.no_grad():
        for data, target, dataset_id in tqdm(dataloader, desc="Validation", mininterval=30):
            data, target, dataset_id = data.to(device), target.to(device), dataset_id.to(device)
            
            with autocast():
                output = model(data, dataset_id)
                loss = criterion(output, target)
            
            total_loss += loss.item()
            batch_metrics = compute_brats_metrics(output, target)
            for key in metrics:
                metrics[key].append(batch_metrics[key])
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    return avg_loss, avg_metrics


def save_fold_results(fold, best_metrics, save_dir):

    import os
    with open(os.path.join(save_dir, f'fold_{fold}_results.txt'), 'w') as f:
        f.write(f"Best Metrics for Fold {fold}:\n")
        f.write(f"WT Dice: {best_metrics['WT']:.4f}\n")
        f.write(f"TC Dice: {best_metrics['TC']:.4f}\n")
        f.write(f"ET Dice: {best_metrics['ET']:.4f}\n")
        f.write(f"Total Dice: {best_metrics['Total']:.4f}\n")


def save_overall_results(fold_metrics, save_dir):

    import os
    avg_wt = np.mean([m['WT'] for m in fold_metrics])
    avg_tc = np.mean([m['TC'] for m in fold_metrics])
    avg_et = np.mean([m['ET'] for m in fold_metrics])
    avg_total = np.mean([m['Total'] for m in fold_metrics])
    

    wt_mean, wt_ci = compute_confidence_interval([m['WT'] for m in fold_metrics])
    tc_mean, tc_ci = compute_confidence_interval([m['TC'] for m in fold_metrics])
    et_mean, et_ci = compute_confidence_interval([m['ET'] for m in fold_metrics])
    
    with open(os.path.join(save_dir, 'overall_results.txt'), 'w') as f:
        f.write("5-Fold Cross Validation Results:\n")
        for i, m in enumerate(fold_metrics):
            f.write(f"Fold {i+1}: WT={m['WT']:.4f}, TC={m['TC']:.4f}, ET={m['ET']:.4f}\n")
        f.write("\nAverage Metrics:\n")
        f.write(f"WT Dice: {avg_wt:.4f} ± {wt_ci:.4f}\n")
        f.write(f"TC Dice: {avg_tc:.4f} ± {tc_ci:.4f}\n")
        f.write(f"ET Dice: {avg_et:.4f} ± {et_ci:.4f}\n")
        f.write(f"Total Dice: {avg_total:.4f}\n")
    
    return avg_wt, avg_tc, avg_et