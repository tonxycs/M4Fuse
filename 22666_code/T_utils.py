import torch
import torch.nn.functional as F
import numpy as np
from Supp_metric import compute_brats_metrics, compute_confidence_interval

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    metrics_batch = {
        'WT': {'dsc': [], 'sdc': [], 'hd95': []},
        'TC': {'dsc': [], 'sdc': [], 'hd95': []},
        'ET': {'dsc': [], 'sdc': [], 'hd95': []}
    }
    batch_count = 0

    for batch_idx, (img, seg, dataset_id, _, spacing) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)
        dataset_id = dataset_id.to(device, non_blocking=True)

        has_tc = (seg == 1).any().item()
        has_et = (seg == 3).any().item()
        if not has_tc or not has_et:
            print(f"Batch {batch_idx}: TC exists? {has_tc}, ET exists? {has_et}")

        with torch.cuda.amp.autocast():
            pred = model(img, dataset_id)
            loss = criterion(pred, seg)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = img.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        batch_metrics = compute_brats_metrics(pred, seg, spacing)
        for region in ['WT', 'TC', 'ET']:
            for metric in ['dsc', 'sdc', 'hd95']:
                metrics_batch[region][metric].append(batch_metrics[region][metric]['mean'])
        batch_count += 1

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")

    avg_loss = total_loss / total_samples
    epoch_metrics = {}
    for region in ['WT', 'TC', 'ET']:
        epoch_metrics[region] = {}
        for metric in ['dsc', 'sdc', 'hd95']:
            valid_values = [v for v in metrics_batch[region][metric] if not np.isinf(v)]
            mean, ci = compute_confidence_interval(valid_values) if valid_values else (np.inf, 0.0)
            epoch_metrics[region][metric] = {
                'mean': mean,
                'ci': ci,
                'num_inf': len(metrics_batch[region][metric]) - len(valid_values)
            }

    print(f"\n【Training Phase】")
    print(f"Training Loss: {avg_loss:.4f}")
    print(f"WT - DSC: {epoch_metrics['WT']['dsc']['mean']:.4f} ± {epoch_metrics['WT']['dsc']['ci']:.4f}, "
          f"SDC: {epoch_metrics['WT']['sdc']['mean']:.4f} ± {epoch_metrics['WT']['sdc']['ci']:.4f}, "
          f"HD95: {epoch_metrics['WT']['hd95']['mean']:.2f} ± {epoch_metrics['WT']['hd95']['ci']:.2f}mm "
          f"(inf: {epoch_metrics['WT']['hd95']['num_inf']})")
    print(f"TC - DSC: {epoch_metrics['TC']['dsc']['mean']:.4f} ± {epoch_metrics['TC']['dsc']['ci']:.4f}, "
          f"SDC: {epoch_metrics['TC']['sdc']['mean']:.4f} ± {epoch_metrics['TC']['sdc']['ci']:.4f}, "
          f"HD95: {epoch_metrics['TC']['hd95']['mean']:.2f} ± {epoch_metrics['TC']['hd95']['ci']:.2f}mm "
          f"(inf: {epoch_metrics['TC']['hd95']['num_inf']})")
    print(f"ET - DSC: {epoch_metrics['ET']['dsc']['mean']:.4f} ± {epoch_metrics['ET']['dsc']['ci']:.4f}, "
          f"SDC: {epoch_metrics['ET']['sdc']['mean']:.4f} ± {epoch_metrics['ET']['sdc']['ci']:.4f}, "
          f"HD95: {epoch_metrics['ET']['hd95']['mean']:.2f} ± {epoch_metrics['ET']['hd95']['ci']:.2f}mm "
          f"(inf: {epoch_metrics['ET']['hd95']['num_inf']})")
    
    return avg_loss, epoch_metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metrics_batch = {
        'WT': {'dsc': [], 'sdc': [], 'hd95': []},
        'TC': {'dsc': [], 'sdc': [], 'hd95': []},
        'ET': {'dsc': [], 'sdc': [], 'hd95': []}
    }
    batch_count = 0

    for img, seg, dataset_id, _, spacing in dataloader:
        img = img.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)
        dataset_id = dataset_id.to(device, non_blocking=True)
        batch_size = img.size(0)
        
        scales = [0.9, 1.0, 1.1]
        preds = []
        target_size = seg.shape[1:]
        for s in scales:
            scaled_img = F.interpolate(
                img, 
                scale_factor=s, 
                mode='trilinear', 
                align_corners=True
            )
            pred = model(scaled_img, dataset_id)
            pred = F.interpolate(
                pred, 
                size=target_size,
                mode='trilinear', 
                align_corners=True
            )
            preds.append(pred)
        
        pred = torch.mean(torch.stack(preds), dim=0)
        loss = criterion(pred, seg)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        batch_metrics = compute_brats_metrics(pred, seg, spacing)
        for region in ['WT', 'TC', 'ET']:
            for metric in ['dsc', 'sdc', 'hd95']:
                metrics_batch[region][metric].append(batch_metrics[region][metric]['mean'])
        batch_count += 1

    avg_loss = total_loss / total_samples
    epoch_metrics = {}
    for region in ['WT', 'TC', 'ET']:
        epoch_metrics[region] = {}
        for metric in ['dsc', 'sdc', 'hd95']:
            valid_values = [v for v in metrics_batch[region][metric] if not np.isinf(v)]
            mean, ci = compute_confidence_interval(valid_values) if valid_values else (np.inf, 0.0)
            epoch_metrics[region][metric] = {
                'mean': mean,
                'ci': ci,
                'num_inf': len(metrics_batch[region][metric]) - len(valid_values)
            }
    
    return avg_loss, epoch_metrics



def save_tve_metrics(history, eval_metrics, best_epoch, save_path='tve_metrics_summary.txt'):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("BraTS2021 M4Fuse Model TVE Complete Metrics Summary (with 95% CI)\n")
        f.write("="*100 + "\n\n")

        f.write("1. Best Validation Model (Valid Set)\n")
        f.write(f"Best Epoch: {best_epoch + 1}\n")
        f.write(f"Best Training Loss: {history['train_loss'][best_epoch]:.4f}\n")
        f.write(f"Best Validation Loss: {history['valid_loss'][best_epoch]:.4f}\n")
        f.write(f"WT - DSC: {history['valid_wt_dsc_mean'][best_epoch]:.4f} ± {history['valid_wt_dsc_ci'][best_epoch]:.4f}, "
                f"SDC: {history['valid_wt_sdc_mean'][best_epoch]:.4f} ± {history['valid_wt_sdc_ci'][best_epoch]:.4f}, "
                f"HD95: {history['valid_wt_hd95_mean'][best_epoch]:.2f} ± {history['valid_wt_hd95_ci'][best_epoch]:.2f}mm "
                f"(inf: {history['valid_wt_hd95_num_inf'][best_epoch]})\n")
        f.write(f"TC - DSC: {history['valid_tc_dsc_mean'][best_epoch]:.4f} ± {history['valid_tc_dsc_ci'][best_epoch]:.4f}, "
                f"SDC: {history['valid_tc_sdc_mean'][best_epoch]:.4f} ± {history['valid_tc_sdc_ci'][best_epoch]:.4f}, "
                f"HD95: {history['valid_tc_hd95_mean'][best_epoch]:.2f} ± {history['valid_tc_hd95_ci'][best_epoch]:.2f}mm "
                f"(inf: {history['valid_tc_hd95_num_inf'][best_epoch]})\n")
        f.write(f"ET - DSC: {history['valid_et_dsc_mean'][best_epoch]:.4f} ± {history['valid_et_dsc_ci'][best_epoch]:.4f}, "
                f"SDC: {history['valid_et_sdc_mean'][best_epoch]:.4f} ± {history['valid_et_sdc_ci'][best_epoch]:.4f}, "
                f"HD95: {history['valid_et_hd95_mean'][best_epoch]:.2f} ± {history['valid_et_hd95_ci'][best_epoch]:.2f}mm "
                f"(inf: {history['valid_et_hd95_num_inf'][best_epoch]})\n\n")

        f.write("2. Final Evaluation Set Metrics (Fully Independent, Ignored all-0 batches)\n")
        f.write(f"WT - DSC: {eval_metrics['WT']['dsc']['mean']:.4f} ± {eval_metrics['WT']['dsc']['ci']:.4f}, "
                f"SDC: {eval_metrics['WT']['sdc']['mean']:.4f} ± {eval_metrics['WT']['sdc']['ci']:.4f}, "
                f"HD95: {eval_metrics['WT']['hd95']['mean']:.2f} ± {eval_metrics['WT']['hd95']['ci']:.2f}mm "
                f"(inf: {eval_metrics['WT']['hd95']['num_inf']}, valid samples: {eval_metrics['WT']['dsc']['total_valid_samples']})\n")
        f.write(f"TC - DSC: {eval_metrics['TC']['dsc']['mean']:.4f} ± {eval_metrics['TC']['dsc']['ci']:.4f}, "
                f"SDC: {eval_metrics['TC']['sdc']['mean']:.4f} ± {eval_metrics['TC']['sdc']['ci']:.4f}, "
                f"HD95: {eval_metrics['TC']['hd95']['mean']:.2f} ± {eval_metrics['TC']['hd95']['ci']:.2f}mm "
                f"(inf: {eval_metrics['TC']['hd95']['num_inf']}, valid samples: {eval_metrics['TC']['dsc']['total_valid_samples']})\n")
        f.write(f"ET - DSC: {eval_metrics['ET']['dsc']['mean']:.4f} ± {eval_metrics['ET']['dsc']['ci']:.4f}, "
                f"SDC: {eval_metrics['ET']['sdc']['mean']:.4f} ± {eval_metrics['ET']['sdc']['ci']:.4f}, "
                f"HD95: {eval_metrics['ET']['hd95']['mean']:.2f} ± {eval_metrics['ET']['hd95']['ci']:.2f}mm "
                f"(inf: {eval_metrics['ET']['hd95']['num_inf']}, valid samples: {eval_metrics['ET']['dsc']['total_valid_samples']})\n\n")

        f.write("3. Detailed Metrics per Epoch\n")
        f.write("="*220 + "\n")
        f.write(f"{'Epoch':<6} {'Train Loss':<12} {'Valid Loss':<12} "
                f"{'WT DSC(T)':<16} {'WT DSC(V)':<16} "
                f"{'TC DSC(T)':<16} {'TC DSC(V)':<16} "
                f"{'ET DSC(T)':<16} {'ET DSC(V)':<16} "
                f"{'WT HD95(V)':<22} {'TC HD95(V)':<22} {'ET HD95(V)':<22}\n")
        f.write("="*220 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1:<6} {history['train_loss'][i]:<12.4f} {history['valid_loss'][i]:<12.4f} "
                    f"{history['wt_dsc_mean'][i]:.4f}±{history['wt_dsc_ci'][i]:.4f}  "
                    f"{history['valid_wt_dsc_mean'][i]:.4f}±{history['valid_wt_dsc_ci'][i]:.4f}  "
                    f"{history['tc_dsc_mean'][i]:.4f}±{history['tc_dsc_ci'][i]:.4f}  "
                    f"{history['valid_tc_dsc_mean'][i]:.4f}±{history['valid_tc_dsc_ci'][i]:.4f}  "
                    f"{history['et_dsc_mean'][i]:.4f}±{history['et_dsc_ci'][i]:.4f}  "
                    f"{history['valid_et_dsc_mean'][i]:.4f}±{history['valid_et_dsc_ci'][i]:.4f}  "
                    f"{history['valid_wt_hd95_mean'][i]:.2f}±{history['valid_wt_hd95_ci'][i]:.2f} "
                    f"(inf:{history['valid_wt_hd95_num_inf'][i]})  "
                    f"{history['valid_tc_hd95_mean'][i]:.2f}±{history['valid_tc_hd95_ci'][i]:.2f} "
                    f"(inf:{history['valid_tc_hd95_num_inf'][i]})  "
                    f"{history['valid_et_hd95_mean'][i]:.2f}±{history['valid_et_hd95_ci'][i]:.2f} "
                    f"(inf:{history['valid_et_hd95_num_inf'][i]})\n")

    print(f"TVE metrics summary saved to: {save_path}")