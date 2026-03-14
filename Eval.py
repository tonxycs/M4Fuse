
import os        
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from torch.utils.data import DataLoader, Subset, random_split
from scipy.ndimage import zoom, label, binary_dilation, generate_binary_structure, binary_fill_holes
import scipy.stats as stats
import math

from Network import M4Fuse
from Data.data import BraTS20213DDataset
from Supp_metric import compute_brats_metrics, get_final_mask, compute_confidence_interval  
from monai.metrics import compute_hausdorff_distance




def split_and_cache_evaluation_set(full_data_dir, split_ratio=0.2, seed=42, cache_dir='dataset_splits'):

    os.makedirs(cache_dir, exist_ok=True)
    split_cache_path = os.path.join(cache_dir, 'eval_split_indices.json')
    

    full_dataset = BraTS20213DDataset(
        data_dir=full_data_dir,
        target_size=(64, 128, 128),
        normalize=True,
        augment=False  
    )
    total_size = len(full_dataset)
    eval_size = int(total_size * split_ratio)
    train_val_size = total_size - eval_size
    
   
    if os.path.exists(split_cache_path):
        print(f"Loading cached evaluation split from: {split_cache_path}")
        with open(split_cache_path, 'r') as f:
            split_data = json.load(f)
        eval_indices = split_data['eval_indices']
        eval_dataset = Subset(full_dataset, eval_indices)
        
    else:
        print(f"Generating new evaluation split (ratio: {split_ratio})...")

        _, eval_dataset = random_split(
            full_dataset,
            [train_val_size, eval_size],
            generator=torch.Generator().manual_seed(seed)  
        )

        eval_indices = eval_dataset.indices
        with open(split_cache_path, 'w') as f:
            json.dump({
                'total_cases': total_size,
                'eval_cases': len(eval_indices),
                'split_ratio': split_ratio,
                'seed': seed,
                'eval_indices': eval_indices,
                'split_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        print(f"Evaluation split saved to: {split_cache_path}")
    
    print(f"Evaluation set contains {len(eval_dataset)} cases (from total {total_size} cases)")
    return eval_dataset



def advanced_post_processing(pred_mask, spacing, region_type=None):


    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    

    if pred_mask.ndim == 2:
        pred_mask = pred_mask[np.newaxis, :, :]  
    elif pred_mask.ndim != 3:
        raise ValueError(f"Expected 2D or 3D mask, got {pred_mask.ndim}D instead")

    

    processed = binary_fill_holes(pred_mask)
    

    if len(spacing) != processed.ndim:
        raise ValueError(f"Spacing dimension ({len(spacing)}) must match mask dimension ({processed.ndim})")
    voxel_volume = np.prod(spacing)
    

    structure = generate_binary_structure(3, 2) 
    labeled_mask, num_labels = label(processed, structure=structure)
    

    if region_type == 'ET': 

        component_volumes = []
        for i in range(1, num_labels + 1):
            component = (labeled_mask == i)
            component_voxels = np.sum(component)
            component_volume = component_voxels * voxel_volume
            component_volumes.append((i, component_volume))
        

        if component_volumes:
            largest_component = max(component_volumes, key=lambda x: x[1])[0]
            processed = (labeled_mask == largest_component)
    
    elif region_type == 'TC': 
        min_volume = 20 
        min_voxels = max(1, math.ceil(min_volume / voxel_volume))
        

        for i in range(1, num_labels + 1):
            component = (labeled_mask == i)
            if np.sum(component) < min_voxels:
                processed[labeled_mask == i] = 0
    
    elif region_type == 'WT':  
        min_volume = 50  
        min_voxels = max(1, math.ceil(min_volume / voxel_volume))
        

        for i in range(1, num_labels + 1):
            component = (labeled_mask == i)
            if np.sum(component) < min_voxels:
                processed[labeled_mask == i] = 0
    

    if region_type in ['WT', 'TC']:
        processed = binary_dilation(processed, structure=structure, iterations=1)
    
    return processed.astype(np.float32)

# optim  eval
@torch.no_grad()
def evaluate(model, dataloader, device, et_config):
    model.eval()
    metrics_samples = {
        'WT': {'dsc': [], 'sdc': [], 'hd95': []},
        'TC': {'dsc': [], 'sdc': [], 'hd95': []},
        'ET': {'dsc': [], 'sdc': [], 'hd95': []}
    }
    case_records = []

    print("="*60)
    print("Starting Enhanced Evaluation (Advanced Inference + Post-Processing)")
    print("="*60)

    for batch_idx, (img, seg, dataset_id, case_name, spacing) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)
        dataset_id = dataset_id.to(device, non_blocking=True)
        batch_size = img.size(0)
        spacing_np = spacing.numpy()  


        scales = [0.8, 0.9, 1.0, 1.1, 1.2]  
        preds_softmax = []
        target_size = seg.shape[1:]  
        

        for s in scales:
       
            scaled_img = F.interpolate(img, scale_factor=s, mode='trilinear', align_corners=True)
            
         
            pred = model(scaled_img, dataset_id)
            
       
            pred_flip = model(torch.flip(scaled_img, dims=[4]), dataset_id) 
            pred_flip = torch.flip(pred_flip, dims=[4])  
            
  
            pred_flip2 = model(torch.flip(scaled_img, dims=[3]), dataset_id) 
            pred_flip2 = torch.flip(pred_flip2, dims=[3]) 
            
    
            pred_combined = (pred + pred_flip + pred_flip2) / 3.0
            
     
            pred_combined = F.interpolate(
                pred_combined, 
                size=target_size, 
                mode='trilinear', 
                align_corners=True
            )
            preds_softmax.append(F.softmax(pred_combined, dim=1)) 


        scale_weights = [0.8, 1.0, 1.4, 1.0, 0.8] 
        scale_weights = np.array(scale_weights) / np.sum(scale_weights)  
        
        pred_softmax = torch.zeros_like(preds_softmax[0])
        for i, weight in enumerate(scale_weights):
            pred_softmax += weight * preds_softmax[i]


        final_pred = get_final_mask(pred_softmax, spacing, et_config)
        
 
        final_pred_np = final_pred.cpu().numpy()
        seg_np = seg.cpu().numpy()
        
      
        for b in range(batch_size):
       
            wt_mask = final_pred_np[b, 1] 
            tc_mask = final_pred_np[b, 2] 
            et_mask = final_pred_np[b, 3]  
            

            wt_processed = advanced_post_processing(wt_mask, spacing_np[b], 'WT')
            tc_processed = advanced_post_processing(tc_mask, spacing_np[b], 'TC')
            et_processed = advanced_post_processing(et_mask, spacing_np[b], 'ET')
            
     
            final_pred_np[b, 1] = wt_processed
            final_pred_np[b, 2] = tc_processed
            final_pred_np[b, 3] = et_processed
        

        final_pred = torch.from_numpy(final_pred_np).to(device)


        batch_metrics = compute_brats_metrics(
            final_pred, seg, spacing, 
        )

     
        for b in range(batch_size):
            case = case_name[b]
     
            wt_dsc = batch_metrics['WT']['dsc']['mean']
            tc_dsc = batch_metrics['TC']['dsc']['mean']
            et_dsc = batch_metrics['ET']['dsc']['mean']
            wt_sdc = batch_metrics['WT']['sdc']['mean']
            tc_sdc = batch_metrics['TC']['sdc']['mean']
            et_sdc = batch_metrics['ET']['sdc']['mean']
            wt_hd95 = batch_metrics['WT']['hd95']['mean']
            tc_hd95 = batch_metrics['TC']['hd95']['mean']
            et_hd95 = batch_metrics['ET']['hd95']['mean']


            if wt_dsc < 1e-6 and tc_dsc < 1e-6 and et_dsc < 1e-6:
                print(f"Batch [{batch_idx+1}] | Case {case} ")
                continue


            metrics_samples['WT']['dsc'].append(wt_dsc)
            metrics_samples['WT']['sdc'].append(wt_sdc)
            metrics_samples['WT']['hd95'].append(wt_hd95)
            metrics_samples['TC']['dsc'].append(tc_dsc)
            metrics_samples['TC']['sdc'].append(tc_sdc)
            metrics_samples['TC']['hd95'].append(tc_hd95)
            metrics_samples['ET']['dsc'].append(et_dsc)
            metrics_samples['ET']['sdc'].append(et_sdc)
            metrics_samples['ET']['hd95'].append(et_hd95)


            case_records.append({
                'case_name': case,
                'WT_DSC': round(wt_dsc, 4),
                'WT_SDC': round(wt_sdc, 4),
                'WT_HD95(mm)': round(wt_hd95, 2) if not np.isinf(wt_hd95) else 'inf',
                'TC_DSC': round(tc_dsc, 4),
                'TC_SDC': round(tc_sdc, 4),
                'TC_HD95(mm)': round(tc_hd95, 2) if not np.isinf(tc_hd95) else 'inf',
                'ET_DSC': round(et_dsc, 4),
                'ET_SDC': round(et_sdc, 4),
                'ET_HD95(mm)': round(et_hd95, 2) if not np.isinf(et_hd95) else 'inf'
            })

    
            print(f"Batch [{batch_idx+1}/{len(dataloader)}] | Case {case}")
            print(f"  WT-DSC: {wt_dsc:.4f} | TC-DSC: {tc_dsc:.4f} | ET-DSC: {et_dsc:.4f}")


    eval_metrics = {}
    for region in ['WT', 'TC', 'ET']:
        eval_metrics[region] = {}
        for metric in ['dsc', 'sdc', 'hd95']:
      
            valid_vals = [v for v in metrics_samples[region][metric] if not np.isinf(v)]
 
            mean, ci = compute_confidence_interval(valid_vals) if valid_vals else (np.inf, 0.0)
            eval_metrics[region][metric] = {
                'mean': mean,
                'ci': ci,
                'num_inf': len(metrics_samples[region][metric]) - len(valid_vals),
                'total_valid_samples': len(valid_vals)
            }

    return eval_metrics, case_records, metrics_samples



def save_eval_results(eval_metrics, case_records, metrics_samples, et_config, checkpoint, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    current_time = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')


    summary_path = os.path.join(save_dir, f'eval_summary_{current_time}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("BraTS2021 Evaluation Summary (Enhanced M4Fuse Model)\n")
        f.write("="*100 + "\n")
        f.write(f"Best Model Epoch: {checkpoint['epoch']}\n")
        f.write(f"Best Valid Loss: {checkpoint['best_valid_loss']:.4f}\n")

        f.write(f"ET Optimization Config: {et_config}\n")
        f.write(f"Evaluation Time: {current_time}\n")
        f.write("Enhancements Applied: Multi-scale inference, Test-time augmentation, Region-specific post-processing\n")
        f.write("="*100 + "\n\n")


        for region in ['WT', 'TC', 'ET']:
            f.write(f"\n{region}:\n")
            f.write(f"  DSC: {eval_metrics[region]['dsc']['mean']:.4f} ± {eval_metrics[region]['dsc']['ci']:.4f} "
                    f"(valid samples: {eval_metrics[region]['dsc']['total_valid_samples']})\n")
            f.write(f"  SDC: {eval_metrics[region]['sdc']['mean']:.4f} ± {eval_metrics[region]['sdc']['ci']:.4f}\n")
            f.write(f"  HD95: {eval_metrics[region]['hd95']['mean']:.2f} ± {eval_metrics[region]['hd95']['ci']:.2f}mm "
                    f"(inf: {eval_metrics[region]['hd95']['num_inf']})\n")


    case_df = pd.DataFrame(case_records)
    case_path = os.path.join(save_dir, f'eval_case_details_{current_time}.csv')
    case_df.to_csv(case_path, index=False, encoding='utf-8-sig')


    raw_path = os.path.join(save_dir, f'eval_raw_samples_{current_time}.npz')
    np.savez_compressed(
        raw_path,
        WT_DSC=np.array(metrics_samples['WT']['dsc']),
        WT_SDC=np.array(metrics_samples['WT']['sdc']),
        WT_HD95=np.array(metrics_samples['WT']['hd95']),
        TC_DSC=np.array(metrics_samples['TC']['dsc']),
        TC_SDC=np.array(metrics_samples['TC']['sdc']),
        TC_HD95=np.array(metrics_samples['TC']['hd95']),
        ET_DSC=np.array(metrics_samples['ET']['dsc']),
        ET_SDC=np.array(metrics_samples['ET']['sdc']),
        ET_HD95=np.array(metrics_samples['ET']['hd95'])
    )


    print("\n" + "="*60)
    print("Evaluation Results Saved To:")
    print(f"1. Summary (Txt): {summary_path}")
    print(f"2. Case Details (CSV): {case_path}")
    print(f"3. Raw Sample Metrics (NPZ): {raw_path}")
    print("="*60)



def main():

    config = {
        'full_data_dir': './Data/BraTS2021', 
        'best_model_path': '',   # from the best valid epochs from T21.py output.
        'result_save_dir': '',
        'split_ratio': 0.2,  
        'split_seed': 42,  
        'split_cache_dir': './data/dataset_splits',  
        'target_size': (64, 128, 128), 
        'batch_size': 2,  
        'num_workers': 8, 
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'mixed_precision': False  
    }


    print("="*60)
    print("Enhanced Evaluation Configuration")
    print("="*60)
    for k, v in config.items():
        print(f"{k}: {v}")
    print("="*60 + "\n")

 
    eval_dataset = split_and_cache_evaluation_set(
        full_data_dir=config['full_data_dir'],
        split_ratio=config['split_ratio'],
        seed=config['split_seed'],
        cache_dir=config['split_cache_dir']
    )


    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True 
    )


    print(f"\nLoading best model from: {config['best_model_path']}")
    checkpoint = torch.load(config['best_model_path'], map_location=config['device'])

    model = M4Fuse(
        num_classes=4,
        input_channels=4,
        c_list=[24, 48, 72, 96, 144, 192],   # e.g. Small  Base: -> 256
        modalities=1
    ).to(config['device'])
    
    
    model.load_state_dict(checkpoint['model_state_dict'])
    

    if config['mixed_precision'] and config['device'] == 'cuda':
        model = torch.compile(model) 
        scaler = torch.cuda.amp.GradScaler()  
    else:
        scaler = None
    
 
    et_config = checkpoint.get('et_config', {
        'base_threshold': 0.3,
        'min_fp_volume': {'high_conf': 50, 'low_conf': 30}, 
        'min_fill_volume': 20,
        'max_dilation_iter': 2,
        'threshold_mode': 'fixed',
        'dilation_iter': {'regular': 3, 'slender': 4}  
    })
    print(f"Model loaded successfully (Epoch {checkpoint['epoch']})")
    print(f"ET Optimization Config: {et_config}\n")

 
    eval_metrics, case_records, metrics_samples = evaluate(
        model=model,
        dataloader=eval_loader,
        device=config['device'],
        et_config=et_config
    )


    save_eval_results(
        eval_metrics=eval_metrics,
        case_records=case_records,
        metrics_samples=metrics_samples,
        et_config=et_config,  
        checkpoint=checkpoint,  
        save_dir=config['result_save_dir']
    )


    print("\n" + "="*80)
    print("Final Enhanced Evaluation Metrics (95% Confidence Interval)")
    print("="*80)
    for region in ['WT', 'TC', 'ET']:
        print(f"\n{region}:")
        print(f"  DSC: {eval_metrics[region]['dsc']['mean']:.4f} ± {eval_metrics[region]['dsc']['ci']:.4f} "
              f"(valid samples: {eval_metrics[region]['dsc']['total_valid_samples']})")
        print(f"  SDC: {eval_metrics[region]['sdc']['mean']:.4f} ± {eval_metrics[region]['sdc']['ci']:.4f}")
        print(f"  HD95: {eval_metrics[region]['hd95']['mean']:.2f} ± {eval_metrics[region]['hd95']['ci']:.2f}mm "
              f"(inf: {eval_metrics[region]['hd95']['num_inf']})")
    print("="*80)


if __name__ == '__main__':
    main()
