import os
import torch
import numpy as np
import logging
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold

from Network import M4Fuse
from Data.data import BraTSDataset, Compose, RandomCrop, RandomFlip, RandomRotate, ElasticTransform
from loss import CombinedLoss
from utils import train_epoch, validate, save_fold_results, save_overall_results
from Supp_metric import compute_metrics, compute_confidence_interval

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def main():
  
    parser = argparse.ArgumentParser(description='M4Fuse for BraTS Segmentation (5-Fold CV)')
    
 
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to BraTS dataset root directory')
    parser.add_argument('--output_root', type=str, 
                        default=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help='Root directory for saving results')
    parser.add_argument('--crop_size', type=int, nargs=3, default=(128, 128, 128),
                        help='3D crop size (depth height width)')
    

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='Total training epochs')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation splits')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Computation device (cuda/cpu)')
    
  
    args = parser.parse_args()
    config = vars(args) 

    
    set_seed()
    
   
    os.makedirs(config['output_root'], exist_ok=True)
    
  
    logging.basicConfig(
        filename=os.path.join(config['output_root'], 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Training configuration: {config}")
    print(f"Using device: {config['device']}")

 
    train_transform = Compose([
        RandomCrop(config['crop_size']),
        RandomFlip(axes=(0, 1, 2), p=0.5),
        RandomRotate(angles=(-15, 15), axes=(1, 2), p=0.3),
        ElasticTransform(alpha=600, sigma=30, p=0.3)
    ])
    val_transform = Compose([
        RandomCrop(config['crop_size'])
    ])

  
    dataset = BraTSDataset(
        data_dir=config['data_dir'],
        transform=None,
        crop_size=config['crop_size']
    )
    all_indices = np.arange(len(dataset))
    patient_types = np.array(dataset.patient_types)  

  
    skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, patient_types)):
        print(f"\n===== Fold {fold+1}/{config['n_splits']} =====")
        logging.info(f"\n===== Fold {fold+1}/{config['n_splits']} =====")
        
   
        fold_dir = os.path.join(config['output_root'], f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        

        train_dataset = dataset 
        train_dataset.transform = train_transform
        val_dataset = dataset
        val_dataset.transform = val_transform
        
        train_subset = [train_dataset[i] for i in train_idx]
        val_subset = [val_dataset[i] for i in val_idx]
        
        train_loader = DataLoader(
            train_subset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        

        model = M4Fuse(
            num_classes=4,
            input_channels=4,
            modalities=2  
        ).to(config['device'])
        # [32, 64, 96, 128, 192, 256] Base
        

        optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-6
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=2e-6  
        )
        
  
        criterion = CombinedLoss()
        best_total = -1
        best_val_metrics = None
        best_epoch = 0
        counter = 0 
        
    
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            logging.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            
     
            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, config['device']
            )
            val_loss, val_metrics = validate(
                model, val_loader, criterion, config['device']
            )
            
        
            print(f"【Train】Loss: {train_loss:.4f} | WT: {train_metrics['WT']:.4f} | "
                  f"TC: {train_metrics['TC']:.4f} | ET: {train_metrics['ET']:.4f}")
            print(f"【Val】Loss: {val_loss:.4f} | WT: {val_metrics['WT']:.4f} | "
                  f"TC: {val_metrics['TC']:.4f} | ET: {val_metrics['ET']:.4f}")
            logging.info(f"Train Loss: {train_loss:.4f}, WT: {train_metrics['WT']:.4f}, "
                         f"TC: {train_metrics['TC']:.4f}, ET: {train_metrics['ET']:.4f}")
            logging.info(f"Val Loss: {val_loss:.4f}, WT: {val_metrics['WT']:.4f}, "
                         f"TC: {val_metrics['TC']:.4f}, ET: {val_metrics['ET']:.4f}")
            
     
            scheduler.step()
            
 
            current_total = val_metrics['Total']
            if current_total > best_total:
                best_total = current_total
                best_val_metrics = val_metrics
                best_epoch = epoch + 1
                counter = 0
                
            
                torch.save({
                    'fold': fold + 1,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metrics': best_val_metrics
                }, os.path.join(fold_dir, 'best_model.pth'))
                print(f"Saved best model (Epoch {best_epoch})")
            else:
                counter += 1
                print(f"Early stop counter: {counter}/{config['patience']}")
                if counter >= config['patience']:
                    print("Early stopping triggered")
                    logging.info("Early stopping triggered")
                    break
        
   
        fold_metrics.append(best_val_metrics)
        save_fold_results(fold+1, best_val_metrics, fold_dir)
        print(f"Fold {fold+1} Best Metrics: WT={best_val_metrics['WT']:.4f}, "
              f"TC={best_val_metrics['TC']:.4f}, ET={best_val_metrics['ET']:.4f}")


    avg_wt, avg_tc, avg_et = save_overall_results(fold_metrics, config['output_root'])
    print(f"\n===== 5-Fold Average Results =====")
    print(f"Avg WT: {avg_wt:.4f}, Avg TC: {avg_tc:.4f}, Avg ET: {avg_et:.4f}")
    logging.info(f"\n5-Fold Average: WT={avg_wt:.4f}, TC={avg_tc:.4f}, ET={avg_et:.4f}")

if __name__ == "__main__":
    main()