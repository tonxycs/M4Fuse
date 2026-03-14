import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from Network import M4Fuse
from Data.data import BraTS20213DDataset, split_tve_dataset, load_independent_tve_dataset
from loss import BraTSLoss
from T_utils import train_one_epoch, validate, save_tve_metrics



def main():
    parser = argparse.ArgumentParser(description='M4Fuse Training for BraTS2021 (Train-Valid, else Eval.py)')
 

    parser.add_argument('--data_mode', type=str, default='split_from_full', choices=['split_from_full', 'independent'],
                        help='Dataset mode: split from full or independent folders')
    parser.add_argument('--full_data_dir', type=str, default='./Data/BraTS2021',
                        help='Path to full dataset (for split mode)')
    parser.add_argument('--train_dir', type=str, default='./Data/BraTS2021_Train',
                        help='Path to train dataset (for independent mode)')
    parser.add_argument('--valid_dir', type=str, default='./Data/BraTS2021_Valid',
                        help='Path to validation dataset (for independent mode)')
    parser.add_argument('--result_dir', type=str, default='./BarTS2021-M4FuseTiny',
                        help='Directory to save results')
    

    parser.add_argument('--target_size', type=int, nargs='+', default=[64, 128, 128],
                        help='Target size for 3D images (D, H, W)')
    parser.add_argument('--tve_ratio', type=list, default=[0.6, 0.2, 0.2],
                        help='Train/valid/eval split ratio (Only for split mode; only the first two parts are actually used.)')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Gradient accumulation batches')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device (cuda/cpu)')
    
    args = parser.parse_args()
    config = vars(args)
    config['target_size'] = tuple(config['target_size']) 

 
    print("="*80)
    print("BraTS2021 Train-Valid Configuration")
    print("="*80)
    for k, v in config.items():
        print(f"{k}: {v}")
    print("="*80 + "\n")


    os.makedirs(config['result_dir'], exist_ok=True)

   
    dataset_kwargs = {
        'target_size': config['target_size'],
        'normalize': True
    }
    if config['data_mode'] == 'split_from_full':
        full_dataset = BraTS20213DDataset(data_dir=config['full_data_dir'],** dataset_kwargs)

        train_dataset, valid_dataset, _ = split_tve_dataset(
            full_dataset,
            train_ratio=config['tve_ratio'][0],
            valid_ratio=config['tve_ratio'][1],
            eval_ratio=config['tve_ratio'][2]  
        )
        train_dataset.dataset.augment = True
        valid_dataset.dataset.augment = False
    else:
        train_dataset = BraTS20213DDataset(data_dir=config['train_dir'], augment=True, **dataset_kwargs)
        valid_dataset = BraTS20213DDataset(data_dir=config['valid_dir'], augment=False,** dataset_kwargs)


    print(f"Dataset split completed:")
    print(f"Training set: {len(train_dataset)} cases | Validation set: {len(valid_dataset)} cases")

 
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

   
    print(f"\nInitializing model (Device: {config['device']})...")
    model = M4Fuse(
        num_classes=4,
        input_channels=4,
        c_list=[32, 64, 96, 128, 192, 256],
        modalities=1
    ).to(config['device'])

    class_weights = torch.tensor([1.0, 3.0, 2.0, 5.0]).to(config['device'])
    criterion = BraTSLoss(
        weight=class_weights,
        dice_weight=0.7,
        class_dice_weights=[1.0, 2.0, 3.0, 5.0],
        device=config['device']
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=1e-6
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=2e-6
    )

    scaler = torch.cuda.amp.GradScaler()

 
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {total_params / 1e6:.2f} M")

   
    print("\n" + "="*50)
    print("Starting Train-Valid Pipeline")
    print("="*50)

  
    history = {
        'train_loss': [], 'valid_loss': [],
        'wt_dsc_mean': [], 'wt_dsc_ci': [],
        'tc_dsc_mean': [], 'tc_dsc_ci': [],
        'et_dsc_mean': [], 'et_dsc_ci': [],
        'valid_wt_dsc_mean': [], 'valid_wt_dsc_ci': [],
        'valid_wt_sdc_mean': [], 'valid_wt_sdc_ci': [],
        'valid_wt_hd95_mean': [], 'valid_wt_hd95_ci': [], 'valid_wt_hd95_num_inf': [],
        'valid_tc_dsc_mean': [], 'valid_tc_dsc_ci': [],
        'valid_tc_sdc_mean': [], 'valid_tc_sdc_ci': [],
        'valid_tc_hd95_mean': [], 'valid_tc_hd95_ci': [], 'valid_tc_hd95_num_inf': [],
        'valid_et_dsc_mean': [], 'valid_et_dsc_ci': [],
        'valid_et_sdc_mean': [], 'valid_et_sdc_ci': [],
        'valid_et_hd95_mean': [], 'valid_et_hd95_ci': [], 'valid_et_hd95_num_inf': []
    }
    best_valid_loss = float('inf')
    best_epoch = 0

    for epoch in range(config['epochs']):
        print(f"\n" + "="*60)
        print(f"Epoch [{epoch+1}/{config['epochs']}] | Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("="*60)


        print(f"\n【Training Phase】")
        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=config['device']
        )


        print(f"\n【Validation Phase】")
        valid_loss, valid_metrics = validate(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=config['device']
        )
        print(f"Validation Loss: {valid_loss:.4f}")
        print(f"Validation WT - DSC: {valid_metrics['WT']['dsc']['mean']:.4f} ± {valid_metrics['WT']['dsc']['ci']:.4f}, "
              f"SDC: {valid_metrics['WT']['sdc']['mean']:.4f} ± {valid_metrics['WT']['sdc']['ci']:.4f}, "
              f"HD95: {valid_metrics['WT']['hd95']['mean']:.2f} ± {valid_metrics['WT']['hd95']['ci']:.2f}mm "
              f"(inf: {valid_metrics['WT']['hd95']['num_inf']})")
        print(f"Validation TC - DSC: {valid_metrics['TC']['dsc']['mean']:.4f} ± {valid_metrics['TC']['dsc']['ci']:.4f}, "
              f"SDC: {valid_metrics['TC']['sdc']['mean']:.4f} ± {valid_metrics['TC']['sdc']['ci']:.4f}, "
              f"HD95: {valid_metrics['TC']['hd95']['mean']:.2f} ± {valid_metrics['TC']['hd95']['ci']:.2f}mm "
              f"(inf: {valid_metrics['TC']['hd95']['num_inf']})")
        print(f"Validation ET - DSC: {valid_metrics['ET']['dsc']['mean']:.4f} ± {valid_metrics['ET']['dsc']['ci']:.4f}, "
              f"SDC: {valid_metrics['ET']['sdc']['mean']:.4f} ± {valid_metrics['ET']['sdc']['ci']:.4f}, "
              f"HD95: {valid_metrics['ET']['hd95']['mean']:.2f} ± {valid_metrics['ET']['hd95']['ci']:.2f}mm "
              f"(inf: {valid_metrics['ET']['hd95']['num_inf']})")

    
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        
        history['wt_dsc_mean'].append(train_metrics['WT']['dsc']['mean'])
        history['wt_dsc_ci'].append(train_metrics['WT']['dsc']['ci'])
        history['tc_dsc_mean'].append(train_metrics['TC']['dsc']['mean'])
        history['tc_dsc_ci'].append(train_metrics['TC']['dsc']['ci'])
        history['et_dsc_mean'].append(train_metrics['ET']['dsc']['mean'])
        history['et_dsc_ci'].append(train_metrics['ET']['dsc']['ci'])
        
      
        for region in ['WT', 'TC', 'ET']:
            for metric in ['dsc', 'sdc', 'hd95']:
                history_key_mean = f'valid_{region.lower()}_{metric}_mean'
                history_key_ci = f'valid_{region.lower()}_{metric}_ci'
                history[history_key_mean].append(valid_metrics[region][metric]['mean'])
                history[history_key_ci].append(valid_metrics[region][metric]['ci'])

                if metric == 'hd95':
                    history_key_inf = f'valid_{region.lower()}_{metric}_num_inf'
                    history[history_key_inf].append(valid_metrics[region][metric]['num_inf'])

    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            save_path = os.path.join(config['result_dir'], 'best_valid_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_valid_loss': best_valid_loss,
                'best_valid_metrics': valid_metrics
            }, save_path)
            print(f"√ Saved best validation model to: {save_path}")

    
        scheduler.step()


    print("\n" + "="*80)
    print("BraTS2021 Train-Valid Pipeline Finished!")
    print("="*80)
    print(f"Results saved to directory: {config['result_dir']}")
    print(f"Best validation model (Epoch {best_epoch+1}): best_valid_model.pth")

if __name__ == '__main__':
    main()
