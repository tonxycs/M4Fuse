import os
import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
import glob
import random
from tqdm import tqdm
import logging
from datetime import datetime
from mamba_ssm import Mamba
import albumentations as A

from torch.cuda.amp import autocast, GradScaler


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


class PetaloMixer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):  # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        n_tokens = D * H * W
        x_flat = x.reshape(B, C, n_tokens).transpose(1, 2)  # (B, N, C)
        x_norm = self.norm(x_flat)
        parts = torch.chunk(x_norm, 4, dim=2)
        outs = []
        for part in parts:
            outs.append(self.mamba(part) + self.skip_scale * part)
        x_mamba = torch.cat(outs, dim=2)
        x_mamba = self.norm(x_mamba)
        x_proj = self.proj(x_mamba).transpose(1, 2)
        return x_proj.reshape(B, -1, D, H, W)


class ChannelBloomBridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_sum = sum(c_list[:-1])
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.get_all_att = nn.Conv1d(1, 1, 3, padding=1, bias=False)
        self.att_layers = nn.ModuleList([
            nn.Linear(c_sum, c_out) if split_att=='fc' else nn.Conv1d(c_sum, c_out, 1)
            for c_out in c_list[:-1]
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, *ts):
        att = torch.cat([self.avgpool(t) for t in ts], dim=1)
        att = att.view(att.size(0), att.size(1), -1).transpose(-1, -2)
        att = self.get_all_att(att)
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        outs = []
        for layer, t in zip(self.att_layers, ts):
            if self.split_att == 'fc':
                a = self.sigmoid(layer(att)).transpose(-1, -2)
            else:
                a = self.sigmoid(layer(att))
            a = a.unsqueeze(-1).unsqueeze(-1).expand_as(t)
            outs.append(a * t)
        return tuple(outs)


class SpatialPetalBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, *ts):
        outs = []
        for t in ts:
            avg_feat = t.mean(1, keepdim=True)
            max_feat, _ = t.max(1, keepdim=True)
            att = torch.cat([avg_feat, max_feat], dim=1)
            outs.append(self.conv(att))
        return tuple(outs)


class FullBloomBridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.satt = SpatialPetalBridge()
        self.catt = ChannelBloomBridge(c_list, split_att)

    def forward(self, *ts):
        spatial_att = self.satt(*ts)
        ts_spatial = [t * a for t, a in zip(ts, spatial_att)]
        channel_att = self.catt(*ts_spatial)
        return tuple([t_sp + c_att + t for t_sp, c_att, t in zip(ts_spatial, channel_att, ts)])


class PetalExpertUnit(nn.Module):
    def __init__(self, in_dim, out_dim, num_modalities=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.shared = PetaloMixer(in_dim, out_dim, d_state, d_conv, expand)
        self.experts = nn.ModuleList([
            PetaloMixer(in_dim, out_dim, d_state, d_conv, expand)
            for _ in range(num_modalities)  # 0:HGG, 1:LGG (experts)
        ])
        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x, dataset_id):
        shared_out = self.shared(x)
        expert_out = torch.zeros_like(shared_out)
        for i in range(x.size(0)):
            m = dataset_id[i].item()
            expert_out[i] = self.experts[m](x[i:i+1])[0]
        return self.dropout(shared_out + expert_out)


class M4Fuse(nn.Module):
    def __init__(self, num_classes=4,
                 input_channels=4,
                 c_list = [32, 64, 96, 128, 192, 256],   #[48, 96, 144, 192, 288, 384] Large   [32, 64, 96, 128, 192, 256]  [24, 48, 72, 96, 144, 192] [16, 32, 48, 64, 96, 128]
                 modalities=2,
                 split_att='fc',
                 bridge=True):
        super().__init__()
        self.bridge = bridge

        # Encoder
        self.enc1 = nn.Conv3d(input_channels, c_list[0], 3, 1, 1)
        self.enc2 = nn.Conv3d(c_list[0], c_list[1], 3, 1, 1)
        self.enc3 = nn.Conv3d(c_list[1], c_list[2], 3, 1, 1)
        self.ex4 = PetalExpertUnit(c_list[2], c_list[3], modalities)
        self.ex5 = PetalExpertUnit(c_list[3], c_list[4], modalities)
        self.ex6 = PetalExpertUnit(c_list[4], c_list[5], modalities)

        if self.bridge:
            self.fbb = FullBloomBridge(c_list, split_att)

        # Decoder
        self.dec1 = PetaloMixer(c_list[5], c_list[4])
        self.dec2 = PetaloMixer(c_list[4], c_list[3])
        self.dec3 = PetaloMixer(c_list[3], c_list[2])
        self.dec4 = nn.Conv3d(c_list[2], c_list[1], 3, 1, 1)
        self.dec5 = nn.Conv3d(c_list[1], c_list[0], 3, 1, 1)

   
        self.ebn1 = nn.GroupNorm(4, c_list[0]); self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2]); self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4]); self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3]); self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1]); self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.dropout = nn.Dropout3d(p=0.2)
        self.final = nn.Conv3d(c_list[0], num_classes, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, dataset_id):
        # Encoder
        out = F.gelu(self.ebn1(self.enc1(x))); t1 = out
        out = F.max_pool3d(out, 2)
        out = F.gelu(self.ebn2(self.enc2(out))); t2 = out
        out = F.max_pool3d(out, 2)
        out = F.gelu(self.ebn3(self.enc3(out))); t3 = out
        out = self.dropout(out)
        out = F.max_pool3d(out, 2)
        out = F.gelu(self.ebn4(self.ex4(out, dataset_id))); t4 = out
        out = F.max_pool3d(out, 2)
        out = F.gelu(self.ebn5(self.ex5(out, dataset_id))); t5 = out

        if self.bridge:
            t1, t2, t3, t4, t5 = self.fbb(t1, t2, t3, t4, t5)

        out = F.gelu(self.ex6(out, dataset_id))

        # Decoder
        out = F.gelu(self.dbn1(self.dec1(out))) + t5
        out = F.interpolate(out, size=t4.shape[2:], mode='trilinear', align_corners=True)
        out = F.gelu(self.dbn2(self.dec2(out))) + t4
        out = F.interpolate(out, size=t3.shape[2:], mode='trilinear', align_corners=True)
        out = F.gelu(self.dbn3(self.dec3(out))) + t3
        out = self.dropout(out)
        out = F.interpolate(out, size=t2.shape[2:], mode='trilinear', align_corners=True)
        out = F.gelu(self.dbn4(self.dec4(out))) + t2
        out = F.interpolate(out, size=t1.shape[2:], mode='trilinear', align_corners=True)
        out = F.gelu(self.dbn5(self.dec5(out))) + t1

        return self.final(out)


class BraTSDataset(Dataset):
    def __init__(self, data_dir, transform=None, crop_size=(128, 128, 128)):
        self.data_dir = data_dir
        self.transform = transform
        self.crop_size = crop_size
        self.patients = []
        self.patient_types = []  # 0:HGG, 1:LGG
        
        for grade_id, grade in enumerate(['HGG', 'LGG']):
            grade_dir = os.path.join(data_dir, grade)
            if os.path.exists(grade_dir):
                patient_dirs = [d for d in glob.glob(os.path.join(grade_dir, '*')) if os.path.isdir(d)]
                self.patients.extend(patient_dirs)
                self.patient_types.extend([grade_id] * len(patient_dirs))
        
        print(f"Total sample size: {len(self.patients)} | HGG/LGG ratio maintained for stratified sampling")

    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        patient_dir = self.patients[idx]
        dataset_id = torch.tensor(self.patient_types[idx], dtype=torch.long)
        
   
        modalities = []
        for modal in ['t1', 't1ce', 't2', 'flair']:
            modal_path = glob.glob(os.path.join(patient_dir, f'*{modal}.nii.gz'))[0]
            img = nib.load(modal_path).get_fdata()
            modalities.append(img)
        
   
        seg_path = glob.glob(os.path.join(patient_dir, '*seg.nii.gz'))[0]
        seg = nib.load(seg_path).get_fdata()
        seg[seg == 4] = 3  
        
   
        data = np.stack(modalities, axis=0).astype(np.float32)
        for i in range(data.shape[0]):
            mean = np.mean(data[i])
            std = np.std(data[i])
            data[i] = (data[i] - mean) / std if std > 0 else data[i] - mean
        
   
        data = torch.from_numpy(data)
        seg = torch.from_numpy(seg).long()
        
     
        if self.transform:
            data, seg = self.transform(data, seg)
        
        return data, seg, dataset_id


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


class RandomElasticTransform:
    def __init__(self, alpha=800, sigma=40, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        
    def __call__(self, data, seg):
        if random.random() < self.p:
 
            data_np = data.permute(1, 2, 3, 0).numpy()  # (C,D,H,W) → (D,H,W,C)
            seg_np = seg.numpy()[..., None]  # (D,H,W) → (D,H,W,1)
            
     
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


class ModalContrastEnhance:
    def __init__(self, p=0.6):
        self.p = p
        
    def __call__(self, data, seg):
        if random.random() < self.p:
            for c in range(data.shape[0]):  
                mean = data[c].mean()
                std = data[c].std()
                if std > 0:
             
                    contrast_factor = random.uniform(0.8, 1.2)
                    data[c] = (data[c] - mean) * contrast_factor + mean
        return data, seg

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, data, seg):
        for t in self.transforms:
            data, seg = t(data, seg)
        return data, seg


class WeightedDiceLoss(nn.Module):
    def __init__(self, class_weights=[1.0, 1.2, 1.2, 2.5], smooth=1e-5):
        super().__init__()
        self.smooth = smooth

        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        total_loss = 0
        device = pred.device
        weights = self.class_weights.to(device)
        
        for c in range(pred.shape[1]):
            pred_c = pred[:, c]
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_loss_c = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
            total_loss += dice_loss_c * weights[c]  
        
        return total_loss / weights.sum() 

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.8, weight_ce=0.2):
        super().__init__()
        self.dice_loss = WeightedDiceLoss()
   
        self.ce_weight = torch.tensor([0.5, 1.0, 1.0, 2.0])  
        self.ce_loss = nn.CrossEntropyLoss()  
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        
    def forward(self, pred, target):

        ce_weight = self.ce_weight.to(pred.device)
        ce_loss_val = F.cross_entropy(pred, target, weight=ce_weight)
        return self.weight_dice * self.dice_loss(pred, target) + self.weight_ce * ce_loss_val

def compute_brats_metrics(pred, target):

    pred = pred.to(torch.float32)
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


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    metrics = {'WT': [], 'TC': [], 'ET': [], 'Total': []}
    scaler = GradScaler()
    
    for data, target, dataset_id in tqdm(dataloader, desc="Training", mininterval=60, maxinterval=60):
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
        for data, target, dataset_id in tqdm(dataloader, desc="Validation", mininterval=60, maxinterval=60):
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


def main():

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_dir = '~/MICCAI_BraTS_2019_Data_Training'
    batch_size = 2
    learning_rate = 1e-4
    num_epochs = 200
    crop_size = (128, 128, 128)
    n_splits = 5
    patience = 80
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    output_root = f"~./Results_2019"
    os.makedirs(output_root, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_root, 'overall.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


    dataset = BraTSDataset(
        data_dir=data_dir,
        transform=None,
        crop_size=crop_size
    )
    all_indices = np.arange(len(dataset))
    patient_types = np.array(dataset.patient_types)
    

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    

    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, patient_types)):
        print(f"\n===== Fold {fold+1}/{n_splits}  =====")
        logging.info(f"\n===== Fold {fold+1}/{n_splits}  =====")
        
 
        fold_dir = os.path.join(output_root, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        

        train_dataset.dataset.transform = Compose([
            RandomCrop(crop_size),
            RandomElasticTransform(alpha=800, sigma=40),
            ModalContrastEnhance(p=0.6)
        ])
        val_dataset.dataset.transform = Compose([])
        

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        

        model = M4Fuse(
            num_classes=4,
            input_channels=4,
            modalities=2
        ).to(device)
        
       
        expert_params = []
        for expert in [model.ex4.experts, model.ex5.experts, model.ex6.experts]:
         
            expert_params.append({
                'params': expert[0].parameters(), 
                'lr': learning_rate,
                'weight_decay': 1e-6
            })
 
            expert_params.append({
                'params': expert[1].parameters(), 
                'lr': learning_rate * 0.5,
                'weight_decay': 5e-6
            })
        
     
        other_params = [
            {'params': model.enc1.parameters(), 'weight_decay': 1e-6},
            {'params': model.enc2.parameters(), 'weight_decay': 1e-6},
            {'params': model.enc3.parameters(), 'weight_decay': 1e-6},
            {'params': model.ex4.shared.parameters(), 'weight_decay': 1e-6},
            {'params': model.ex5.shared.parameters(), 'weight_decay': 1e-6},
            {'params': model.ex6.shared.parameters(), 'weight_decay': 1e-6},
            {'params': model.dec1.parameters(), 'weight_decay': 1e-6},
            {'params': model.dec2.parameters(), 'weight_decay': 1e-6},
            {'params': model.dec3.parameters(), 'weight_decay': 1e-6},
            {'params': model.dec4.parameters(), 'weight_decay': 1e-6},
            {'params': model.dec5.parameters(), 'weight_decay': 1e-6},
            {'params': model.final.parameters(), 'weight_decay': 1e-6},
        ]
        if model.bridge:
            other_params.append({'params': model.fbb.parameters(), 'weight_decay': 1e-6})
        

        optimizer = optim.AdamW(
            expert_params + other_params,
            lr=learning_rate,
            weight_decay=1e-6  
        )
        
   
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=5e-7, verbose=True
        )
        
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {total_params / 1e6:.2f} M")
        
      
        best_total = -1
        best_val_metrics = None
        best_epoch = 0
        counter = 0  
        

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
           
            train_loss, train_metrics = train_epoch(model, train_loader, CombinedLoss(), optimizer, device)
           
            val_loss, val_metrics = validate(model, val_loader, CombinedLoss(), device)
            
         
            print(f"【Epoch {epoch+1} Train Metric】")
            print(f"Train Loss: {train_loss:.4f} | WT: {train_metrics['WT']:.4f} | TC: {train_metrics['TC']:.4f} | ET: {train_metrics['ET']:.4f} | Total: {train_metrics['Total']:.4f}")
            print(f"【Epoch {epoch+1} Val Metric】")
            print(f"Val Metric: {val_loss:.4f} | WT: {val_metrics['WT']:.4f} | TC: {val_metrics['TC']:.4f} | ET: {val_metrics['ET']:.4f} | Total: {val_metrics['Total']:.4f}")
            logging.info(f"【Epoch {epoch+1} train Metric】Train Metric: {train_loss:.4f}, WT: {train_metrics['WT']:.4f}, TC: {train_metrics['TC']:.4f}, ET: {train_metrics['ET']:.4f}, Total: {train_metrics['Total']:.4f}")
            logging.info(f"【Epoch {epoch+1} Val Metric】Val Metric: {val_loss:.4f}, WT: {val_metrics['WT']:.4f}, TC: {val_metrics['TC']:.4f}, ET: {val_metrics['ET']:.4f}, Total: {val_metrics['Total']:.4f}")

            
         
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
                    'best_metrics': best_val_metrics
                }, os.path.join(fold_dir, 'best_model.pth'))
        
                print(f"Discovering the new best model（Epoch {best_epoch}）：")
                print(f"WT: {best_val_metrics['WT']:.4f} | TC: {best_val_metrics['TC']:.4f} | ET: {best_val_metrics['ET']:.4f} | Total: {best_val_metrics['Total']:.4f}")
                logging.info(f"Discovering the new best model（Epoch {best_epoch}）：WT: {best_val_metrics['WT']:.4f}, TC: {best_val_metrics['TC']:.4f}, ET: {best_val_metrics['ET']:.4f}, Total: {best_val_metrics['Total']:.4f}")
            else:
                counter += 1  
                print(f"No performance improvement detected; stop counting early.: {counter}/{patience}")
                if counter >= patience:
                    print(f"Early Stopping Mechanism Triggered! Stop current fold training.")
                    logging.info(f"Early Stopping Mechanism Triggered! Stop current fold training.")
                    break
        
        fold_metrics.append(best_val_metrics)
        print(f"\nFold {fold+1} Ultimate Best Metric（Epoch {best_epoch}）：")
        print(f"WT: {best_val_metrics['WT']:.4f} | TC: {best_val_metrics['TC']:.4f} | ET: {best_val_metrics['ET']:.4f} | Total: {best_val_metrics['Total']:.4f}")
        logging.info(f"\nFold {fold+1} Ultimate Best Metric：WT: {best_val_metrics['WT']:.4f}, TC: {best_val_metrics['TC']:.4f}, ET: {best_val_metrics['ET']:.4f}, Total: {best_val_metrics['Total']:.4f}")
    

    avg_wt = np.mean([m['WT'] for m in fold_metrics])
    avg_tc = np.mean([m['TC'] for m in fold_metrics])
    avg_et = np.mean([m['ET'] for m in fold_metrics])
    avg_total = np.mean([m['Total'] for m in fold_metrics])
    
    print(f"\n===== 5-Fold Cross-Validation Average Results =====")
    print(f"Avg-WT Dice: {avg_wt:.4f}")
    print(f"Avg-TC Dice: {avg_tc:.4f}")
    print(f"Avg-ET Dice: {avg_et:.4f}")
    print(f"Avg-Total Dice: {avg_total:.4f}")
    
    logging.info(f"\n===== 5-Fold Average Results =====")
    logging.info(f"Avg-WT Dice: {avg_wt:.4f}")
    logging.info(f"Avg-TC Dice: {avg_tc:.4f}")
    logging.info(f"Avg-ET Dice: {avg_et:.4f}")
    logging.info(f"Avg-Total Dice: {avg_total:.4f}")
    

    with open(os.path.join(output_root, 'fold_results.txt'), 'w') as f:
        for i, m in enumerate(fold_metrics):
            f.write(f"Fold {i+1}: WT {m['WT']:.4f}, TC {m['TC']:.4f}, ET {m['ET']:.4f}, Total {m['Total']:.4f}\n")
        f.write(f"Average: WT {avg_wt:.4f}, TC {avg_tc:.4f}, ET {avg_et:.4f}, Total {avg_total:.4f}\n")

if __name__ == "__main__":
    main()
    
