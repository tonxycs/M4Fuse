import os
import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
import glob
import random
from tqdm import tqdm
import logging
import csv
import albumentations as A
from scipy.ndimage import distance_transform_edt
from torch.cuda.amp import autocast, GradScaler
from mamba_ssm import Mamba
import warnings



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
set_seed()


AblationConfig = {
    "Gate_LearnedBias": {  
        "gate_type": "learned_bias",
        "granularity": "pixel+channel",
        "scale_type": "all",
        "period_type": "all_stages",
        "csb_module": "full",
        "peu_module": "full",
        "pom_module": "full"
    }
}
exp_name = "Gate_LearnedBias" 
exp_params = AblationConfig[exp_name]  


# AblationConfig = {
#     "POM3_Skip+Linproj": {
#         "gate_type": "original", "granularity": "pixel+channel", 
#         "scale_type": "all", "period_type": "all_stages",
#         "csb_module": "full", "peu_module": "full",
#         "pom_module": "skip_linproj"  # Skip + l+j
#     }
# }
# exp_name = "POM3_Skip+Linproj"
# exp_params = AblationConfig[exp_name]
#....





class PetaloMixer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, pom_module="full"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.norm = nn.LayerNorm(input_dim)

        self.mamba = Mamba(d_model=input_dim // 4, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(input_dim, output_dim) 
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.pom_module = pom_module


    def forward(self, x):
        B, C, D, H, W = x.shape
        n_tokens = D * H * W
        x_flat = x.reshape(B, C, n_tokens).transpose(1, 2)  # (B, N, C)
        x_norm = self.norm(x_flat)

        if self.pom_module == "ssms_only":

            parts = torch.chunk(x_norm, 4, dim=2)
            outs = [self.mamba(part) for part in parts]
            x_mamba = torch.cat(outs, dim=2)
            x_proj = self.proj(x_mamba)  
            return x_proj.transpose(1, 2).reshape(B, -1, D, H, W)
        
        elif self.pom_module == "ssm_skip":  
            parts = torch.chunk(x_norm, 4, dim=2)
            outs = [self.mamba(part) + self.skip_scale * part for part in parts]  
            x_mamba = torch.cat(outs, dim=2)  
     
            x_proj = self.proj(x_mamba)
            return x_proj.transpose(1, 2).reshape(B, -1, D, H, W)
        elif self.pom_module == "ssm_linproj":  
            parts = torch.chunk(x_norm, 4, dim=2)
            outs = [self.mamba(part) for part in parts]  
            x_mamba = torch.cat(outs, dim=2)
            x_proj = self.proj(x_mamba).transpose(1, 2) 
            return x_proj.reshape(B, -1, D, H, W)
        elif self.pom_module == "skip_linproj": 
            x_skip = self.skip_scale * x_norm  
            x_proj = self.proj(x_skip).transpose(1, 2)  
            return x_proj.reshape(B, -1, D, H, W)
        elif self.pom_module == "skip_only":
            x_skip = self.skip_scale * x_norm
            return x_skip.transpose(1, 2).reshape(B, -1, D, H, W)
        
        elif self.pom_module == "linproj_only":
            x_proj = self.proj(x_norm).transpose(1, 2)
            return x_proj.reshape(B, -1, D, H, W)
        
        elif self.pom_module == "route_only":
            return x
        
        else:  # full
            parts = torch.chunk(x_norm, 4, dim=2)
            outs = [self.mamba(part) + self.skip_scale * part for part in parts]
            x_mamba = torch.cat(outs, dim=2)
            x_mamba = self.norm(x_mamba)
            x_proj = self.proj(x_mamba).transpose(1, 2)
            return x_proj.reshape(B, -1, D, H, W)

class ChannelAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv(x_cat)
        return x * self.sigmoid(x_att)

class FeatureFusion(nn.Module):

    def __init__(self, c_list):
        super().__init__()

        self.channel_converters = nn.ModuleDict({
            "t2_to_t3": nn.Conv3d(c_list[1], c_list[2], 1),  # 64->96
            "t3_to_t4": nn.Conv3d(c_list[2], c_list[3], 1),  # 96->128
            "t4_to_t5": nn.Conv3d(c_list[3], c_list[4], 1),  # 128->192
            "t5_to_t4": nn.Conv3d(c_list[4], c_list[3], 1),  # 192->128
            "t4_to_t3": nn.Conv3d(c_list[3], c_list[2], 1),  # 128->96
            "t3_to_t2": nn.Conv3d(c_list[2], c_list[1], 1)   # 96->64
        })
        

        self.attention = nn.ModuleDict({
            "t1": nn.Sequential(SpatialAttention(), ChannelAttention(c_list[0])),
            "t2": nn.Sequential(SpatialAttention(), ChannelAttention(c_list[1])),
            "t3": nn.Sequential(SpatialAttention(), ChannelAttention(c_list[2])),
            "t4": nn.Sequential(SpatialAttention(), ChannelAttention(c_list[3])),
            "t5": nn.Sequential(SpatialAttention(), ChannelAttention(c_list[4]))
        })

    def forward(self, t1, t2, t3, t4, t5, scale_type):
        if scale_type == "jump":
     
            target_size = t3.shape[2:]
            target_channels = t3.shape[1] 
            

            t4_adjusted = F.interpolate(t4, size=target_size, mode='trilinear', align_corners=True)
            t4_adjusted = self.channel_converters["t4_to_t3"](t4_adjusted)
            
  
            t5_adjusted = F.interpolate(t5, size=target_size, mode='trilinear', align_corners=True)
            t5_adjusted = self.channel_converters["t5_to_t4"](t5_adjusted)
            t5_adjusted = self.channel_converters["t4_to_t3"](t5_adjusted)
            
       
            fused = (t3 + t4_adjusted + t5_adjusted) / 3
            
          
            fused = self.attention["t3"](fused)
            
  
            t3_fused = fused  
            
            t4_fused = F.interpolate(fused, size=t4.shape[2:], mode='trilinear', align_corners=True)
            t4_fused = self.channel_converters["t3_to_t4"](t4_fused)  # 96->128
            
            t5_fused = F.interpolate(fused, size=t5.shape[2:], mode='trilinear', align_corners=True)
            t5_fused = self.channel_converters["t3_to_t4"](t5_fused)  # 96->128
            t5_fused = self.channel_converters["t4_to_t5"](t5_fused)  # 128->192
            
            return t1, t2, t3_fused, t4_fused, t5_fused
            
        elif scale_type == "shallow_deep":

            target_size = t3.shape[2:]
            t2_adjusted = F.interpolate(t2, size=target_size, mode='trilinear', align_corners=True)
            t2_adjusted = self.channel_converters["t2_to_t3"](t2_adjusted)
            
            fused = (t3 + t2_adjusted) / 2
            fused = self.attention["t3"](fused)
            
            return t1, t2, fused, t4, t5
            
        elif scale_type == "deep_shallow":

            target_size = t4.shape[2:]
            t5_adjusted = F.interpolate(t5, size=target_size, mode='trilinear', align_corners=True)
            t5_adjusted = self.channel_converters["t5_to_t4"](t5_adjusted)
            
            fused = (t4 + t5_adjusted) / 2
            fused = self.attention["t4"](fused)
            
            return t1, t2, t3, fused, t5
            
        else:  
 
            return (
                self.attention["t1"](t1),
                self.attention["t2"](t2),
                self.attention["t3"](t3),
                self.attention["t4"](t4),
                self.attention["t5"](t5)
            )

class FullBloomBridge(nn.Module):
    def __init__(self, c_list, split_att='fc', csb_module="full", scale_type="all"):
        super().__init__()
        self.fusion = FeatureFusion(c_list)
        self.csb_module = csb_module
        self.scale_type = scale_type

    def forward(self, *ts):
        if self.csb_module == "route_only":
            return ts
        

        assert len(ts) == 5, f"Expected 5 feature maps, got {len(ts)}"
        t1, t2, t3, t4, t5 = ts
        

        return self.fusion(t1, t2, t3, t4, t5, self.scale_type)

class PetalExpertUnit(nn.Module):
    def __init__(self, in_dim, out_dim, num_modalities=2, d_state=16, d_conv=4, expand=2,
                 gate_type="original", granularity="pixel+channel", peu_module="full"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.shared = PetaloMixer(in_dim, out_dim, d_state, d_conv, expand)
        self.experts = nn.ModuleList([
            PetaloMixer(in_dim, out_dim, d_state, d_conv, expand)
            for _ in range(num_modalities)
        ])
        self.dropout = nn.Dropout3d(p=0.2)
        self.gate_type = gate_type
        self.granularity = granularity
        self.peu_module = peu_module
        

        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_modalities * out_dim, (num_modalities * out_dim) // 4, 1),
            nn.GELU(),
            nn.Conv3d((num_modalities * out_dim) // 4, num_modalities * out_dim, 1),
            nn.Sigmoid()
        ) if granularity == "channel" else None
        

        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim//4),
            nn.GELU(),
            nn.Linear(in_dim//4, num_modalities)
        ) if gate_type == "mini_attn" else None
        
 
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_modalities)]) \
            if gate_type == "learned_bias" else None
        

        self.softmax_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), 
            nn.Flatten(),  
            nn.Linear(in_dim, in_dim // 2), 
            nn.GELU(),
            nn.Linear(in_dim // 2, num_modalities)  
        ) if gate_type == "softmax_top1" else None


    def forward(self, x, dataset_id):
        if self.peu_module == "route_only":
            return self.shared(x)
        elif self.peu_module == "none":
            return torch.zeros(x.shape[0], self.out_dim, *x.shape[2:], device=x.device)
        elif self.peu_module == "expert_only":
            expert_out = torch.zeros(x.shape[0], self.out_dim, *x.shape[2:], device=x.device)
            for i in range(x.size(0)):
                m = dataset_id[i].item()
                m = min(m, len(self.experts) - 1) 
                expert_out[i] = self.experts[m](x[i:i+1])[0]
            return self.dropout(expert_out)
        
        shared_out = self.shared(x)
        B, C, D, H, W = x.shape
        num_experts = len(self.experts)
        out_dim = self.out_dim 

        if self.granularity == "patch":
  
            patch_d, patch_h, patch_w = D//2, H//2, W//2
            patches = [
                x[:, :, d:d+patch_d, h:h+patch_h, w:w+patch_w]
                for d in [0, patch_d] for h in [0, patch_h] for w in [0, patch_w]
            ]
            expert_outs = [self.experts[i % len(self.experts)](p) for i, p in enumerate(patches)]
            w0 = torch.cat([expert_outs[0], expert_outs[1]], dim=4)
            w1 = torch.cat([expert_outs[2], expert_outs[3]], dim=4)
            h = torch.cat([w0, w1], dim=3)
            w2 = torch.cat([expert_outs[4], expert_outs[5]], dim=4)
            w3 = torch.cat([expert_outs[6], expert_outs[7]], dim=4)
            h2 = torch.cat([w2, w3], dim=3)
            expert_out = torch.cat([h, h2], dim=2)
            expert_out = F.interpolate(expert_out, size=(D, H, W), mode='trilinear', align_corners=True)
        
        elif self.granularity == "token":
      
            expert_outs = [expert(x) for expert in self.experts]
            fused = torch.stack(expert_outs, dim=1).flatten(2, 5).transpose(1, 2)
            attn = nn.MultiheadAttention(out_dim, 4, batch_first=True).to(x.device)
            attn_output = attn(fused, fused, fused)[0]
            expert_out = attn_output.transpose(1, 2).reshape(B, len(self.experts), out_dim, D, H, W).mean(dim=1)
        
        elif self.granularity == "channel":
     
            expert_outs = [expert(x) for expert in self.experts]
            fused = torch.cat(expert_outs, dim=1)
            se = self.channel_se(fused)
            expert_out = (fused * se).mean(dim=1, keepdim=True).expand_as(shared_out)
        
        else:  
            expert_out = torch.zeros_like(shared_out)
            if self.gate_type == "softmax_top1":
            
                expert_logits = self.softmax_gate(x)  
                expert_weights = F.softmax(expert_logits, dim=1)  
                for i in range(B):
                    m = torch.argmax(expert_weights[i]).item()  
                    expert_out[i] = self.experts[m](x[i:i+1])[0]
            
            elif self.gate_type == "softmax_top2":
        
                expert_weights = F.softmax(torch.randn(B, len(self.experts), device=x.device), dim=1)
                top2 = torch.topk(expert_weights, 2, dim=1)
                for i in range(B):
                    m1, m2 = top2.indices[i]
                    w1, w2 = top2.values[i]
                    out1 = self.experts[m1](x[i:i+1])[0]
                    out2 = self.experts[m2](x[i:i+1])[0]
                    expert_out[i] = (w1*out1 + w2*out2) / (w1 + w2)
            
            elif self.gate_type == "gumbel":
          
                def gumbel_softmax(logits, tau=1):
                    gumbels = -torch.empty_like(logits).exponential_().log()
                    return F.softmax((logits + gumbels)/tau, dim=1)
                logits = torch.randn(B, len(self.experts), device=x.device)
                expert_weights = gumbel_softmax(logits, tau=0.5)
                for i in range(B):
                    m = torch.argmax(expert_weights[i]).item()
                    expert_out[i] = self.experts[m](x[i:i+1])[0]
            
            elif self.gate_type == "mini_attn":
              
                x_flat = x.mean([2,3,4])
                attn_weights = torch.exp(self.attn(x_flat))
                for i in range(B):
                    m = torch.argmax(attn_weights[i]).item()
                    expert_out[i] = self.experts[m](x[i:i+1])[0]
            
            elif self.gate_type == "learned_bias":
         
                for i in range(B):
                    m = dataset_id[i].item()
                    expert_out[i] = self.experts[m](x[i:i+1])[0] + self.biases[m]
            
            else:  # original
             
                for i in range(B):
                    m = dataset_id[i].item()
                    m = min(m, len(self.experts) - 1) 
                    expert_out[i] = self.experts[m](x[i:i+1])[0]

        return self.dropout(shared_out + expert_out)


class M4Fuse(nn.Module):
    def __init__(self, num_classes=4, input_channels=4,
                 c_list=[32, 64, 96, 128, 192, 256], modalities=2,
                 split_att='fc', bridge=True,** Ablation):
        super().__init__()
        self.bridge = bridge
        self.period_type = Ablation.get("period_type", "all_stages")
        self.scale_type = Ablation.get("scale_type", "all")
        self.c_list = c_list

        peu_params = {k: v for k, v in Ablation.items() 
                      if k in ["gate_type", "granularity", "peu_module", "d_state", "d_conv", "expand"]}
        csb_params = {k: v for k, v in Ablation.items()
                      if k in ["csb_module", "scale_type"]}

 
        self.enc1 = nn.Conv3d(input_channels, c_list[0], 3, 1, 1)
        self.enc2 = nn.Conv3d(c_list[0], c_list[1], 3, 1, 1)
        self.enc3 = nn.Conv3d(c_list[1], c_list[2], 3, 1, 1)
        self.ex4 = PetalExpertUnit(c_list[2], c_list[3], modalities, **peu_params)  
        self.ex5 = PetalExpertUnit(c_list[3], c_list[4], modalities,** peu_params) 
        self.ex6 = PetalExpertUnit(c_list[4], c_list[5], modalities, **peu_params)  

 
        if self.bridge:
            self.fbb = FullBloomBridge(c_list, split_att,** csb_params)


        self.dec1 = PetaloMixer(c_list[5], c_list[4], pom_module=Ablation.get("pom_module", "full")) 
        self.dec2 = PetaloMixer(c_list[4], c_list[3], pom_module=Ablation.get("pom_module", "full")) 
        self.dec3 = PetaloMixer(c_list[3], c_list[2], pom_module=Ablation.get("pom_module", "full"))  
        self.dec4 = nn.Conv3d(c_list[2], c_list[1], 3, 1, 1)  
        self.dec5 = nn.Conv3d(c_list[1], c_list[0], 3, 1, 1)  

   
        self.ebn1 = nn.GroupNorm(4, c_list[0])  
        self.ebn2 = nn.GroupNorm(4, c_list[1])  
        self.ebn3 = nn.GroupNorm(4, c_list[2])  
        self.ebn4 = nn.GroupNorm(4, c_list[3])  
        self.ebn5 = nn.GroupNorm(4, c_list[4])  
        self.dbn1 = nn.GroupNorm(4, c_list[4])  
        self.dbn2 = nn.GroupNorm(4, c_list[3])  
        self.dbn3 = nn.GroupNorm(4, c_list[2])  
        self.dbn4 = nn.GroupNorm(4, c_list[1])  
        self.dbn5 = nn.GroupNorm(4, c_list[0])  

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
 
        out = F.gelu(self.ebn1(self.enc1(x))); t1 = out  # 32
        out = F.max_pool3d(out, 2)
        out = F.gelu(self.ebn2(self.enc2(out))); t2 = out  # 64
        out = F.max_pool3d(out, 2)
        out = F.gelu(self.ebn3(self.enc3(out))); t3 = out  # 96
        out = self.dropout(out)
        out = F.max_pool3d(out, 2)
        out = F.gelu(self.ebn4(self.ex4(out, dataset_id))); t4 = out  # 128
        out = F.max_pool3d(out, 2)
        out = F.gelu(self.ebn5(self.ex5(out, dataset_id))); t5 = out  # 192


        if self.bridge:
            if self.period_type == "early":
                t1 = self.fbb(t1, t2, t3, t4, t5)[0]
            elif self.period_type == "late":
                t5 = self.fbb(t1, t2, t3, t4, t5)[4]
            else:
                t1, t2, t3, t4, t5 = self.fbb(t1, t2, t3, t4, t5)


        out = F.gelu(self.ex6(out, dataset_id))  

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