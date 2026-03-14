import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from mamba_ssm import Mamba  


# 1.PEUnit-PEU (MoE)
class PetalExpertUnit(nn.Module):
    def __init__(self, in_dim, out_dim, num_data_exps=1, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.shared = PetaloMixer(in_dim, out_dim, d_state, d_conv, expand)
        self.experts = nn.ModuleList([
            PetaloMixer(in_dim, out_dim, d_state, d_conv, expand)
            for _ in range(num_data_exps)
        ])
        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x, dataset_id):
        shared_out = self.shared(x)
        expert_out = torch.zeros_like(shared_out)
        for i in range(x.size(0)):
            m = dataset_id[i].item()
            expert_out[i] = self.experts[m](x[i:i+1])[0]
        return self.dropout(shared_out + expert_out)
    


# 2-1.CBridge-CB
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

# 2-2.SBridge-SB
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

# 2.CSBridge-CSB
class CSBridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.satt = SpatialPetalBridge()
        self.catt = ChannelBloomBridge(c_list, split_att)

    def forward(self, *ts):
        spatial_att = self.satt(*ts)
        ts_spatial = [t * a for t, a in zip(ts, spatial_att)]
        channel_att = self.catt(*ts_spatial)
        return tuple([t_sp + c_att + t for t_sp, c_att, t in zip(ts_spatial, channel_att, ts)])

# 3.POMixer-POM (Mamba)
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
        x_flat = x.reshape(B, C, n_tokens).transpose(1, 2)
        x_norm = self.norm(x_flat)
        parts = torch.chunk(x_norm, 4, dim=2)
        outs = []
        for part in parts:
            outs.append(self.mamba(part) + self.skip_scale * part)
        x_mamba = torch.cat(outs, dim=2)
        x_mamba = self.norm(x_mamba)
        x_proj = self.proj(x_mamba).transpose(1, 2)
        return x_proj.reshape(B, -1, D, H, W)




# Overall & Config.
class M4Fuse(nn.Module):
    def __init__(self, num_classes=4,
                 input_channels=4,
                 c_list = [32, 64, 96, 128, 192, 256],   
                # [48, 96, 144, 192, 288, 384] Large #Params:2.45M
                # [32, 64, 96, 128, 192, 256] Base #Params:1.11M
                # [24, 48, 72, 96, 144, 192] Small #Params:0.63M 
                # [16, 32, 48, 64, 96, 128] Tiny #Params:0.29M(Expand for a better and innovative future)
                 data_exps=1,   
                # data_exps: (the number of experts) 1/2/3/4...
                # Allocated based on Data-Types (HGG/LGG...)/modality(MRI/CT...)/Structure(V/L)
                 split_att='fc',
                 bridge=True):
        super().__init__()
        self.bridge = bridge

        # Encoder
        self.enc1 = nn.Conv3d(input_channels, c_list[0], 3, 1, 1)
        self.enc2 = nn.Conv3d(c_list[0], c_list[1], 3, 1, 1)
        self.enc3 = nn.Conv3d(c_list[1], c_list[2], 3, 1, 1)
        self.ex4 = PetalExpertUnit(c_list[2], c_list[3], data_exps)
        self.ex5 = PetalExpertUnit(c_list[3], c_list[4], data_exps)
        self.ex6 = PetalExpertUnit(c_list[4], c_list[5], data_exps)

        if self.bridge:
            self.fbb = CSBridge(c_list, split_att)

        # Decoder
        self.dec1 = PetaloMixer(c_list[5], c_list[4])
        self.dec2 = PetaloMixer(c_list[4], c_list[3])
        self.dec3 = PetaloMixer(c_list[3], c_list[2])
        self.dec4 = nn.Conv3d(c_list[2], c_list[1], 3, 1, 1)
        self.dec5 = nn.Conv3d(c_list[1], c_list[0], 3, 1, 1)

        # Normalization layers
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
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
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

        out = self.final(out)
        return out