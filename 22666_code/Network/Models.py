# Compare-Configure 12s
'''
data_exps = 1/2
-1.Base [32, 64, 96, 128, 192, 256] -> 1.11M

-2.Large [48, 96, 144, 192, 288, 384]  -> 2.45M

-3.Small [24, 48, 72, 96, 144, 192] ->0.63m

-4.Tiny [16, 32, 48, 64, 96, 128] ->0.29M

-5.SuperLightUnet -> 2.97M

    model = NormalU_Net(
        init_channels=4,          
        n_channels=24,            
        class_nums=4,             
        depths_unidirectional="small" 
    ).to(config['device'])

-6.LightM-UNet -> 5.02M

    model = LightMUNet(
        spatial_dims=3,
        init_filters=32,
        in_channels=4,
        out_channels=4,
        dropout_prob=0.2,
        norm=("GROUP", {"num_groups": 8}),
        blocks_down=(1, 1, 2, 4),
        blocks_up=(1, 1, 1)
    ).to(config['device'])
    
-7.3D-Unet

    model = UNet3D(
        in_channels=4,
        out_channels=4,
        num_groups=8,
        dropout_prob=0.2
    ).to(config['device'])
    

-8.nnUnet:

    model = nnUnet(
        num_classes=4,
        input_channels=4,
        base_channels=16
    ).to(config['device'])
    
-9.TransBTS:

     model = TransBTS(
         dataset='brats',
         conv_repr=True,
         pe_type="learned"
     ).to(config['device'])
     
     
-10.SegResNet: 

from monai.networks.nets import SwinUNETR
    model = SegResNet(
        in_channels=4,
        out_channels=4,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=32
    ).to(config['device'], dtype=torch.float32)
    
-11.SwinUNETR: 

from monai.networks.nets import SwinUNETR
    model = SwinUNETR(
        img_size=config['target_size'],
        in_channels=4,
        out_channels=4,
        feature_size=48,
        use_checkpoint=False
    ).to(config['device'], dtype=torch.float32)

-12.SegMamba:

    model = SegMamba(
        in_chans=4,
        out_chans=4, 
        depths=[2, 2, 2, 2],
        feat_size=[32, 64, 128, 256],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6
    ).to(config['device'], dtype=torch.float32)
'''






#----------------------------------------Parts(Lightweight Architecture Coding)------------------------------------------------#







    
    
'''
8.LightMUnet

    model = LightMUNet(
        spatial_dims=3,
        init_filters=32,
        in_channels=4,
        out_channels=4,
        dropout_prob=0.2,
        norm=("GROUP", {"num_groups": 8}),
        blocks_down=(1, 1, 2, 4),
        blocks_up=(1, 1, 1)
    ).to(config['device'])
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.cuda.amp import autocast, GradScaler
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom, rotate
import scipy.stats as stats
from typing import Union, Tuple, List, Optional


from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.metrics.utils import get_surface_distance
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.utils import InterpolateMode, UpsampleMode
from monai.networks.layers import get_act_layer, get_norm_layer, Dropout
from monai.networks.layers import AffineTransform


from mamba_ssm import Mamba

# ----------------------
# LightMUNet: 5.02M
# ----------------------
def get_dwconv_layer(
    spatial_dims: int, 
    in_channels: int, 
    out_channels: int, 
    kernel_size: int = 3, 
    stride: int = 1, 
    bias: bool = False
):

    depth_conv = torch.nn.Conv3d(
        in_channels=in_channels,
        out_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,  
        groups=in_channels,  
        bias=bias
    )
    # （Point-wise Conv）
    point_conv = torch.nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=bias
    )
    return torch.nn.Sequential(depth_conv, point_conv)


class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


def get_mamba_layer(
    spatial_dims: int, 
    in_channels: int, 
    out_channels: int, 
    stride: int = 1
):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        if spatial_dims == 3:
            return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer


class ResMambaBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: Union[Tuple, str],
        kernel_size: int = 3,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )
        self.conv2 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        x += identity
        return x


class ResUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: Union[Tuple, str],
        kernel_size: int = 3,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_dwconv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)
        return x


class LightMUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 4,
        out_channels: int = 4,
        dropout_prob: Union[float, None] = None,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: Tuple = (1, 2, 2, 4),
        blocks_up: Tuple = (1, 1, 1),
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_dwconv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            downsample_mamba = (
                get_mamba_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                downsample_mamba, *[ResMambaBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 **(n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResUpBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        torch.nn.Conv3d(sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_act_layer(self.act),
                        UpSample(
                            spatial_dims=spatial_dims,
                            in_channels=sample_in_channels // 2,
                            out_channels=sample_in_channels // 2,
                            mode=self.upsample_mode,
                            kernel_size=2,
                            scale_factor=2,  # 用scale_factor替代stride
                        ),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            torch.nn.Conv3d(self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []
        for down in self.down_layers:
            x = down(x)
            down_x.append(x)
        return x, down_x

    def decode(self, x: torch.Tensor, down_x: List[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x)
            if x.shape[2:] != down_x[i + 1].shape[2:]:
                x = F.interpolate(
                    x,
                    size=down_x[i + 1].shape[2:],
                    mode='trilinear' if x.dim() == 5 else 'bilinear',
                    align_corners=False
                )
            x = x + down_x[i + 1]
            x = upl(x)
        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, input: torch.Tensor, dummy_tensor=None) -> torch.Tensor:
        x, down_x = self.encode(input)
        down_x.reverse()
        x = self.decode(x, down_x)
        return x


'''

SuperlightNet

    model = NormalU_Net(
        init_channels=4,        
        n_channels=24,          
        class_nums=4,            
        depths_unidirectional="small"  
    ).to(config['device'])
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.cuda.amp import autocast, GradScaler
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom, rotate
import scipy.stats as stats


from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.metrics.utils import get_surface_distance
# from monai.networks.layers import Convolution, UpSample, UpsampleMode, InterpolateMode


from einops import rearrange, repeat
from torch import Tensor

import torch.utils.checkpoint as checkpoint

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.utils import InterpolateMode, UpsampleMode
# ----------------------
# SuperLightUnet（2.97M）
# ----------------------

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, input_x):
        if self.data_format == "channels_last":
            return F.layer_norm(input_x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = input_x.mean(1, keepdim=True)
            s = (input_x - u).pow(2).mean(1, keepdim=True)
            input_x = (input_x - u) / torch.sqrt(s + self.eps)
            input_x = self.weight[:, None, None] * input_x + self.bias[:, None, None]
            return input_x
        return None


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, x=8, y=8):
        super().__init__()
        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_in, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(
            F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(
            F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        x4 = self.dw(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.norm2(x)
        x = self.ldw(x)
        return x


class THPAEncFR3(nn.Module):
    def __init__(self, in_channels, expr):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(in_channels // 2)
        self.GHPA_dim = Grouped_multi_axis_Hadamard_Product_Attention(in_channels // 2, in_channels // 2)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.mlp = MlpChannel(in_channels, expr)

    def forward(self, input_x: Tensor, dummy_tensor=None):
        input_x, x_residual = torch.chunk(input_x, 2, dim=1)
        input_x = self.norm1(input_x)
        B, C, W, H, D = input_x.shape

        random_direction = torch.randint(0, 3, (1,)).item()
        if random_direction == 0:
            WHD_dim = rearrange(self.GHPA_dim(rearrange(input_x, "b c w h d -> (h b) c w d")),
                                "(h b) c w d -> b c w h d", b=B)
            x_re = rearrange(input_x, "b c w h d -> (h b) c w d").flip([0])
            rWHD_dim = rearrange(self.GHPA_dim(x_re), "(h b) c w d -> b c w h d", b=B).flip([0])
            WHD_dim = WHD_dim + rWHD_dim
        elif random_direction == 1:
            WHD_dim = rearrange(self.GHPA_dim(rearrange(input_x, "b c w h d -> (w b) c h d")),
                                "(w b) c h d -> b c w h d", b=B)
            x_re = rearrange(input_x, "b c w h d -> (w b) c h d").flip([0])
            rWHD_dim = rearrange(self.GHPA_dim(x_re), "(w b) c h d -> b c w h d", b=B).flip([0])
            WHD_dim = WHD_dim + rWHD_dim
        elif random_direction == 2:
            WHD_dim = rearrange(self.GHPA_dim(rearrange(input_x, "b c w h d -> (d b) c w h")),
                                "(d b) c w h -> b c w h d", b=B)
            x_re = rearrange(input_x, "b c w h d -> (d b) c w h").flip([0])
            rWHD_dim = rearrange(self.GHPA_dim(x_re), "(d b) c w h -> b c w h d", b=B).flip([0])
            WHD_dim = WHD_dim + rWHD_dim
        else:
            raise NotImplementedError
        input_x = torch.cat((WHD_dim, x_residual), dim=1)
        input_x = self.norm2(input_x)
        input_x = self.mlp(input_x)
        return input_x


class NormDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channels)
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, input, dummy_tensor=None):
        return self.proj(self.norm(input))


class Learnable_Res_Skip_UpRepr4(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims=3):
        super().__init__()
        self.upc = Convolution(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, strides=1,
            kernel_size=1, bias=False, conv_only=True
        )
        self.ups = UpSample(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            scale_factor=2,
            mode=UpsampleMode.NONTRAINABLE,
            interp_mode=InterpolateMode.LINEAR,
            align_corners=False,
        )

        self.repr_mldw = nn.Sequential(Convolution(spatial_dims=spatial_dims, in_channels=out_channels,
                                                   out_channels=out_channels, strides=1,
                                                   kernel_size=3, bias=False, conv_only=True, groups=out_channels // 12),
                                       nn.GELU(),
                                       Convolution(spatial_dims=spatial_dims, in_channels=out_channels,
                                                   out_channels=out_channels,
                                                   strides=1, kernel_size=1, bias=False, conv_only=True, groups=1)
                                       )

        self.norm = nn.InstanceNorm3d(out_channels)
        self.group_skip_scale = nn.Parameter(torch.Tensor(1, out_channels, 1, 1, 1), requires_grad=True)
        nn.init.ones_(self.group_skip_scale)
        self.group_res_scale = nn.Parameter(torch.Tensor(1), requires_grad=True)
        nn.init.ones_(self.group_res_scale)

    def forward(self, inp_skip, dummy_tensor=None):
        input, skip = inp_skip
        # 1. 通道调整 + 上采样
        input = self.upc(input)
        input = self.ups(input)  # 上采样后可能存在尺寸偏差
        
        # 2. 关键修复：强制对齐 input 和 skip 的尺寸（核心步骤）
        # 若 input 与 skip 尺寸不同，用插值调整 input 到 skip 的尺寸
        if input.shape[2:] != skip.shape[2:]:  # 只比较空间维度（忽略 batch 和 channel）
            input = F.interpolate(
                input,
                size=skip.shape[2:],  # 对齐到 skip 的空间尺寸
                mode='trilinear' if input.dim() == 5 else 'bilinear',  # 3D用trilinear，2D用bilinear
                align_corners=False
            )
        
        # 3. 跳跃连接融合（此时尺寸已匹配）
        input = input + skip * self.group_skip_scale
        res = input

        input = self.norm(input)
        out = self.repr_mldw(input)

        return out + res * self.group_res_scale


class MlpChannel(nn.Module):
    def __init__(self, in_channels, expr=1, out_channels=None):
        if out_channels is None:
            out_channels = in_channels
        super().__init__()
        self.fc1 = nn.Conv3d(in_channels, in_channels * expr, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(in_channels * expr, out_channels, 1)

    def forward(self, input_x):
        input_x = self.fc1(input_x)
        input_x = self.act(input_x)
        input_x = self.fc2(input_x)
        return input_x


def block_creator(coder_str, depths_unidirectional, in_channels, out_channels=0):
    if out_channels == 0:
        out_channels = in_channels

    if coder_str == "NormDownsample":
        block = NormDownsample(in_channels, out_channels)
    elif coder_str == "THPAEncFR3":
        block = nn.Sequential(*[
            THPAEncFR3(in_channels, expr=2)
            for _ in range(depths_unidirectional)
        ])
    elif coder_str == "Learnable_Res_Skip_UpRepr4":
        block = Learnable_Res_Skip_UpRepr4(in_channels, out_channels)
    else:
        print("encoder error")
        raise NotImplementedError
    return block


class JCMNetv8Enc(nn.Module):
    def __init__(self,
                 init_channels=4,
                 n_channels=32,
                 class_nums=4,
                 checkpoint_style="",
                 expr=2,
                 depths_unidirectional=None,
                 ):
        super(JCMNetv8Enc, self).__init__()

        # 核心修复：将depths_unidirectional定义为实例属性
        self.depths_unidirectional = depths_unidirectional

        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        else:
            self.outside_block_checkpointing = False

        if depths_unidirectional is None:
            raise NotImplementedError
        elif depths_unidirectional == "small":
            self.depths = [1, 1, 2, 2, 2]  # 用self.depths存储具体深度值
        elif depths_unidirectional == "medium":
            self.depths = [3, 4, 4, 4, 4]
        elif depths_unidirectional == "large":
            self.depths = [3, 4, 8, 8, 8]

        encoder = ["THPAEncFR3", "THPAEncFR3", "THPAEncFR3", "THPAEncFR3", "THPAEncFR3"]
        downcoder = "NormDownsample"

        self.stem = nn.Conv3d(init_channels, n_channels, kernel_size=1)

        # 使用self.depths创建模块（确保深度值正确）
        self.repr_block_0 = block_creator(encoder[0], self.depths[0], n_channels)
        self.dwn_block_0 = block_creator(downcoder, 1, n_channels, n_channels * 2)

        self.repr_block_1 = block_creator(encoder[1], self.depths[1], n_channels * 2)
        self.dwn_block_1 = block_creator(downcoder, 1, n_channels * 2, n_channels * 4)

        self.repr_block_2 = block_creator(encoder[2], self.depths[2], n_channels * 4)
        self.dwn_block_2 = block_creator(downcoder, 1, n_channels * 4, n_channels * 8)

        self.repr_block_3 = block_creator(encoder[3], self.depths[3], n_channels * 8)
        self.dwn_block_3 = block_creator(downcoder, 1, n_channels * 8, n_channels * 16)

        self.emb_block = block_creator(encoder[4], self.depths[4], n_channels * 16)

        if self.outside_block_checkpointing:
            self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

    def iterative_checkpoint(self, sequential_block, x):
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor, use_reentrant=True)
        return x

    def forward(self, input: Tensor):
        if self.outside_block_checkpointing:
            pass
        else:
            input = self.stem(input)
            skips = []
            repr0 = self.repr_block_0(input)
            dwn0 = self.dwn_block_0(repr0)
            skips.append(repr0)
            del repr0

            repr1 = self.repr_block_1(dwn0)
            dwn1 = self.dwn_block_1(repr1)
            skips.append(repr1)
            del repr1

            repr2 = self.repr_block_2(dwn1)
            dwn2 = self.dwn_block_2(repr2)
            skips.append(repr2)
            del repr2

            repr3 = self.repr_block_3(dwn2)
            dwn3 = self.dwn_block_3(repr3)
            skips.append(repr3)
            del repr3

            hidden = self.emb_block(dwn3)

            return hidden, tuple(skips)


class JCMNetv8Dec(nn.Module):
    def __init__(self,
                 init_channels=4,
                 n_channels=32,
                 class_nums=4,
                 checkpoint_style="",
                 expr=2,
                 depths_unidirectional=None,
                 ):
        super(JCMNetv8Dec, self).__init__()

        # 核心修复：存储深度参数
        self.depths_unidirectional = depths_unidirectional
        if depths_unidirectional is None:
            raise NotImplementedError
        elif depths_unidirectional == "small":
            self.depths = [1, 1, 2, 2, 2]
        elif depths_unidirectional == "medium":
            self.depths = [3, 4, 4, 4, 4]
        elif depths_unidirectional == "large":
            self.depths = [3, 4, 8, 8, 8]

        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        else:
            self.outside_block_checkpointing = False

        decoder = ["Learnable_Res_Skip_UpRepr4", "Learnable_Res_Skip_UpRepr4",
                   "Learnable_Res_Skip_UpRepr4", "Learnable_Res_Skip_UpRepr4"]

        self.repr_block_up_3 = block_creator(decoder[3], self.depths[3], n_channels * 16, n_channels * 8)
        self.repr_block_up_2 = block_creator(decoder[2], self.depths[2], n_channels * 8, n_channels * 4)
        self.repr_block_up_1 = block_creator(decoder[1], self.depths[1], n_channels * 4, n_channels * 2)
        self.repr_block_up_0 = block_creator(decoder[0], self.depths[0], n_channels * 2, n_channels)

        if self.outside_block_checkpointing:
            self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

    def iterative_checkpoint(self, sequential_block, x):
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor, use_reentrant=True)
        return x

    def forward(self, hidden, skips):
        if self.outside_block_checkpointing:
            pass
        else:
            dec = self.repr_block_up_3((hidden, skips[3]))
            dec = self.repr_block_up_2((dec, skips[2]))
            dec = self.repr_block_up_1((dec, skips[1]))
            dec = self.repr_block_up_0((dec, skips[0]))

            return dec


class NormalU_Net(nn.Module):
    def __init__(self,
                 init_channels=4,
                 n_channels=24,
                 class_nums=4,
                 checkpoint_style="",
                 expr=2,
                 depths_unidirectional=None,
                 ):
        super().__init__()
        self.depths_unidirectional = depths_unidirectional  # 存储深度配置名称
        args_list = [init_channels,
                     n_channels,
                     class_nums,
                     checkpoint_style,
                     expr,
                     depths_unidirectional]
        self.ParallelU_Net_enc_m = JCMNetv8Enc(*args_list)
        self.ParallelU_Net_dec_m = JCMNetv8Dec(*args_list)

        self.norm = nn.GroupNorm(n_channels, n_channels)
        self.proj = MlpChannel(n_channels, expr, class_nums)

    def forward(self, input, dummy_tensor=None):
        hidden_m, skips_m = self.ParallelU_Net_enc_m(input)
        out = self.ParallelU_Net_dec_m(hidden_m, skips_m)
        out = self.proj(self.norm(out))
        return out