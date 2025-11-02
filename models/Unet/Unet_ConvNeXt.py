import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm3d(nn.Module):
    """3D Layer Normalization"""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    
    def forward(self, x):
        # x: (B, C, D, H, W)
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtBlock3D(nn.Module):
    """
    ConvNeXt Block for 3D
    Based on: https://github.com/facebookresearch/ConvNeXt
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        
        # 深度可分离卷积 (Depthwise conv)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # Layer Normalization
        self.norm = LayerNorm3d(dim)
        
        # 点卷积 (Pointwise conv) - 1x1x1 conv
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        
        # Scale factor for residual connection
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        input = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Layer Norm
        x = self.norm(x)
        
        # Pointwise conv (1x1x1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Scale and residual
        if self.gamma is not None:
            # gamma shape: (dim,) needs to be reshaped for broadcasting
            x = self.gamma[:, None, None, None] * x
        
        x = input + self.drop_path(x)
        
        return x


class ConvNeXtStem3D(nn.Module):
    """
    ConvNeXt Stem for 3D
    First stage of ConvNeXt
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用4x4x4 stride=4的卷积代替patch embedding
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=4),
            LayerNorm3d(out_channels)
        )
    
    def forward(self, x):
        return self.stem(x)


class DownSample3D(nn.Module):
    """
    Downsampling layer for 3D ConvNeXt
    """
    def __init__(self, dim, norm_layer=LayerNorm3d):
        super().__init__()
        self.downsample = nn.Sequential(
            LayerNorm3d(dim),
            nn.Conv3d(dim, 2 * dim, kernel_size=2, stride=2),
        )
    
    def forward(self, x):
        return self.downsample(x)


class ConvNeXtStage3D(nn.Module):
    """
    ConvNeXt Stage for 3D
    Multiple ConvNeXt blocks at a specific stage
    """
    def __init__(self, dim, depth, drop_path_rates=None, layer_scale_init_value=1e-6):
        super().__init__()
        if drop_path_rates is None:
            # 线性增加 drop_path rate
            drop_path_rates = [x.item() for x in torch.linspace(0, 0.1, depth)]
        
        self.blocks = nn.ModuleList([
            ConvNeXtBlock3D(
                dim=dim,
                drop_path=drop_path_rates[i],
                layer_scale_init_value=layer_scale_init_value
            )
            for i in range(depth)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DoubleConv(nn.Module):
    """Double Convolution with ConvNeXt blocks"""
    def __init__(self, in_channels, out_channels, mid_channels=None, num_blocks=2):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            LayerNorm3d(mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            LayerNorm3d(out_channels),
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downsampling with ConvNeXt style"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling with ConvNeXt style"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class FaultSeg3D(nn.Module):
    """
    ConvNeXt-based 3D UNet for Fault Segmentation
    
    Features:
    - LayerNorm instead of BatchNorm
    - GELU activation instead of ReLU
    - Depthwise separable convolution with 7x7x7 kernel
    - Modern design choices from ConvNeXt
    """
    def __init__(self, n_channels, n_classes):
        super(FaultSeg3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder - 使用ConvNeXt风格的块
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        
        # 在深层使用ConvNeXt blocks
        self.convnext_block1 = ConvNeXtBlock3D(32)
        self.convnext_block2 = ConvNeXtBlock3D(64)
        self.convnext_block3 = ConvNeXtBlock3D(128)
        
        # Decoder
        self.up2 = Up(192, 64)
        self.up3 = Up(96, 32)
        self.up4 = Up(48, 16)
        self.outc = OutConv(16, n_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)                    # 16 × 128³
        
        x2 = self.down1(x1)                 # 32 × 64³
        x2 = self.convnext_block1(x2)       # Apply ConvNeXt
        
        x3 = self.down2(x2)                 # 64 × 32³
        x3 = self.convnext_block2(x3)       # Apply ConvNeXt
        
        x4 = self.down3(x3)                 # 128 × 16³
        x4 = self.convnext_block3(x4)       # Apply ConvNeXt
        
        # Decoder
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        outputs = self.softmax(logits)
        
        return outputs


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))

# ================================================================
# Total params: 1,711,666
# Trainable params: 1,711,666
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 6378.00
# Params size (MB): 6.53
# Estimated Total Size (MB): 6392.53
# ----------------------------------------------------------------