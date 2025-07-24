import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

#conda pip install einops


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.swin_skip = SwinSkipConnection(out_channels)
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
        x2 = self.swin_skip(x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

import torch.nn as nn
from monai.networks.blocks import PatchEmbed
from monai.networks.nets.swin_unetr import BasicLayer


class SwinSkipConnection(nn.Module):
    """
    通用SwinTransformer3D跳跃连接模块。
    输入输出形状均为 (B, C, D, H, W)，空间尺寸与通道数保持一致。
    """

    def __init__(
            self,
            in_channels: int,
            window_size=(4, 8, 8),
            depth: int = 2,
            num_heads: int = 4,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        # 1x1x1 patch embedding 保留输入空间大小
        self.patch_embed = PatchEmbed(
            patch_size=(1, 1, 1),
            in_chans=in_channels,
            embed_dim=in_channels,
            norm_layer=None,  # 不使用归一化
            spatial_dims=3
        )
        # Swin Transformer 基础层，不做降采样，以保持空间尺寸
        self.swin_layer = BasicLayer(
            dim=in_channels,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            drop_path=[0.0] * depth,  # 可使用线性递增的 drop_path
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=dropout,
            attn_drop=attn_dropout,
            norm_layer=nn.LayerNorm,
            downsample=None,  # 关闭下采样，保持尺寸不变:contentReference[oaicite:10]{index=10}
        )

    def forward(self, x):
        # 输入验证：应为 (B, C, D, H, W) 的5维张量
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}"
            )
        # Patch嵌入（1x1卷积）：输出形状 (B, C, D, H, W):contentReference[oaicite:11]{index=11}
        x = self.patch_embed(x)
        # SwinTransformer BasicLayer：输出形状 (B, C, D, H, W):contentReference[oaicite:12]{index=12}
        x = self.swin_layer(x)
        return x

class FaultSeg3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FaultSeg3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        self.up2 = Up(192, 64)
        self.up3 = Up(96, 32)
        self.up4 = Up(48, 16)
        self.outc = OutConv(16, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder部分
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # decoder部分
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        outputs = self.softmax(logits)
        return outputs


if __name__ == '__main__':
    # 查看网络参数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))

# ================================================================
# Total params: 1,461,010
# Trainable params: 1,461,010
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 5962.00
# Params size (MB): 5.57
# Estimated Total Size (MB): 5975.57
# ----------------------------------------------------------------