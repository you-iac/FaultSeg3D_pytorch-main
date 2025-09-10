import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer

#注意力机制加到最下一层

#   x_00 -----------------> 16 128^3 ----------------> x_01
#
#           x_10 ---------> 32  64^3 --------> x_11
#
#                   x_20 -> 64  32^3 -> x_21
#
#                           x_30
#
#
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
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Transformer3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, num_layers=4):
        super(Transformer3D, self).__init__()

        # Transformer要求输入形状为 (seq_len, batch_size, feature_dim)
        # 所以我们需要把输入从 [B, C, L, H, W] 变为 [L*H*W, B, C] 这样才可以进入Transformer
        self.input_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.transformer = Transformer(d_model=out_channels,
                                       nhead=num_heads,
                                       num_encoder_layers=num_layers,
                                       num_decoder_layers=num_layers)

        self.output_proj = nn.Conv3d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x shape: [B, C, L, H, W]

        # 第一部分：卷积层
        B, C, L, H, W = x.shape
        x = self.input_proj(x)  # [B, out_channels, L, H, W]

        # Transformer需要的输入格式 (L*H*W, B, C)
        x = x.view(B, C, -1).permute(2, 0, 1)  # [L*H*W, B, C]

        # 使用Transformer进行处理
        x = self.transformer(x, x)  # 使用相同的输入作为encoder和decoder

        # 将输出变回 [B, out_channels, L, H, W]
        x = x.permute(1, 2, 0).view(B, C, L, H, W)

        # 使用卷积层得到最终输出
        x = self.output_proj(x)

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
        self.transformer = Transformer3D(128, 128, num_heads=4, num_layers=12)

    def forward(self, x):
        # encoder部分
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 自注意力机制
        x4 = self.transformer(x4)

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