import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

class CA_Block_3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block_3D, self).__init__()  # 必须添加

        self.conv_1x1x1 = nn.Conv3d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            bias=False
        )

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(channel // reduction)  # 正确应该是 BatchNorm3d

        # 通道恢复（注意力）
        self.F_l = nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False)
        self.F_w = nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False)
        self.F_h = nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False)

        self.sigmoid_l = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
        self.sigmoid_h = nn.Sigmoid()

    def forward(self, x):
        B, C, H, L, W = x.size()

        # 三个方向的全局平均
        x_l = torch.mean(x, dim=4, keepdim=True).permute(0, 1, 2, 4, 3)  # -> (B, C, H, 1, L)
        x_w = torch.mean(x, dim=3, keepdim=True)                         # -> (B, C, H, 1, W)
        x_h = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2, 4)  # -> (B, C, 1, H, W)

        # 拼接：沿最后一维拼接 (L + W + H)
        x_cat = torch.cat([x_l, x_w, x_h], dim=4)  # -> (B, C, H, 1, L+W+H)

        # 降维 + BN + ReLU
        x_cat_conv = self.relu(self.bn(self.conv_1x1x1(x_cat)))  # -> (B, C//r, H, 1, L+W+H)

        # 拆分
        x_l_conv, x_w_conv, x_h_conv = torch.split(x_cat_conv, [L, W, H], dim=4)

        # 还原维度
        x_l_conv = x_l_conv.permute(0, 1, 2, 4, 3)  # -> (B, C//r, H, L, 1)
        x_h_conv = x_h_conv.permute(0, 1, 4, 3, 2)  # -> (B, C//r, H, 1, W)

        # 通道注意力生成
        s_l = self.sigmoid_l(self.F_l(x_l_conv))  # -> (B, C, H, L, 1)
        s_w = self.sigmoid_w(self.F_w(x_w_conv))  # -> (B, C, H, 1, W)
        s_h = self.sigmoid_h(self.F_h(x_h_conv))  # -> (B, C, H, L, W)

        # 扩展注意力维度
        s_l = s_l.expand_as(x)
        s_w = s_w.expand_as(x)
        s_h = s_h.expand_as(x)

        return x * s_l * s_w * s_h

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


# 改进后的Down模块：包含原始池化下采样 + shortcut卷积下采样 + 1x1x1融合
class DownWithShortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 正常下采样路径：MaxPool + DoubleConv
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

        # shortcut卷积路径：直接用stride=2卷积降分辨率
        self.shortcut_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.CA_Block_3D = CA_Block_3D(out_channels*2)
        # 1x1x1卷积进行融合
        self.fuse_conv = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)


    def forward(self, x):
        x_pool = self.pool_conv(x)
        x_short = self.shortcut_conv(x)
        x_cat = torch.cat([x_pool, x_short], dim=1)
        x = self.CA_Block_3D(x_cat)
        return self.fuse_conv(x)


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
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FaultSeg3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FaultSeg3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = DownWithShortcut(16, 32)
        self.down2 = DownWithShortcut(32, 64)
        self.down3 = DownWithShortcut(64, 128)

        self.up2 = Up(192, 64)   # 128 (x4) + 64 (x3)
        self.up3 = Up(96, 32)    # 64 (x3) + 32 (x2)
        self.up4 = Up(48, 16)    # 32 (x2) + 16 (x1)
        self.outc = OutConv(16, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)         # 16 x 128³
        x2 = self.down1(x1)      # 32 x 64³
        x3 = self.down2(x2)      # 64 x 32³
        x4 = self.down3(x3)      # 128 x 16³

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        outputs = self.softmax(logits)
        return outputs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))

# ================================================================
# Total params: 1,816,778
# Trainable params: 1,816,778
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 6488.97
# Params size (MB): 6.93
# Estimated Total Size (MB): 6503.90
# ----------------------------------------------------------------