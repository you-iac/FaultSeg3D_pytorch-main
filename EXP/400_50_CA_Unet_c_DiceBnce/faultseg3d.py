import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

#CA模块加在跳跃连接

#   x_00 -----------------> 16 128^3 ----------------> x_01
#
#           x_10 ---------> 32  64^3 --------> x_11
#
#                   x_20 -> 64  32^3 -> x_21
#
#                           x_30
#
#
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
        self.CABlock = CA_Block_3D(out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
        #加入注意力模块
        x2 = self.CABlock(x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


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
# Total params: 1,462,368
# Trainable params: 1,462,368
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 6320.97
# Params size (MB): 5.58
# Estimated Total Size (MB): 6334.55
# ----------------------------------------------------------------

