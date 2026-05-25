import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

#加入上采样的多尺度融合

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


class FaultSeg3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FaultSeg3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- 关键修改点 1 ---
        # 即使外部传入 n_channels=1，模型内部第一个卷积层 self.inc 需要接收 2 个通道
        # 2 个通道 = 原始数据 (1) + 幅度谱 (1)
        self.inc = DoubleConv(n_channels + 1, 16)

        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        self.up2 = Up(192, 64)
        self.up3 = Up(96, 32)
        self.up4 = Up(48, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        # --- 关键修改点 2: 嵌入傅里叶处理 ---
        # 1. 计算 3D 傅里叶变换
        # x 尺寸: [Batch, 1, D, H, W]
        fft_x = torch.fft.fftn(x, dim=(-3, -2, -1))

        # 2. 获取幅度谱并进行 Log 缩放（压缩动态范围）
        mag = torch.abs(fft_x)
        mag = torch.log1p(mag)

        # 3. 归一化幅度谱至 [0, 1]
        # 注意：为了数值稳定，加一个极小值 eps
        mag_min = mag.view(mag.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1, 1)
        mag_max = mag.view(mag.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1, 1)
        mag = (mag - mag_min) / (mag_max - mag_min + 1e-8)

        # 4. 拼接原始输入和幅度谱 -> [Batch, 2, D, H, W]
        x_input = torch.cat([x, mag], dim=1)

        # 后续 Encoder 部分，使用拼接后的 x_input
        x1 = self.inc(x_input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder 部分保持不变
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    # 查看网络参数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))

# ================================================================
# Total params: 1,461,442
# Trainable params: 1,461,442
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 5962.00
# Params size (MB): 5.57
# Estimated Total Size (MB): 5975.57
# ----------------------------------------------------------------


