import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

# 导入 tvdcn 包中的 PackedDeformConv3d
from tvdcn.ops import PackedDeformConv3d


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, use_deformable=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if use_deformable:
            # 使用 tvdcn.ops.PackedDeformConv3d 替换 nn.Conv3d
            self.double_conv = nn.Sequential(
                PackedDeformConv3d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                PackedDeformConv3d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
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
    def __init__(self, in_channels, out_channels, use_deformable=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, use_deformable=use_deformable)
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

        # 第一和第二层使用可变形卷积
        self.inc = DoubleConv(n_channels, 16, use_deformable=True)
        self.down1 = Down(16, 32, use_deformable=True)

        # 剩下的层使用普通卷积
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
# Total params: 1,603,165
# Trainable params: 1,603,165
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 8878.00
# Params size (MB): 6.12
# Estimated Total Size (MB): 8892.12
# ----------------------------------------------------------------