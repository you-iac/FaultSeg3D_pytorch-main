import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import sys
# 导入 tvdcn 包中的 PackedDeformConv3d
import torch

from tvdcn.ops import deform_conv3d


class DoubleDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleDeformConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        kernel_size = 3
        padding = 1
        num_points = kernel_size * kernel_size * kernel_size  # 3*3*3 = 27
        offset_mask_channels = 4 * num_points  # 3*K^3 (offset) + 1*K^3 (mask) = 108

        # --- 第一层可变形卷积的组件：in_channels -> mid_channels ---

        # 1. 预测 Offset 和 Mask 的辅助 Conv3D
        self.offset_mask_conv1 = nn.Conv3d(
            in_channels, offset_mask_channels, kernel_size=kernel_size, padding=padding
        )

        # 2. 学习可变形卷积的权重 (作为 DeformConv3d 的 weight 参数)
        # 权重形状: (Out_C, In_C, K, K, K) = (mid_channels, in_channels, 3, 3, 3)
        self.weight1 = nn.Parameter(torch.Tensor(mid_channels, in_channels,
                                                 kernel_size, kernel_size, kernel_size))
        self.bias1 = nn.Parameter(torch.Tensor(mid_channels))

        nn.init.kaiming_uniform_(self.weight1, a=5)
        self.bias1.data.zero_()

        self.bn1 = nn.BatchNorm3d(mid_channels)

        # --- 第二层可变形卷积的组件：mid_channels -> out_channels ---

        # 1. 预测 Offset 和 Mask 的辅助 Conv3D
        # 输入通道: mid_channels (上一层的输出)
        self.offset_mask_conv2 = nn.Conv3d(
            mid_channels, offset_mask_channels, kernel_size=kernel_size, padding=padding
        )

        # 2. 学习可变形卷积的权重 (作为 DeformConv3d 的 weight 参数)
        # 权重形状: (Out_C, In_C, K, K, K) = (out_channels, mid_channels, 3, 3, 3)
        self.weight2 = nn.Parameter(torch.Tensor(out_channels, mid_channels,
                                                 kernel_size, kernel_size, kernel_size))
        self.bias2 = nn.Parameter(torch.Tensor(out_channels))

        nn.init.kaiming_uniform_(self.weight2, a=5)
        self.bias2.data.zero_()

        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        num_points = 27

        # --- 第一层可变形卷积 ---

        # 1. 生成 offset (3*K^3) 和 mask (1*K^3)
        offset_mask1 = self.offset_mask_conv1(x)
        offset1 = offset_mask1[:, :3 * num_points, :, :, :]
        mask1 = torch.sigmoid(offset_mask1[:, 3 * num_points:, :, :, :])

        # 2. 执行可变形卷积
        x = deform_conv3d(
            x, self.weight1, offset1, mask1, self.bias1,
            stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), groups=1
        )

        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        # --- 第二层可变形卷积 ---

        # 1. 生成 offset 和 mask
        offset_mask2 = self.offset_mask_conv2(x)
        offset2 = offset_mask2[:, :3 * num_points, :, :, :]
        mask2 = torch.sigmoid(offset_mask2[:, 3 * num_points:, :, :, :])

        # 2. 执行可变形卷积
        x = deform_conv3d(
            x, self.weight2, offset2, mask2, self.bias2,
            stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), groups=1
        )

        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        return x


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, use_deformable=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if use_deformable:
            # 使用 tvdcn.ops.PackedDeformConv3d 替换 nn.Conv3d
            self.double_conv = DoubleDeformConv(in_channels, out_channels, mid_channels)
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
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32, use_deformable=True)

        # 剩下的层使用普通卷积
        self.down2 = Down(32, 64, use_deformable=True)
        self.down3 = Down(64, 128, use_deformable=True)

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
# Total params: 1,559,658
# Trainable params: 1,559,658
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 6202.00
# Params size (MB): 5.95
# Estimated Total Size (MB): 6215.95
# ----------------------------------------------------------------