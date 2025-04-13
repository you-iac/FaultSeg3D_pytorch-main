import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

class CA_Block_3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block_3D, self).__init__()

        self.conv_1x1x1 = nn.Conv3d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            bias=False
        )

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(channel // reduction)

        # 注意：输出通道数仍为 channel，因为要对原始通道加权
        self.F_l = nn.Conv3d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.F_w = nn.Conv3d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            bias=False
        )

        self.sigmoid_l = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        # 输入尺寸: (batch_size, channel, L, W, H)
        _, _, L, W, H = x.size()

        # 沿 H 维度求均值（保留 L 和 W）
        x_l = torch.mean(x, dim=4, keepdim=True)      # 形状: (batch, C, L, W, 1)
        x_w = torch.mean(x, dim=3, keepdim=True)      # 形状: (batch, C, L, 1, H)

        # 拼接时忽略 H 维度（因为操作仅针对 L 和 W）
        x_cat = torch.cat([x_l, x_w], dim=3)          # 形状: (batch, C, L, W + 1, 1)
        x_cat = x_cat.permute(0, 1, 2, 4, 3)          # 形状: (batch, C, L, 1, W + 1)

        # 降维 + 激活
        x_cat_conv = self.conv_1x1x1(x_cat)           # 形状: (batch, C//reduction, L, 1, W + 1)
        x_cat_conv_relu = self.relu(self.bn(x_cat_conv))

        # 拆分回 L 和 W 部分
        x_cat_conv_split_l, x_cat_conv_split_w = torch.split(
            x_cat_conv_relu,
            [W, 1],  # 注意拆分顺序和维度
            dim=4
        )
        x_cat_conv_split_l = x_cat_conv_split_l.permute(0, 1, 2, 4, 3)  # 恢复形状: (batch, C//reduction, L, W, 1)
        x_cat_conv_split_w = x_cat_conv_split_w.permute(0, 1, 2, 4, 3)  # 恢复形状: (batch, C//reduction, L, 1, 1)

        # 生成注意力权重
        s_l = self.sigmoid_l(self.F_l(x_cat_conv_split_l))  # 形状: (batch, C, L, W, 1)
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))  # 形状: (batch, C, L, 1, 1)

        # 扩展权重并应用
        out = x * s_l.expand_as(x) * s_w.expand_as(x)
        return out

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

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv3d-1    [-1, 16, 128, 128, 128]             448
#        BatchNorm3d-2    [-1, 16, 128, 128, 128]              32
#               ReLU-3    [-1, 16, 128, 128, 128]               0
#             Conv3d-4    [-1, 16, 128, 128, 128]           6,928
#        BatchNorm3d-5    [-1, 16, 128, 128, 128]              32
#               ReLU-6    [-1, 16, 128, 128, 128]               0
#         DoubleConv-7    [-1, 16, 128, 128, 128]               0
#          MaxPool3d-8       [-1, 16, 64, 64, 64]               0
#             Conv3d-9       [-1, 32, 64, 64, 64]          13,856
#       BatchNorm3d-10       [-1, 32, 64, 64, 64]              64
#              ReLU-11       [-1, 32, 64, 64, 64]               0
#            Conv3d-12       [-1, 32, 64, 64, 64]          27,680
#       BatchNorm3d-13       [-1, 32, 64, 64, 64]              64
#              ReLU-14       [-1, 32, 64, 64, 64]               0
#        DoubleConv-15       [-1, 32, 64, 64, 64]               0
#              Down-16       [-1, 32, 64, 64, 64]               0
#         MaxPool3d-17       [-1, 32, 32, 32, 32]               0
#            Conv3d-18       [-1, 64, 32, 32, 32]          55,360
#       BatchNorm3d-19       [-1, 64, 32, 32, 32]             128
#              ReLU-20       [-1, 64, 32, 32, 32]               0
#            Conv3d-21       [-1, 64, 32, 32, 32]         110,656
#       BatchNorm3d-22       [-1, 64, 32, 32, 32]             128
#              ReLU-23       [-1, 64, 32, 32, 32]               0
#        DoubleConv-24       [-1, 64, 32, 32, 32]               0
#              Down-25       [-1, 64, 32, 32, 32]               0
#         MaxPool3d-26       [-1, 64, 16, 16, 16]               0
#            Conv3d-27      [-1, 128, 16, 16, 16]         221,312
#       BatchNorm3d-28      [-1, 128, 16, 16, 16]             256
#              ReLU-29      [-1, 128, 16, 16, 16]               0
#            Conv3d-30      [-1, 128, 16, 16, 16]         442,496
#       BatchNorm3d-31      [-1, 128, 16, 16, 16]             256
#              ReLU-32      [-1, 128, 16, 16, 16]               0
#        DoubleConv-33      [-1, 128, 16, 16, 16]               0
#              Down-34      [-1, 128, 16, 16, 16]               0
#          Upsample-35      [-1, 128, 32, 32, 32]               0
#            Conv3d-36       [-1, 64, 32, 32, 32]         331,840
#       BatchNorm3d-37       [-1, 64, 32, 32, 32]             128
#              ReLU-38       [-1, 64, 32, 32, 32]               0
#            Conv3d-39       [-1, 64, 32, 32, 32]         110,656
#       BatchNorm3d-40       [-1, 64, 32, 32, 32]             128
#              ReLU-41       [-1, 64, 32, 32, 32]               0
#        DoubleConv-42       [-1, 64, 32, 32, 32]               0
#                Up-43       [-1, 64, 32, 32, 32]               0
#          Upsample-44       [-1, 64, 64, 64, 64]               0
#            Conv3d-45       [-1, 32, 64, 64, 64]          82,976
#       BatchNorm3d-46       [-1, 32, 64, 64, 64]              64
#              ReLU-47       [-1, 32, 64, 64, 64]               0
#            Conv3d-48       [-1, 32, 64, 64, 64]          27,680
#       BatchNorm3d-49       [-1, 32, 64, 64, 64]              64
#              ReLU-50       [-1, 32, 64, 64, 64]               0
#        DoubleConv-51       [-1, 32, 64, 64, 64]               0
#                Up-52       [-1, 32, 64, 64, 64]               0
#          Upsample-53    [-1, 32, 128, 128, 128]               0
#            Conv3d-54    [-1, 16, 128, 128, 128]          20,752
#       BatchNorm3d-55    [-1, 16, 128, 128, 128]              32
#              ReLU-56    [-1, 16, 128, 128, 128]               0
#            Conv3d-57    [-1, 16, 128, 128, 128]           6,928
#       BatchNorm3d-58    [-1, 16, 128, 128, 128]              32
#              ReLU-59    [-1, 16, 128, 128, 128]               0
#        DoubleConv-60    [-1, 16, 128, 128, 128]               0
#                Up-61    [-1, 16, 128, 128, 128]               0
#            Conv3d-62     [-1, 2, 128, 128, 128]              34
#           OutConv-63     [-1, 2, 128, 128, 128]               0
#           Softmax-64     [-1, 2, 128, 128, 128]               0
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