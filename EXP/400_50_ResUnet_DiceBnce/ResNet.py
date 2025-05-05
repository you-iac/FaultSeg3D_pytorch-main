import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # 跳跃连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.conv = ResidualBlock(in_channels, out_channels)

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

        self.inc = ResidualBlock(n_channels, 16)
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
#             Conv3d-1    [-1, 16, 128, 128, 128]             432
#        BatchNorm3d-2    [-1, 16, 128, 128, 128]              32
#               ReLU-3    [-1, 16, 128, 128, 128]               0
#             Conv3d-4    [-1, 16, 128, 128, 128]           6,912
#        BatchNorm3d-5    [-1, 16, 128, 128, 128]              32
#             Conv3d-6    [-1, 16, 128, 128, 128]              16
#        BatchNorm3d-7    [-1, 16, 128, 128, 128]              32
#               ReLU-8    [-1, 16, 128, 128, 128]               0
#      ResidualBlock-9    [-1, 16, 128, 128, 128]               0
#         MaxPool3d-10       [-1, 16, 64, 64, 64]               0
#            Conv3d-11       [-1, 32, 64, 64, 64]          13,824
#       BatchNorm3d-12       [-1, 32, 64, 64, 64]              64
#              ReLU-13       [-1, 32, 64, 64, 64]               0
#            Conv3d-14       [-1, 32, 64, 64, 64]          27,648
#       BatchNorm3d-15       [-1, 32, 64, 64, 64]              64
#            Conv3d-16       [-1, 32, 64, 64, 64]             512
#       BatchNorm3d-17       [-1, 32, 64, 64, 64]              64
#              ReLU-18       [-1, 32, 64, 64, 64]               0
#     ResidualBlock-19       [-1, 32, 64, 64, 64]               0
#              Down-20       [-1, 32, 64, 64, 64]               0
#         MaxPool3d-21       [-1, 32, 32, 32, 32]               0
#            Conv3d-22       [-1, 64, 32, 32, 32]          55,296
#       BatchNorm3d-23       [-1, 64, 32, 32, 32]             128
#              ReLU-24       [-1, 64, 32, 32, 32]               0
#            Conv3d-25       [-1, 64, 32, 32, 32]         110,592
#       BatchNorm3d-26       [-1, 64, 32, 32, 32]             128
#            Conv3d-27       [-1, 64, 32, 32, 32]           2,048
#       BatchNorm3d-28       [-1, 64, 32, 32, 32]             128
#              ReLU-29       [-1, 64, 32, 32, 32]               0
#     ResidualBlock-30       [-1, 64, 32, 32, 32]               0
#              Down-31       [-1, 64, 32, 32, 32]               0
#         MaxPool3d-32       [-1, 64, 16, 16, 16]               0
#            Conv3d-33      [-1, 128, 16, 16, 16]         221,184
#       BatchNorm3d-34      [-1, 128, 16, 16, 16]             256
#              ReLU-35      [-1, 128, 16, 16, 16]               0
#            Conv3d-36      [-1, 128, 16, 16, 16]         442,368
#       BatchNorm3d-37      [-1, 128, 16, 16, 16]             256
#            Conv3d-38      [-1, 128, 16, 16, 16]           8,192
#       BatchNorm3d-39      [-1, 128, 16, 16, 16]             256
#              ReLU-40      [-1, 128, 16, 16, 16]               0
#     ResidualBlock-41      [-1, 128, 16, 16, 16]               0
#              Down-42      [-1, 128, 16, 16, 16]               0
#          Upsample-43      [-1, 128, 32, 32, 32]               0
#            Conv3d-44       [-1, 64, 32, 32, 32]         331,776
#       BatchNorm3d-45       [-1, 64, 32, 32, 32]             128
#              ReLU-46       [-1, 64, 32, 32, 32]               0
#            Conv3d-47       [-1, 64, 32, 32, 32]         110,592
#       BatchNorm3d-48       [-1, 64, 32, 32, 32]             128
#            Conv3d-49       [-1, 64, 32, 32, 32]          12,288
#       BatchNorm3d-50       [-1, 64, 32, 32, 32]             128
#              ReLU-51       [-1, 64, 32, 32, 32]               0
#     ResidualBlock-52       [-1, 64, 32, 32, 32]               0
#                Up-53       [-1, 64, 32, 32, 32]               0
#          Upsample-54       [-1, 64, 64, 64, 64]               0
#            Conv3d-55       [-1, 32, 64, 64, 64]          82,944
#       BatchNorm3d-56       [-1, 32, 64, 64, 64]              64
#              ReLU-57       [-1, 32, 64, 64, 64]               0
#            Conv3d-58       [-1, 32, 64, 64, 64]          27,648
#       BatchNorm3d-59       [-1, 32, 64, 64, 64]              64
#            Conv3d-60       [-1, 32, 64, 64, 64]           3,072
#       BatchNorm3d-61       [-1, 32, 64, 64, 64]              64
#              ReLU-62       [-1, 32, 64, 64, 64]               0
#     ResidualBlock-63       [-1, 32, 64, 64, 64]               0
#                Up-64       [-1, 32, 64, 64, 64]               0
#          Upsample-65    [-1, 32, 128, 128, 128]               0
#            Conv3d-66    [-1, 16, 128, 128, 128]          20,736
#       BatchNorm3d-67    [-1, 16, 128, 128, 128]              32
#              ReLU-68    [-1, 16, 128, 128, 128]               0
#            Conv3d-69    [-1, 16, 128, 128, 128]           6,912
#       BatchNorm3d-70    [-1, 16, 128, 128, 128]              32
#            Conv3d-71    [-1, 16, 128, 128, 128]             768
#       BatchNorm3d-72    [-1, 16, 128, 128, 128]              32
#              ReLU-73    [-1, 16, 128, 128, 128]               0
#     ResidualBlock-74    [-1, 16, 128, 128, 128]               0
#                Up-75    [-1, 16, 128, 128, 128]               0
#            Conv3d-76     [-1, 2, 128, 128, 128]              34
#           OutConv-77     [-1, 2, 128, 128, 128]               0
#           Softmax-78     [-1, 2, 128, 128, 128]               0
# ================================================================
# Total params: 1,487,906
# Trainable params: 1,487,906
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 7314.00
# Params size (MB): 5.68
# Estimated Total Size (MB): 7327.68
# ----------------------------------------------------------------