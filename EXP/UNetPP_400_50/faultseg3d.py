import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F


#   x_00
#
#           x_10
#
#                   x_20
#
#                         x_30
#
#                                 x_40


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


class Merge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    #低层在前，高层在后，中间若干
    def forward(self, x1, x2, items):
        # 上采样操作
        x1 = self.up(x1)

        # diffZ = x2.size()[2] - x1.size()[2]
        # diffY = x2.size()[3] - x1.size()[3]
        # diffX = x2.size()[4] - x1.size()[4]
        # x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                             diffY // 2, diffY - diffY // 2,
        #                             diffZ // 2, diffZ - diffZ // 2])
        # 拼接所有张量（x2, x1, 以及 items 中的张量）

        x = torch.cat([x2, x1, *items], dim=1)
        return self.conv(x)


#用于中间融合
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

        self.Merge01 = Merge(16+32,16)
        self.Merge02 = Merge(16 + 16 + 32, 16)
        self.Merge11 = Merge(32 + 64, 32)

        self.up3 = Merge(128+64, 64)
        self.up2 = Merge(32+32+64, 32)
        self.up1 = Merge(16+16+16+32, 16)
        self.outc = OutConv(16, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder部分
        x00 = self.inc(x)
        x10 = self.down1(x00)
        x20 = self.down2(x10)
        x30 = self.down3(x20)


        x01 = self.Merge01(x10, x00, [] )
        x11 = self.Merge11(x20, x10, [])
        x02 = self.Merge02(x11, x00, [x01])

        # decoder部分
        x21 = self.up3(x30, x20, [])
        x13 = self.up2(x21, x10, [x11])
        x03 = self.up1(x13, x00, [x01, x02])
        logits = self.outc(x03)
        outputs = self.softmax(logits)
        return outputs


if __name__ == '__main__':
    # 查看网络参数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))


# D:\Users\28695\anaconda3\envs\Fault\python.exe D:\WorkStation\Code\Python\FaultSeg3D_pytorch-main\models\Unet++3D.py
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
#          Upsample-35    [-1, 32, 128, 128, 128]               0
#            Conv3d-36    [-1, 16, 128, 128, 128]          20,752
#       BatchNorm3d-37    [-1, 16, 128, 128, 128]              32
#              ReLU-38    [-1, 16, 128, 128, 128]               0
#            Conv3d-39    [-1, 16, 128, 128, 128]           6,928
#       BatchNorm3d-40    [-1, 16, 128, 128, 128]              32
#              ReLU-41    [-1, 16, 128, 128, 128]               0
#        DoubleConv-42    [-1, 16, 128, 128, 128]               0
#             Merge-43    [-1, 16, 128, 128, 128]               0
#          Upsample-44       [-1, 64, 64, 64, 64]               0
#            Conv3d-45       [-1, 32, 64, 64, 64]          82,976
#       BatchNorm3d-46       [-1, 32, 64, 64, 64]              64
#              ReLU-47       [-1, 32, 64, 64, 64]               0
#            Conv3d-48       [-1, 32, 64, 64, 64]          27,680
#       BatchNorm3d-49       [-1, 32, 64, 64, 64]              64
#              ReLU-50       [-1, 32, 64, 64, 64]               0
#        DoubleConv-51       [-1, 32, 64, 64, 64]               0
#             Merge-52       [-1, 32, 64, 64, 64]               0
#          Upsample-53    [-1, 32, 128, 128, 128]               0
#            Conv3d-54    [-1, 16, 128, 128, 128]          27,664
#       BatchNorm3d-55    [-1, 16, 128, 128, 128]              32
#              ReLU-56    [-1, 16, 128, 128, 128]               0
#            Conv3d-57    [-1, 16, 128, 128, 128]           6,928
#       BatchNorm3d-58    [-1, 16, 128, 128, 128]              32
#              ReLU-59    [-1, 16, 128, 128, 128]               0
#        DoubleConv-60    [-1, 16, 128, 128, 128]               0
#             Merge-61    [-1, 16, 128, 128, 128]               0
#          Upsample-62      [-1, 128, 32, 32, 32]               0
#            Conv3d-63       [-1, 64, 32, 32, 32]         331,840
#       BatchNorm3d-64       [-1, 64, 32, 32, 32]             128
#              ReLU-65       [-1, 64, 32, 32, 32]               0
#            Conv3d-66       [-1, 64, 32, 32, 32]         110,656
#       BatchNorm3d-67       [-1, 64, 32, 32, 32]             128
#              ReLU-68       [-1, 64, 32, 32, 32]               0
#        DoubleConv-69       [-1, 64, 32, 32, 32]               0
#             Merge-70       [-1, 64, 32, 32, 32]               0
#          Upsample-71       [-1, 64, 64, 64, 64]               0
#            Conv3d-72       [-1, 32, 64, 64, 64]         110,624
#       BatchNorm3d-73       [-1, 32, 64, 64, 64]              64
#              ReLU-74       [-1, 32, 64, 64, 64]               0
#            Conv3d-75       [-1, 32, 64, 64, 64]          27,680
#       BatchNorm3d-76       [-1, 32, 64, 64, 64]              64
#              ReLU-77       [-1, 32, 64, 64, 64]               0
#        DoubleConv-78       [-1, 32, 64, 64, 64]               0
#             Merge-79       [-1, 32, 64, 64, 64]               0
#          Upsample-80    [-1, 32, 128, 128, 128]               0
#            Conv3d-81    [-1, 16, 128, 128, 128]          34,576
#       BatchNorm3d-82    [-1, 16, 128, 128, 128]              32
#              ReLU-83    [-1, 16, 128, 128, 128]               0
#            Conv3d-84    [-1, 16, 128, 128, 128]           6,928
#       BatchNorm3d-85    [-1, 16, 128, 128, 128]              32
#              ReLU-86    [-1, 16, 128, 128, 128]               0
#        DoubleConv-87    [-1, 16, 128, 128, 128]               0
#             Merge-88    [-1, 16, 128, 128, 128]               0
#            Conv3d-89     [-1, 2, 128, 128, 128]              34
#           OutConv-90     [-1, 2, 128, 128, 128]               0
#           Softmax-91     [-1, 2, 128, 128, 128]               0
# ================================================================
# Total params: 1,675,666
# Trainable params: 1,675,666
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 11722.00
# Params size (MB): 6.39
# Estimated Total Size (MB): 11736.39
# ----------------------------------------------------------------