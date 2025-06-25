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
        _, _, H, L, W = x.size()

        # 沿 H 维度求均值（保留 L 和 W）
        x_l = torch.mean(x, dim=4, keepdim=True).permute(0, 1, 2, 4, 3)# 形状: (batch, C, H, 1, L)
        x_w = torch.mean(x, dim=3, keepdim=True)      # 形状: (batch, C, H, 1, W)


        # 拼接时忽略 H 维度（因为操作仅针对 L 和 W）
        x_cat = torch.cat([x_l, x_w], dim=4) #形状:(batch, C, H, 1, W+L)
        # 降维 + 激活
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1x1(x_cat)))

        # 拆分回 L 和 W 部分
        x_cat_conv_split_l, x_cat_conv_split_w = x_cat_conv_relu.split([L, W],dim=4)

        x_cat_conv_split_l = x_cat_conv_split_l.permute(0, 1, 2, 4, 3)  # 恢复形状: (batch, C, H, 1, L)
        x_cat_conv_split_w = x_cat_conv_split_w                         #          (batch, C, H, 1, W)

        # 生成注意力权重
        s_l = self.sigmoid_l(self.F_l(x_cat_conv_split_l))  # 形状: (batch, C, H, 1, L)
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))  # 形状: (batch, C, H, 1, W)

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
        # 加入CA模块
        self.CA_Block_3D = CA_Block_3D(out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)

        return  self.CA_Block_3D(self.conv(x))


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

# D:\Users\28695\anaconda3\envs\Fault\python.exe D:\WorkSpace\Code\Python\FaultSeg3D_pytorch-main\models\CA_Unet3D_A.py
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
#            Conv3d-43         [-1, 4, 32, 1, 64]             256
#       BatchNorm3d-44         [-1, 4, 32, 1, 64]               8
#              ReLU-45         [-1, 4, 32, 1, 64]               0
#            Conv3d-46        [-1, 64, 32, 32, 1]             256
#           Sigmoid-47        [-1, 64, 32, 32, 1]               0
#            Conv3d-48        [-1, 64, 32, 1, 32]             256
#           Sigmoid-49        [-1, 64, 32, 1, 32]               0
#       CA_Block_3D-50       [-1, 64, 32, 32, 32]               0
#                Up-51       [-1, 64, 32, 32, 32]               0
#          Upsample-52       [-1, 64, 64, 64, 64]               0
#            Conv3d-53       [-1, 32, 64, 64, 64]          82,976
#       BatchNorm3d-54       [-1, 32, 64, 64, 64]              64
#              ReLU-55       [-1, 32, 64, 64, 64]               0
#            Conv3d-56       [-1, 32, 64, 64, 64]          27,680
#       BatchNorm3d-57       [-1, 32, 64, 64, 64]              64
#              ReLU-58       [-1, 32, 64, 64, 64]               0
#        DoubleConv-59       [-1, 32, 64, 64, 64]               0
#            Conv3d-60        [-1, 2, 64, 1, 128]              64
#       BatchNorm3d-61        [-1, 2, 64, 1, 128]               4
#              ReLU-62        [-1, 2, 64, 1, 128]               0
#            Conv3d-63        [-1, 32, 64, 64, 1]              64
#           Sigmoid-64        [-1, 32, 64, 64, 1]               0
#            Conv3d-65        [-1, 32, 64, 1, 64]              64
#           Sigmoid-66        [-1, 32, 64, 1, 64]               0
#       CA_Block_3D-67       [-1, 32, 64, 64, 64]               0
#                Up-68       [-1, 32, 64, 64, 64]               0
#          Upsample-69    [-1, 32, 128, 128, 128]               0
#            Conv3d-70    [-1, 16, 128, 128, 128]          20,752
#       BatchNorm3d-71    [-1, 16, 128, 128, 128]              32
#              ReLU-72    [-1, 16, 128, 128, 128]               0
#            Conv3d-73    [-1, 16, 128, 128, 128]           6,928
#       BatchNorm3d-74    [-1, 16, 128, 128, 128]              32
#              ReLU-75    [-1, 16, 128, 128, 128]               0
#        DoubleConv-76    [-1, 16, 128, 128, 128]               0
#            Conv3d-77       [-1, 1, 128, 1, 256]              16
#       BatchNorm3d-78       [-1, 1, 128, 1, 256]               2
#              ReLU-79       [-1, 1, 128, 1, 256]               0
#            Conv3d-80      [-1, 16, 128, 128, 1]              16
#           Sigmoid-81      [-1, 16, 128, 128, 1]               0
#            Conv3d-82      [-1, 16, 128, 1, 128]              16
#           Sigmoid-83      [-1, 16, 128, 1, 128]               0
#       CA_Block_3D-84    [-1, 16, 128, 128, 128]               0
#                Up-85    [-1, 16, 128, 128, 128]               0
#            Conv3d-86     [-1, 2, 128, 128, 128]              34
#           OutConv-87     [-1, 2, 128, 128, 128]               0
#           Softmax-88     [-1, 2, 128, 128, 128]               0
# ================================================================
# Total params: 1,462,032
# Trainable params: 1,462,032
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 6313.31
# Params size (MB): 5.58
# Estimated Total Size (MB): 6326.89
# ----------------------------------------------------------------
#
# 进程已结束，退出代码为 0
