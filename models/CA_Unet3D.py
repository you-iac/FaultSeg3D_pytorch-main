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
        self.F_h = nn.Conv3d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.sigmoid_l = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
        self.sigmoid_h = nn.Sigmoid()

    def forward(self, x):
        # 输入尺寸: (batch_size, channel, L, W, H)
        _, _, H, L, W = x.size()

        # 沿 H 维度求均值（保留 L 和 W）
        x_l = torch.mean(x, dim=4, keepdim=True).permute(0, 1, 2, 4, 3)# 形状: (batch, C, H, 1, L)
        x_w = torch.mean(x, dim=3, keepdim=True)                             # 形状: (batch, C, H, 1, W)
        x_h = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2, 4)# 形状: (batch, C, L, 1, W)

        # 拼接时忽略 H 维度（因为操作仅针对 L 和 W）
        x_cat = torch.cat([x_l, x_w,x_h], dim=4) #形状:(batch, C, H, 1, W+L+W=384)
        # 降维 + 激活
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1x1(x_cat)))

        # 拆分回 L 和 W 部分
        x_cat_conv_split_l, x_cat_conv_split_w,x_cat_conv_split_h = x_cat_conv_relu.split([L, W, W],dim=4)

        x_cat_conv_split_l = x_cat_conv_split_l.permute(0, 1, 2, 4, 3)  # 恢复形状: (batch, C, H, L, 1)
        x_cat_conv_split_w = x_cat_conv_split_w                         #          (batch, C, H, 1, W)
        x_cat_conv_split_h = x_cat_conv_split_h.permute(0, 1, 3, 2, 4)  # 恢复形状: (batch, C, 1, L, W)
        # 生成注意力权重
        s_l = self.sigmoid_l(self.F_l(x_cat_conv_split_l))  # 形状: (batch, C, H, 1, L)
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))  # 形状: (batch, C, H, 1, W)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h))  # 形状: (batch, C, H, 1, W)

        # 扩展权重并应用
        out = x * s_l.expand_as(x) * s_w.expand_as(x) * s_h.expand_as(x)
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
        #加入CA模块
        self.CA_Block_3D = CA_Block_3D(out_channels)

    def forward(self, x):
        T = self.double_conv(x)
        return self.CA_Block_3D(T)


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

# ================================================================
# Total params: 1,466,142
# Trainable params: 1,466,142
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 6669.72
# Params size (MB): 5.59
# Estimated Total Size (MB): 6683.31
# ----------------------------------------------------------------

# ================================================================
# Total params: 1,467,838
# Trainable params: 1,467,838
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 6685.58
# Params size (MB): 5.60
# Estimated Total Size (MB): 6699.18
# ----------------------------------------------------------------

