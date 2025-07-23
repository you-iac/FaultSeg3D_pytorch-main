import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

#加入上采样的多尺度融合

#   x_00                                        x_01
#
#           x_10                         x_11
#
#                   x_20        x_21
#
#                         x_30
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


class EndMerge(nn.Module):
    """融合多层特征的模块（支持3D）"""
    def __init__(self, in_channels, out_channels, scale_factors=None):
        """
        Args:
            in_channels: 输入总通道数（所有特征图通道之和）
            out_channels: 输出通道数
            scale_factors: 各层相对于目标层的缩放因子列表
        """
        super().__init__()

        # 默认缩放因子（假设4层结构）
        self.scale_factors = scale_factors or [8, 4, 2, 1]

        # 创建对应的上采样层
        self.upsamplers = nn.ModuleList([
            nn.Upsample(scale_factor=sf, mode='trilinear', align_corners=True)
            for sf in self.scale_factors
        ])

        # 最后的卷积处理
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, *features):
        """
        融合多层特征图
        Args:
            features: 按分辨率从低到高排列的特征图列表
                      [最低分辨率特征, ..., 最高分辨率特征]
        Returns:
            融合后的特征图
        """
        # 确定目标尺寸（最高分辨率特征图的尺寸）
        target_size = features[-1].shape[2:]

        # 上采样所有特征图到目标尺寸
        upsampled_features = []
        for i, (feat, upsample) in enumerate(zip(features, self.upsamplers)):
            # 直接上采样到目标尺寸
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode='trilinear',
                    align_corners=True
                )
            upsampled_features.append(feat)

        # 沿通道维度拼接所有特征图
        x = torch.cat(upsampled_features, dim=1)

        # 通过卷积层融合特征
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
        self.outc = EndMerge(16+32+64+128, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder部分
        x00 = self.inc(x)
        x01 = self.down1(x00)
        x02 = self.down2(x01)
        x30 = self.down3(x02)

        # decoder部分
        x21 = self.up2(x30, x02)
        x11 = self.up3(x21, x01)
        x01 = self.up4(x11, x00)
        logits = self.outc(x30,x21,x11,x01)
        outputs = self.softmax(logits)
        return outputs


if __name__ == '__main__':
    # 查看网络参数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))

# ================================================================
# Total params: 1,474,056
# Trainable params: 1,474,056
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 6154.00
# Params size (MB): 5.62
# Estimated Total Size (MB): 6167.62
# ----------------------------------------------------------------