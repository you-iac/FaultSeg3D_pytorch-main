import torch
import torch.nn as nn
from torchsummary import summary
"""
MultiDirectionalSpatialAttention (多方向空间注意力)
功能: 在X、Y、Z三个方向分别计算空间注意力
优势: 针对细长断层特征，捕捉不同方向的空间信息
适用场景: 细长断层分割

"""
class MultiDirectionalSpatialAttention(nn.Module):
    """
    多方向空间注意力模块
    针对细长断层特征，在不同方向上进行注意力计算
    """

    def __init__(self, in_channels):
        super(MultiDirectionalSpatialAttention, self).__init__()
        # 不同方向的卷积核
        self.conv_x = nn.Conv3d(in_channels, 1, (1, 3, 3), padding=(0, 1, 1))
        self.conv_y = nn.Conv3d(in_channels, 1, (3, 1, 3), padding=(1, 0, 1))
        self.conv_z = nn.Conv3d(in_channels, 1, (3, 3, 1), padding=(1, 1, 0))
        self.fusion = nn.Conv3d(3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, D, H, W)
        Returns:
            out: 增强后的特征图 (B, C, D, H, W)
        """
        # 不同方向的注意力
        att_x = self.conv_x(x)
        att_y = self.conv_y(x)
        att_z = self.conv_z(x)

        # 融合多方向注意力
        multi_dir_att = torch.cat([att_x, att_y, att_z], dim=1)
        fused_att = self.fusion(multi_dir_att)
        att = self.sigmoid(fused_att)

        return x * att


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

# 方案C：编码器深层 + 解码器（完整型）
class FaultSeg3D(nn.Module):
    """
    在编码器深层和解码器应用多方向空间注意力的3D UNet网络
    完整型方案，效果最佳但计算量较大
    """

    def __init__(self, n_channels, n_classes):
        super(FaultSeg3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 编码器
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        # 在编码器深层添加多方向空间注意力
        self.multi_dir_att_encoder = MultiDirectionalSpatialAttention(128)

        # 解码器
        self.up2 = Up(192, 64)
        self.up3 = Up(96, 32)
        self.up4 = Up(48, 16)

        # 在解码器添加多方向空间注意力
        self.multi_dir_att_decoder1 = MultiDirectionalSpatialAttention(64)
        self.multi_dir_att_decoder2 = MultiDirectionalSpatialAttention(32)
        self.multi_dir_att_decoder3 = MultiDirectionalSpatialAttention(16)

        self.outc = OutConv(16, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder部分
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 在编码器深层应用多方向空间注意力
        x4 = self.multi_dir_att_encoder(x4)

        # decoder部分
        x = self.up2(x4, x3)
        x = self.multi_dir_att_decoder1(x)  # 解码器注意力

        x = self.up3(x, x2)
        x = self.multi_dir_att_decoder2(x)  # 解码器注意力

        x = self.up4(x, x1)
        x = self.multi_dir_att_decoder3(x)  # 解码器注意力

        logits = self.outc(x)
        outputs = self.softmax(logits)
        return outputs

# 方案C：编码器深层 + 解码器（完整型）
if __name__ == '__main__':
    # 查看网络参数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))