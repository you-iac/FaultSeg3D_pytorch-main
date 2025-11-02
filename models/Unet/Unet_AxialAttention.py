import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class AxialAttention(nn.Module):
    """
    轴向注意力 - 沿单个轴计算注意力
    将3D注意力分解为3个1D注意力，大幅降低计算量
    """
    def __init__(self, in_channels, axis, reduction=8):
        """
        Args:
            in_channels: 输入通道数
            axis: 计算注意力的轴 (0=D, 1=H, 2=W)
            reduction: 降维比例
        """
        super(AxialAttention, self).__init__()
        self.in_channels = in_channels
        self.axis = axis
        self.inter_channels = max(in_channels // reduction, 1)
        
        # Query, Key, Value 投影
        self.query_conv = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        # 输出投影
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            out: (B, C, D, H, W)
        """
        B, C, D, H, W = x.size()
        
        # 根据axis重排维度
        if self.axis == 0:  # 沿D轴
            # (B, C, D, H, W) -> (B*H*W, C, D)
            x_reshaped = x.permute(0, 3, 4, 1, 2).contiguous()
            x_reshaped = x_reshaped.view(B * H * W, C, D)
            spatial_dim = D
            restore_shape = lambda out: out.view(B, H, W, C, D).permute(0, 3, 4, 1, 2)
            
        elif self.axis == 1:  # 沿H轴
            # (B, C, D, H, W) -> (B*D*W, C, H)
            x_reshaped = x.permute(0, 2, 4, 1, 3).contiguous()
            x_reshaped = x_reshaped.view(B * D * W, C, H)
            spatial_dim = H
            restore_shape = lambda out: out.view(B, D, W, C, H).permute(0, 3, 1, 4, 2)
            
        else:  # axis == 2, 沿W轴
            # (B, C, D, H, W) -> (B*D*H, C, W)
            x_reshaped = x.permute(0, 2, 3, 1, 4).contiguous()
            x_reshaped = x_reshaped.view(B * D * H, C, W)
            spatial_dim = W
            restore_shape = lambda out: out.view(B, D, H, C, W).permute(0, 3, 1, 2, 4)
        
        # 计算 Q, K, V
        query = self.query_conv(x_reshaped)  # (B*spatial_prod, C', spatial_dim)
        key = self.key_conv(x_reshaped)      # (B*spatial_prod, C', spatial_dim)
        value = self.value_conv(x_reshaped)  # (B*spatial_prod, C, spatial_dim)
        
        # 计算注意力
        # (B*spatial_prod, spatial_dim, C') @ (B*spatial_prod, C', spatial_dim)
        # -> (B*spatial_prod, spatial_dim, spatial_dim)
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention / (self.inter_channels ** 0.5), dim=-1)
        
        # 应用注意力
        # (B*spatial_prod, C, spatial_dim) @ (B*spatial_prod, spatial_dim, spatial_dim)
        # -> (B*spatial_prod, C, spatial_dim)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        
        # 恢复原始形状
        out = restore_shape(out)
        
        # 残差连接
        out = self.gamma * out + x
        return out


class MultiAxisAttention(nn.Module):
    """
    多轴注意力 - 依次在D、H、W三个轴上应用注意力
    """
    def __init__(self, in_channels, reduction=8):
        super(MultiAxisAttention, self).__init__()
        self.att_d = AxialAttention(in_channels, axis=0, reduction=reduction)
        self.att_h = AxialAttention(in_channels, axis=1, reduction=reduction)
        self.att_w = AxialAttention(in_channels, axis=2, reduction=reduction)
        
    def forward(self, x):
        x = self.att_d(x)  # D轴注意力
        x = self.att_h(x)  # H轴注意力
        x = self.att_w(x)  # W轴注意力
        return x


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
    """
    使用轴向注意力的3D UNet
    可以在大尺寸特征图上使用，显存消耗低
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
        
        # 在多个层使用轴向注意力（可以在大尺寸上使用）
        # 64×64×64层
        self.axial_att_64 = MultiAxisAttention(32, reduction=8)
        
        # 32×32×32层
        self.axial_att_32 = MultiAxisAttention(64, reduction=8)
        
        # 16×16×16层
        self.axial_att_16 = MultiAxisAttention(128, reduction=8)
        
        # 解码器
        self.up2 = Up(192, 64)
        self.up3 = Up(96, 32)
        self.up4 = Up(48, 16)
        self.outc = OutConv(16, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder部分
        x1 = self.inc(x)        # 16 × 128³
        
        x2 = self.down1(x1)     # 32 × 64³
        x2 = self.axial_att_64(x2)  # 轴向注意力
        
        x3 = self.down2(x2)     # 64 × 32³
        x3 = self.axial_att_32(x3)  # 轴向注意力
        
        x4 = self.down3(x3)     # 128 × 16³
        x4 = self.axial_att_16(x4)  # 轴向注意力
        
        # decoder部分
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        outputs = self.softmax(logits)
        return outputs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))

# 显存分析（轴向注意力）：
# 
# 对于128×128×128的特征图：
# - 传统自注意力: 128³ × 128³ = 4.4 trillion（不可行）
# - 轴向注意力: 
#   * D轴: (128×128) × 128 × 128 = 268M
#   * H轴: (128×128) × 128 × 128 = 268M  
#   * W轴: (128×128) × 128 × 128 = 268M
#   总计: 804M（可行！）
# 
# 复杂度对比:
# - 传统: O(N²) = O((128³)²)
# - 轴向: O(3×N×sqrt(N)) = O(3×128³×128)
# 降低约 128³/3 = 700,000倍！✓


