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
        self.bn = nn.BatchNorm2d(channel // reduction)

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


if __name__ == '__main__':
    CA_Block  = CA_Block_3D(128)
