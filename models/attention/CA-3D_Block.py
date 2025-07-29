import torch
import torch.nn as nn
import torch.nn.functional as F

class CA_Block_3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block_3D, self).__init__()  # 必须添加

        self.conv_1x1x1 = nn.Conv3d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            bias=False
        )

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(channel // reduction)  # 正确应该是 BatchNorm3d

        # 通道恢复（注意力）
        self.F_l = nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False)
        self.F_w = nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False)
        self.F_h = nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False)

        self.sigmoid_l = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
        self.sigmoid_h = nn.Sigmoid()

    def forward(self, x):
        B, C, H, L, W = x.size()

        # 三个方向的全局平均
        x_l = torch.mean(x, dim=4, keepdim=True).permute(0, 1, 2, 4, 3)  # -> (B, C, H, 1, L)
        x_w = torch.mean(x, dim=3, keepdim=True)                         # -> (B, C, H, 1, W)
        x_h = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2, 4)  # -> (B, C, 1, H, W)

        # 拼接：沿最后一维拼接 (L + W + H)
        x_cat = torch.cat([x_l, x_w, x_h], dim=4)  # -> (B, C, H, 1, L+W+H)

        # 降维 + BN + ReLU
        x_cat_conv = self.relu(self.bn(self.conv_1x1x1(x_cat)))  # -> (B, C//r, H, 1, L+W+H)

        # 拆分
        x_l_conv, x_w_conv, x_h_conv = torch.split(x_cat_conv, [L, W, H], dim=4)

        # 还原维度
        x_l_conv = x_l_conv.permute(0, 1, 2, 4, 3)  # -> (B, C//r, H, L, 1)
        x_h_conv = x_h_conv.permute(0, 1, 4, 3, 2)  # -> (B, C//r, H, 1, W)

        # 通道注意力生成
        s_l = self.sigmoid_l(self.F_l(x_l_conv))  # -> (B, C, H, L, 1)
        s_w = self.sigmoid_w(self.F_w(x_w_conv))  # -> (B, C, H, 1, W)
        s_h = self.sigmoid_h(self.F_h(x_h_conv))  # -> (B, C, H, L, W)

        # 扩展注意力维度
        s_l = s_l.expand_as(x)
        s_w = s_w.expand_as(x)
        s_h = s_h.expand_as(x)

        return x * s_l * s_w * s_h


# 测试
if __name__ == '__main__':
    model = CA_Block_3D(channel=16)
    input_tensor = torch.randn(2, 16, 128, 128, 128)  # (B, C, H, L, W)
    out = model(input_tensor)
    print("Output shape:", out.shape)
