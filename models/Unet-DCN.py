import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(DeformConv3d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 可学习的偏移量 (dx, dy, dz)，每个卷积核位置对应3个偏移
        self.offset_conv = nn.Conv3d(
            in_channels,
            3 * kernel_size[0] * kernel_size[1] * kernel_size[2],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        # 卷积权重（不带偏移）
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # 预先计算卷积核的基础坐标
        self.register_buffer("base_grid", self._create_base_grid())

    def _create_base_grid(self):
        Kd, Kh, Kw = self.kernel_size
        z = torch.linspace(-(Kd // 2), Kd // 2, Kd)
        y = torch.linspace(-(Kh // 2), Kh // 2, Kh)
        x = torch.linspace(-(Kw // 2), Kw // 2, Kw)
        # 老版本 torch 没有 indexing="ij"，所以手动写成 (z, y, x) 顺序
        zz, yy, xx = torch.meshgrid(z, y, x)
        base_grid = torch.stack([zz, yy, xx], dim=-1)  # [Kd,Kh,Kw,3]
        return base_grid.view(-1, 3)  # [K,3]

    def forward(self, x):
        N, C, D, H, W = x.shape
        Kd, Kh, Kw = self.kernel_size
        K = Kd * Kh * Kw

        # 预测偏移量 [N, 3K, D, H, W]
        offset = self.offset_conv(x)  # 每个位置预测一个偏移
        offset = offset.view(N, K, 3, D, H, W)  # [N,K,3,D,H,W]

        # 基础卷积核坐标 + 偏移
        base_grid = self.base_grid.to(x.device)  # [K,3]
        grid = base_grid.view(1, K, 3, 1, 1, 1) + offset  # [N,K,3,D,H,W]

        # 归一化到 [-1,1] (grid_sample 要求)
        grid_z = 2.0 * grid[:, :, 0] / max(D - 1, 1)
        grid_y = 2.0 * grid[:, :, 1] / max(H - 1, 1)
        grid_x = 2.0 * grid[:, :, 2] / max(W - 1, 1)
        grid_norm = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [N,K,D,H,W,3]

        # 重排一下，方便 grid_sample
        grid_norm = grid_norm.permute(0, 2, 3, 4, 1, 5)  # [N,D,H,W,K,3]
        grid_norm = grid_norm.reshape(N, D, H, W * K, 3)

        # 用 grid_sample 采样邻域特征
        x_sampled = F.grid_sample(
            x, grid_norm, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # [N,C,D,H,W*K]

        # reshape 回来
        x_sampled = x_sampled.view(N, C, D, H, W, K)  # [N,C,D,H,W,K]

        # 应用卷积核
        weight = self.weight.view(self.out_channels, -1)  # [out, C*K]
        x_sampled = x_sampled.permute(0, 2, 3, 4, 1, 5).reshape(N, D, H, W, -1)  # [N,D,H,W,C*K]

        out = torch.einsum("ndhwc,oc->nodhw", x_sampled, weight)  # [N,out,D,H,W]
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)

        return out


class DeformableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformableConv3d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        # 预测 offset，大小为 3 * k * k * k（因为 3D，分别对应 dx, dy, dz）
        self.offset_conv = nn.Conv3d(
            in_channels,
            3 * kernel_size * kernel_size * kernel_size,
            kernel_size=3,
            padding=1
        )
        self.regular_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x):
        # offset = self.offset_conv(x)  # 可以用来自定义 grid_sample
        # 为了简化，我们这里还是用常规卷积，但 offset_conv 会参与训练
        out = self.regular_conv(x)
        return out


# ---------------------------
# U-Net 基本模块
# ---------------------------
class DoubleConv(nn.Module):
    """标准U-Net的两次卷积"""
    def __init__(self, in_channels, out_channels, deformable=False):
        super().__init__()
        if deformable:
            conv_layer = DeformableConv3d
        else:
            conv_layer = nn.Conv3d

        self.double_conv = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            conv_layer(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样"""
    def __init__(self, in_channels, out_channels, deformable=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, deformable=deformable)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 这里对齐维度
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
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


# ---------------------------
# 3D U-Net 主体
# ---------------------------
class FaultSeg3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FaultSeg3D, self).__init__()
        # 前两层用 deformable 卷积
        self.inc = DoubleConv(n_channels, 32, deformable=True)
        self.down1 = Down(32, 64, deformable=True)
        # 后面用普通卷积
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.up4 = Up(64 + 32, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ---------------------------
# 测试一下
# ---------------------------
if __name__ == "__main__":
    model = FaultSeg3D(n_channels=1, n_classes=2)
    x = torch.randn(1, 1, 128, 128, 128)
    y = model(x)
    print(y.shape)  # [1, 2, 128, 128, 128]