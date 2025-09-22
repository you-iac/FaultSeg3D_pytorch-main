import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

# ----------------- 简化 DeformConv3D -----------------
class DeformConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 偏移量卷积
        self.offset_conv = nn.Conv3d(
            in_channels,
            3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2],
            kernel_size=self.kernel_size,
            padding=self.padding
        )

        # 卷积权重
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        N, C, D, H, W = x.shape
        Kz, Ky, Kx = self.kernel_size

        # 偏移量
        offset = self.offset_conv(x)  # [N, 3*K^3, D, H, W]
        offset = offset.view(N, Kz*Ky*Kx, 3, D, H, W)

        # 基础卷积网格
        z = torch.linspace(-(Kz//2), Kz//2, Kz, device=x.device)
        y = torch.linspace(-(Ky//2), Ky//2, Ky, device=x.device)
        xx, yy, zz = torch.meshgrid(x, y, z)  # 兼容老版本 PyTorch
        base_grid = torch.stack([xx, yy, zz], dim=-1).view(-1,3)  # [K^3,3]

        sampled = []
        for k in range(Kz*Ky*Kx):
            g = base_grid[k].view(1,3,1,1,1) + offset[:,k]  # [N,3,D,H,W]
            g = g.permute(0,2,3,4,1)  # [N,D,H,W,3]
            x_s = F.grid_sample(x, g, mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled.append(x_s)

        sampled = torch.stack(sampled, dim=2)  # [N,C,K^3,D,H,W]
        weight = self.weight.view(self.out_channels, self.in_channels, -1)
        out = torch.einsum("oik,nckdhw->nodhw", weight, sampled) + self.bias.view(1,-1,1,1,1)
        return out

# ----------------- 原 DoubleConv -----------------
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

# ----------------- Down / Up / OutConv -----------------
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
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ----------------- FaultSeg3D -----------------
class FaultSeg3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 前两层使用 DCN
        self.inc = nn.Sequential(
            DeformConv3D(n_channels, 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.MaxPool3d(2),
            DeformConv3D(16, 32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        # 后面保持原 DoubleConv
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        self.up2 = Up(192, 64)
        self.up3 = Up(96, 32)
        self.up4 = Up(48, 16)
        self.outc = OutConv(16, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        outputs = self.softmax(logits)
        return outputs

# ----------------- 测试 -----------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)

    # 注意 input_size 是不含 batch 的
    summary(net, input_size=(1, 128, 128, 128))
