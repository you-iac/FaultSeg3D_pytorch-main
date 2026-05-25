import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class FourierUnit3D(nn.Module):
    """Global branch: FFT -> 1x1x1 conv on real/imag channels -> inverse FFT."""

    def __init__(self, in_channels, out_channels, norm="ortho"):
        super().__init__()
        self.norm = norm
        self.freq_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, out_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        spatial_size = x.shape[-3:]
        x_fft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm=self.norm)

        freq = torch.cat([x_fft.real, x_fft.imag], dim=1)
        freq = self.freq_conv(freq)
        real, imag = torch.chunk(freq, 2, dim=1)

        x_fft = torch.complex(real, imag)
        return torch.fft.irfftn(x_fft, s=spatial_size, dim=(-3, -2, -1), norm=self.norm)


class FourierConv3D(nn.Module):
    """Conv3d replacement with local convolution and global Fourier branch."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        self.local = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.global_branch = FourierUnit3D(in_channels, out_channels)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        local = self.local(x)
        global_feat = self.global_branch(x)

        if global_feat.shape[-3:] != local.shape[-3:]:
            global_feat = F.interpolate(
                global_feat,
                size=local.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        return self.act(self.bn(local + global_feat))


class FFCDoubleConv(nn.Module):
    """DoubleConv variant used only in the deeper encoder stages."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            FourierConv3D(in_channels, mid_channels, kernel_size=3, padding=1),
            FourierConv3D(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=DoubleConv):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            conv_block(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
                diff_z // 2,
                diff_z - diff_z // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FaultSeg3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64, conv_block=FFCDoubleConv)
        self.down3 = Down(64, 128, conv_block=FFCDoubleConv)

        self.up2 = Up(192, 64)
        self.up3 = Up(96, 32)
        self.up4 = Up(48, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


FFC3DBlock = FourierConv3D
FFC = FourierConv3D


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))
