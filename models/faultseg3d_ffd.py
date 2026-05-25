import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
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


class FourierHighPassInput3D(nn.Module):
    """Create a spatially aligned high-frequency branch using FFT and iFFT."""

    def __init__(self, cutoff=0.18, eps=1e-6):
        super().__init__()
        self.cutoff = cutoff
        self.eps = eps
        self.branch_scale = nn.Parameter(torch.tensor(0.0))

    def _high_pass_mask(self, spatial_size, device, dtype):
        depth, height, width = spatial_size
        freq_z = torch.fft.fftfreq(depth, device=device, dtype=dtype)
        freq_y = torch.fft.fftfreq(height, device=device, dtype=dtype)
        freq_x = torch.fft.rfftfreq(width, device=device, dtype=dtype)

        radius = torch.sqrt(
            freq_z[:, None, None].square()
            + freq_y[None, :, None].square()
            + freq_x[None, None, :].square()
        )
        radius = radius / radius.max().clamp_min(self.eps)

        # Smooth high-pass mask avoids ringing from a hard frequency cutoff.
        mask = 1.0 - torch.exp(-((radius / self.cutoff).square()))
        return mask.view(1, 1, depth, height, width // 2 + 1)

    def forward(self, x):
        spatial_size = x.shape[-3:]
        mask = self._high_pass_mask(spatial_size, x.device, x.dtype)

        x_fft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
        high = torch.fft.irfftn(
            x_fft * mask,
            s=spatial_size,
            dim=(-3, -2, -1),
            norm="ortho",
        )

        dims = tuple(range(2, high.dim()))
        mean = high.mean(dim=dims, keepdim=True)
        std = high.std(dim=dims, keepdim=True).clamp_min(self.eps)
        high = (high - mean) / std

        high = high * torch.sigmoid(self.branch_scale)
        return torch.cat([x, high], dim=1)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels),
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

        self.freq_input = FourierHighPassInput3D(cutoff=0.18)
        self.inc = DoubleConv(n_channels * 2, 16)

        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        self.up2 = Up(192, 64)
        self.up3 = Up(96, 32)
        self.up4 = Up(48, 16)
        self.outc = OutConv(16, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.freq_input(x)

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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))
