import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        mid_channels = out_channels // reduction

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm3d(mid_channels)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # shortcut 分支
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.in1(self.conv1(x)))
        out = self.relu(self.in2(self.conv2(out)))
        out = self.in3(self.conv3(out))

        out += identity
        return self.relu(out)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            ResidualBlock3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        # 拼接后用 1x1 conv 降维
        self.reduce_channels = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.resblock = ResidualBlock3D(out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # padding 保证尺寸对齐
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.reduce_channels(x)
        return self.resblock(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResUNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = ResidualBlock3D(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        self.up1 = Up(128 + 64, 64)
        self.up2 = Up(64 + 32, 32)
        self.up3 = Up(32 + 16, 16)

        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)
        return logits  # 不做 softmax


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResUNet3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))

# ================================================================
# Total params: 83,654
# Trainable params: 83,654
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 5932.00
# Params size (MB): 0.32
# Estimated Total Size (MB): 5940.32
# ----------------------------------------------------------------