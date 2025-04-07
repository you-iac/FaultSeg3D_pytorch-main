# file: 3d_unet_plus_plus.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3DPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3DPlusPlus, self).__init__()

        features = init_features
        self.encoder1 = ConvBlock(in_channels, features)
        self.encoder2 = ConvBlock(features, features * 2)
        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.encoder4 = ConvBlock(features * 4, features * 8)

        self.bottleneck = ConvBlock(features * 8, features * 16)

        self.upconv4 = UpConv(features * 16, features * 8)
        self.upconv3 = UpConv(features * 8, features * 4)
        self.upconv2 = UpConv(features * 4, features * 2)
        self.upconv1 = UpConv(features * 2, features)

        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, 2))
        enc3 = self.encoder3(F.max_pool3d(enc2, 2))
        enc4 = self.encoder4(F.max_pool3d(enc3, 2))

        bottleneck = self.bottleneck(F.max_pool3d(enc4, 2))

        dec4 = self.upconv4(bottleneck, enc4)
        dec3 = self.upconv3(dec4, enc3)
        dec2 = self.upconv2(dec3, enc2)
        dec1 = self.upconv1(dec2, enc1)

        return torch.sigmoid(self.conv(dec1))

# Example usage
if __name__ == "__main__":
    model = UNet3DPlusPlus(in_channels=1, out_channels=1)
    x = torch.randn((1, 1, 64, 64, 64))  # Example input tensor
    preds = model(x)
    print(preds.shape)
    # 查看网络参数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet3DPlusPlus(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))

