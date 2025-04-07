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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class FaultSeg3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(FaultSeg3D, self).__init__()

        features = init_features
        self.encoder1 = ConvBlock(in_channels, features)
        self.encoder2 = ConvBlock(features, features * 2)
        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.encoder4 = ConvBlock(features * 4, features * 8)

        self.bottleneck = ConvBlock(features * 8, features * 16)

        self.upconv4 = UpConv(features * 16, features * 8)
        self.decoder4 = ConvBlock(features * 16, features * 8)
        self.upconv3 = UpConv(features * 8, features * 4)
        self.decoder3 = ConvBlock(features * 8, features * 4)
        self.upconv2 = UpConv(features * 4, features * 2)
        self.decoder2 = ConvBlock(features * 4, features * 2)
        self.upconv1 = UpConv(features * 2, features)
        self.decoder1 = ConvBlock(features * 2, features)

        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool3d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool3d(enc3, kernel_size=2, stride=2))

        bottleneck = self.bottleneck(F.max_pool3d(enc4, kernel_size=2, stride=2))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))

if __name__ == '__main__':
    # 查看网络参数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))


# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv3d-1    [-1, 32, 128, 128, 128]             896
#        BatchNorm3d-2    [-1, 32, 128, 128, 128]              64
#               ReLU-3    [-1, 32, 128, 128, 128]               0
#             Conv3d-4    [-1, 32, 128, 128, 128]          27,680
#        BatchNorm3d-5    [-1, 32, 128, 128, 128]              64
#               ReLU-6    [-1, 32, 128, 128, 128]               0
#          ConvBlock-7    [-1, 32, 128, 128, 128]               0
#             Conv3d-8       [-1, 64, 64, 64, 64]          55,360
#        BatchNorm3d-9       [-1, 64, 64, 64, 64]             128
#              ReLU-10       [-1, 64, 64, 64, 64]               0
#            Conv3d-11       [-1, 64, 64, 64, 64]         110,656
#       BatchNorm3d-12       [-1, 64, 64, 64, 64]             128
#              ReLU-13       [-1, 64, 64, 64, 64]               0
#         ConvBlock-14       [-1, 64, 64, 64, 64]               0
#            Conv3d-15      [-1, 128, 32, 32, 32]         221,312
#       BatchNorm3d-16      [-1, 128, 32, 32, 32]             256
#              ReLU-17      [-1, 128, 32, 32, 32]               0
#            Conv3d-18      [-1, 128, 32, 32, 32]         442,496
#       BatchNorm3d-19      [-1, 128, 32, 32, 32]             256
#              ReLU-20      [-1, 128, 32, 32, 32]               0
#         ConvBlock-21      [-1, 128, 32, 32, 32]               0
#            Conv3d-22      [-1, 256, 16, 16, 16]         884,992
#       BatchNorm3d-23      [-1, 256, 16, 16, 16]             512
#              ReLU-24      [-1, 256, 16, 16, 16]               0
#            Conv3d-25      [-1, 256, 16, 16, 16]       1,769,728
#       BatchNorm3d-26      [-1, 256, 16, 16, 16]             512
#              ReLU-27      [-1, 256, 16, 16, 16]               0
#         ConvBlock-28      [-1, 256, 16, 16, 16]               0
#            Conv3d-29         [-1, 512, 8, 8, 8]       3,539,456
#       BatchNorm3d-30         [-1, 512, 8, 8, 8]           1,024
#              ReLU-31         [-1, 512, 8, 8, 8]               0
#            Conv3d-32         [-1, 512, 8, 8, 8]       7,078,400
#       BatchNorm3d-33         [-1, 512, 8, 8, 8]           1,024
#              ReLU-34         [-1, 512, 8, 8, 8]               0
#         ConvBlock-35         [-1, 512, 8, 8, 8]               0
#   ConvTranspose3d-36      [-1, 256, 16, 16, 16]       1,048,832
#            Conv3d-37      [-1, 256, 16, 16, 16]       1,769,728
#       BatchNorm3d-38      [-1, 256, 16, 16, 16]             512
#              ReLU-39      [-1, 256, 16, 16, 16]               0
#            Conv3d-40      [-1, 256, 16, 16, 16]       1,769,728
#       BatchNorm3d-41      [-1, 256, 16, 16, 16]             512
#              ReLU-42      [-1, 256, 16, 16, 16]               0
#         ConvBlock-43      [-1, 256, 16, 16, 16]               0
#            UpConv-44      [-1, 256, 16, 16, 16]               0
#            Conv3d-45      [-1, 256, 16, 16, 16]       3,539,200
#       BatchNorm3d-46      [-1, 256, 16, 16, 16]             512
#              ReLU-47      [-1, 256, 16, 16, 16]               0
#            Conv3d-48      [-1, 256, 16, 16, 16]       1,769,728
#       BatchNorm3d-49      [-1, 256, 16, 16, 16]             512
#              ReLU-50      [-1, 256, 16, 16, 16]               0
#         ConvBlock-51      [-1, 256, 16, 16, 16]               0
#   ConvTranspose3d-52      [-1, 128, 32, 32, 32]         262,272
#            Conv3d-53      [-1, 128, 32, 32, 32]         442,496
#       BatchNorm3d-54      [-1, 128, 32, 32, 32]             256
#              ReLU-55      [-1, 128, 32, 32, 32]               0
#            Conv3d-56      [-1, 128, 32, 32, 32]         442,496
#       BatchNorm3d-57      [-1, 128, 32, 32, 32]             256
#              ReLU-58      [-1, 128, 32, 32, 32]               0
#         ConvBlock-59      [-1, 128, 32, 32, 32]               0
#            UpConv-60      [-1, 128, 32, 32, 32]               0
#            Conv3d-61      [-1, 128, 32, 32, 32]         884,864
#       BatchNorm3d-62      [-1, 128, 32, 32, 32]             256
#              ReLU-63      [-1, 128, 32, 32, 32]               0
#            Conv3d-64      [-1, 128, 32, 32, 32]         442,496
#       BatchNorm3d-65      [-1, 128, 32, 32, 32]             256
#              ReLU-66      [-1, 128, 32, 32, 32]               0
#         ConvBlock-67      [-1, 128, 32, 32, 32]               0
#   ConvTranspose3d-68       [-1, 64, 64, 64, 64]          65,600
#            Conv3d-69       [-1, 64, 64, 64, 64]         110,656
#       BatchNorm3d-70       [-1, 64, 64, 64, 64]             128
#              ReLU-71       [-1, 64, 64, 64, 64]               0
#            Conv3d-72       [-1, 64, 64, 64, 64]         110,656
#       BatchNorm3d-73       [-1, 64, 64, 64, 64]             128
#              ReLU-74       [-1, 64, 64, 64, 64]               0
#         ConvBlock-75       [-1, 64, 64, 64, 64]               0
#            UpConv-76       [-1, 64, 64, 64, 64]               0
#            Conv3d-77       [-1, 64, 64, 64, 64]         221,248
#       BatchNorm3d-78       [-1, 64, 64, 64, 64]             128
#              ReLU-79       [-1, 64, 64, 64, 64]               0
#            Conv3d-80       [-1, 64, 64, 64, 64]         110,656
#       BatchNorm3d-81       [-1, 64, 64, 64, 64]             128
#              ReLU-82       [-1, 64, 64, 64, 64]               0
#         ConvBlock-83       [-1, 64, 64, 64, 64]               0
#   ConvTranspose3d-84    [-1, 32, 128, 128, 128]          16,416
#            Conv3d-85    [-1, 32, 128, 128, 128]          27,680
#       BatchNorm3d-86    [-1, 32, 128, 128, 128]              64
#              ReLU-87    [-1, 32, 128, 128, 128]               0
#            Conv3d-88    [-1, 32, 128, 128, 128]          27,680
#       BatchNorm3d-89    [-1, 32, 128, 128, 128]              64
#              ReLU-90    [-1, 32, 128, 128, 128]               0
#         ConvBlock-91    [-1, 32, 128, 128, 128]               0
#            UpConv-92    [-1, 32, 128, 128, 128]               0
#            Conv3d-93    [-1, 32, 128, 128, 128]          55,328
#       BatchNorm3d-94    [-1, 32, 128, 128, 128]              64
#              ReLU-95    [-1, 32, 128, 128, 128]               0
#            Conv3d-96    [-1, 32, 128, 128, 128]          27,680
#       BatchNorm3d-97    [-1, 32, 128, 128, 128]              64
#              ReLU-98    [-1, 32, 128, 128, 128]               0
#         ConvBlock-99    [-1, 32, 128, 128, 128]               0
#           Conv3d-100     [-1, 2, 128, 128, 128]              66
# ================================================================
# Total params: 27,284,290
# Trainable params: 27,284,290
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 15686.00
# Params size (MB): 104.08
# Estimated Total Size (MB): 15798.08
# ----------------------------------------------------------------



