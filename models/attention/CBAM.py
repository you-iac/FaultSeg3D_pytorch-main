import torch
import torch.nn as nn

class CBAM3D(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM3D, self).__init__()

        # channel attention: 压缩 D, H, W 为 1
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False)
        )

        # spatial attention: 压缩通道维 -> 2 通道 (max+avg)
        self.conv = nn.Conv3d(
            2, 1,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ----- Channel Attention -----
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        # ----- Spatial Attention -----
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # (B,1,D,H,W)
        avg_out = torch.mean(x, dim=1, keepdim=True)     # (B,1,D,H,W)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x

if __name__ == '__main__':


    # 测试
    x = torch.randn(2, 32, 128, 128, 128)  # B,C,D,H,W
    net = CBAM3D(32)
    y = net(x)
    print(y.shape)  # 期望: torch.Size([1,32,16,64,64])
