import torch
import torch.nn as nn

class CA_Block_3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block_3D, self).__init__()
        reduced_c = max(8, channel // reduction)

        self.conv_reduce = nn.Conv3d(channel, reduced_c, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm3d(reduced_c)
        self.relu = nn.ReLU(inplace=True)

        self.F_d = nn.Conv3d(reduced_c, channel, kernel_size=1, stride=1, bias=False)
        self.F_h = nn.Conv3d(reduced_c, channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv3d(reduced_c, channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.size()

        # Spatial attention maps
        x_d = torch.mean(x, dim=4, keepdim=True)              # (B, C, D, H, 1)
        x_h = torch.mean(x, dim=3, keepdim=True)              # (B, C, D, 1, W)
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0,1,3,2,4)  # (B, C, H, 1, W)

        x_cat = torch.cat([x_d, x_h, x_w], dim=4)             # (B, C, D, H, L) — 可调

        out = self.conv_reduce(x_cat)
        out = self.bn(out)
        out = self.relu(out)

        out_d, out_h, out_w = torch.chunk(out, chunks=3, dim=4)

        s_d = self.sigmoid(self.F_d(out_d))   # (B, C, D, H, 1)
        s_h = self.sigmoid(self.F_h(out_h))   # (B, C, D, 1, W)
        s_w = self.sigmoid(self.F_w(out_w.permute(0,1,3,2,4)))  # (B, C, D, H, W)

        s_d = s_d.expand_as(x)
        s_h = s_h.expand_as(x)
        s_w = s_w.expand_as(x)

        return x * s_d * s_h * s_w


if __name__ == '__main__':
    model = CA_Block_3D(channel=128)
    x = torch.randn(2, 128, 16, 32, 32)
    y = model(x)
    print(y.shape)  # should match input
