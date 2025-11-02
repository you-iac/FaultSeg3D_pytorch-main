"""
FaultSeg3D - 基于 CEDNet 架构的3D地震断层分割网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# ===== DropPath 实现 =====
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    实现随机深度正则化，在训练时随机丢弃整个残差路径

    参数:
        x: 输入张量
        drop_prob: 丢弃概率 (0 表示不丢弃)
        training: 是否在训练模式
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output






class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    这是一个PyTorch模块，可以直接替代 timm.models.layers.DropPath
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm3d(nn.Module):
    """3D LayerNorm"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[None, :, None, None, None] * x + self.bias[None, :, None, None, None]
        return x


class CEDBlock(nn.Module):
    """
    CEDNet基础块 - 3D版本
    类似ConvNeXt Block但针对3D数据优化
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = LayerNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma[None, :, None, None, None] * x
        x = input + self.drop_path(x)
        return x


class DownsampleLayer(nn.Module):
    """下采样层"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class UpsampleLayer(nn.Module):
    """上采样层"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = LayerNorm3d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class FaultSeg3D(nn.Module):
    """
    CEDNet for 3D Fault Segmentation

    保持与FaultSeg3D相同的接口：
    - 输入: (B, n_channels, D, H, W)
    - 输出: (B, n_classes, D, H, W)

    参数:
        n_channels: 输入通道数 (默认1，地震数据)
        n_classes: 输出类别数 (默认2，二分类)
        depths: 每个stage的block数量
        dims: 每个stage的通道数
        drop_path_rate: DropPath率
        kernel_sizes: 每个stage的卷积核大小
    """
    def __init__(self, n_channels=1, n_classes=2, model_size='small'):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        # 根据 model_size 选择配置
        configs = {
            'tiny': {
                'depths': [2, 2, 4, 2],
                'dims': [24, 48, 96, 192],
                'drop_path_rate': 0.05,
                'kernel_sizes': [7, 7, 5, 3]
            },
            'small': {
                'depths': [2, 2, 6, 2],
                'dims': [32, 64, 128, 256],
                'drop_path_rate': 0.1,
                'kernel_sizes': [7, 7, 5, 5]
            },
            'base': {
                'depths': [3, 3, 9, 3],
                'dims': [48, 96, 192, 384],
                'drop_path_rate': 0.15,
                'kernel_sizes': [7, 7, 5, 5]
            },
            'large': {
                'depths': [3, 3, 12, 3],
                'dims': [64, 128, 256, 512],
                'drop_path_rate': 0.2,
                'kernel_sizes': [7, 7, 5, 5]
            }
        }

        config = configs.get(model_size, configs['small'])
        depths = config['depths']
        dims = config['dims']
        drop_path_rate = config['drop_path_rate']
        kernel_sizes = config['kernel_sizes']
        layer_scale_init_value = 1e-6

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(n_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm3d(dims[0])
        )

        # 编码器
        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[CEDBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                          layer_scale_init_value=layer_scale_init_value,
                          kernel_size=kernel_sizes[i]) for j in range(depths[i])]
            )
            self.encoder_stages.append(stage)
            cur += depths[i]
            if i < 3:
                self.downsample_layers.append(DownsampleLayer(dims[i], dims[i+1]))

        # 解码器
        self.decoder_stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()

        for i in range(3, 0, -1):
            self.upsample_layers.append(UpsampleLayer(dims[i], dims[i-1]))
            self.fusion_convs.append(nn.Sequential(
                nn.Conv3d(dims[i-1] * 2, dims[i-1], kernel_size=3, padding=1),
                LayerNorm3d(dims[i-1]),
                nn.GELU()
            ))
            decoder_stage = nn.Sequential(
                *[CEDBlock(dim=dims[i-1], drop_path=0.,
                          layer_scale_init_value=layer_scale_init_value,
                          kernel_size=kernel_sizes[i-1]) for _ in range(depths[i-1])]
            )
            self.decoder_stages.append(decoder_stage)

        # 输出头
        self.head = nn.Sequential(
            nn.Conv3d(dims[0], dims[0], kernel_size=3, padding=1),
            LayerNorm3d(dims[0]),
            nn.GELU(),
            nn.Conv3d(dims[0], n_classes, kernel_size=1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        skip_connections = []

        # 编码器
        for i in range(4):
            x = self.encoder_stages[i](x)
            skip_connections.append(x)
            if i < 3:
                x = self.downsample_layers[i](x)

        # 解码器
        for i in range(3):
            x = self.upsample_layers[i](x)
            skip = skip_connections[2 - i]
            if x.shape != skip.shape:
                diff_d = skip.size()[2] - x.size()[2]
                diff_h = skip.size()[3] - x.size()[3]
                diff_w = skip.size()[4] - x.size()[4]
                x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                             diff_h // 2, diff_h - diff_h // 2,
                             diff_d // 2, diff_d - diff_d // 2])
            x = torch.cat([x, skip], dim=1)
            x = self.fusion_convs[i](x)
            x = self.decoder_stages[i](x)

        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=True)
        x = self.head(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':

    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))


# ================================================================
# Total params: 5,891,138
# Trainable params: 5,891,138
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 2327.50
# Params size (MB): 22.47
# Estimated Total Size (MB): 2357.97
# ----------------------------------------------------------------