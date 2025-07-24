import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

#普通卷积操作可以提取局部特征，而自注意力机制可以实现对全局特征的提取，考虑到普通的自注意力机制直接应用于3d模型的计算会导致参数量过大影响计算性能，
#我们引入了基于SwinTransformer的滑动窗口机制，可以有效的减少运算量

# 标准卷积操作能够有效提取局部空间特征，
# 而自注意力机制则具备建模长距离依赖关系的能力，
# 可捕获全局上下文信息。
# 然而，在三维数据处理场景中，
# 传统自注意力机制因其计算复杂度随输入尺寸呈立方级增长，
# 会导致参数量激增和计算效率显著下降。
# 为解决这一问题，我们引入基于Swin Transformer的滑动窗口注意力机制，
# 该设计通过局部窗口内的自注意力计算与窗口间信息交互相结合的策略，
# 在保持全局建模能力的同时，将计算复杂度降至线性水平，从而显著提升模型的计算效率。
import torch
import torch
import torch.nn as nn
from monai.networks.blocks import PatchEmbed
from monai.networks.nets.swin_unetr import BasicLayer


class SwinSkipConnection(nn.Module):
    """
    通用SwinTransformer3D跳跃连接模块。
    输入输出形状均为 (B, C, D, H, W)，空间尺寸与通道数保持一致。
    """

    def __init__(
            self,
            in_channels: int,
            window_size=(4, 8, 8),
            depth: int = 2,
            num_heads: int = 4,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        # 1x1x1 patch embedding 保留输入空间大小
        self.patch_embed = PatchEmbed(
            patch_size=(1, 1, 1),
            in_chans=in_channels,
            embed_dim=in_channels,
            norm_layer=None,  # 不使用归一化
            spatial_dims=3
        )
        # Swin Transformer 基础层，不做降采样，以保持空间尺寸
        self.swin_layer = BasicLayer(
            dim=in_channels,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            drop_path=[0.0] * depth,  # 可使用线性递增的 drop_path
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=dropout,
            attn_drop=attn_dropout,
            norm_layer=nn.LayerNorm,
            downsample=None,  # 关闭下采样，保持尺寸不变:contentReference[oaicite:10]{index=10}
        )

    def forward(self, x):
        # 输入验证：应为 (B, C, D, H, W) 的5维张量
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}"
            )
        # Patch嵌入（1x1卷积）：输出形状 (B, C, D, H, W):contentReference[oaicite:11]{index=11}
        x = self.patch_embed(x)
        # SwinTransformer BasicLayer：输出形状 (B, C, D, H, W):contentReference[oaicite:12]{index=12}
        x = self.swin_layer(x)
        return x


if __name__ == '__main__':
    # 创建模型实例
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SwinSkipConnection(16).to(device)

    # 使用 5D 张量输入，修正 input_size 参数
    summary(net, input_size=(16, 128, 128, 128))  # 这里的 (16, 128, 128, 128) 是 (C, D, H, W)

