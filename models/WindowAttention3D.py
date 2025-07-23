import torch
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
class WindowAttention3D(nn.Module):
    """3D窗口注意力机制（修复版）"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 1. 计算相对位置索引
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)

        # 计算相对位置
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # 将相对位置索引归一化到非负范围
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # 2. 修复相对位置偏置表（正确初始化）
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) *
                (2 * window_size[1] - 1) *
                (2 * window_size[2] - 1),
                num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # 3. 其他参数
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        head_dim = C // self.num_heads

        # 生成QKV
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 添加相对位置偏置（修复）
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 输出投影
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """3D Swin Transformer块（修复版）"""

    def __init__(self, dim, window_size, num_heads, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size, num_heads, qkv_bias, attn_drop, drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, mask_matrix=None):
        B, L, H, W, C = x.shape
        window_size = self.window_size

        # 短连接
        shortcut = x
        x = self.norm1(x)

        # 填充处理
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - L % window_size[0]) % window_size[0]
        pad_h = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_w = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_w, pad_t, pad_h, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # 窗口划分
        x_windows = window_partition(x, window_size)
        x_windows = x_windows.view(-1, window_size[0] * window_size[1] * window_size[2], C)

        # 注意力计算
        attn_windows = self.attn(x_windows, mask_matrix)

        # 窗口合并
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], window_size[2], C)
        x = window_reverse(attn_windows, window_size, Dp, Hp, Wp)

        # 移除填充
        if pad_d1 > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :L, :H, :W, :].contiguous()

        # 残差连接
        x = shortcut + x

        # MLP部分
        x = x + self.mlp(self.norm2(x))

        return x


def window_partition(x, window_size):
    """将输入划分为窗口"""
    B, D, H, W, C = x.shape
    x = x.view(B,
               D // window_size[0], window_size[0],
               H // window_size[1], window_size[1],
               W // window_size[2], window_size[2],
               C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, D, H, W):
    """将窗口合并回原始形状"""
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B,
                     D // window_size[0],
                     H // window_size[1],
                     W // window_size[2],
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    return x


class Transformer3D(nn.Module):
    """3D Transformer模块（修复版）"""

    def __init__(self, dim, depths, num_heads, window_size=(4, 4, 4),
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.dim = dim
        self.depths = depths
        self.window_size = window_size

        # 构建Transformer块
        self.blocks = nn.ModuleList()
        for i_layer in range(depths):
            layer = SwinTransformerBlock3D(
                dim=dim,
                window_size=window_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=nn.LayerNorm
            )
            self.blocks.append(layer)

    def forward(self, x):
        # 输入形状: [B, C, L, H, W]
        B, C, L, H, W = x.shape

        # 转换为通道后置: [B, L, H, W, C]
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        # 计算注意力掩码（可选）
        mask_matrix = None

        # 通过Transformer块
        for blk in self.blocks:
            x = blk(x, mask_matrix)

        # 转换回通道前置: [B, C, L, H, W]
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x