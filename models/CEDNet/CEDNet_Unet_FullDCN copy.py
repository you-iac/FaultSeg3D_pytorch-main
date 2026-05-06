"""
FaultSeg3D - 基于 CEDNet 架构的 3D 地震断层分割网络 (全面DCN增强版)

╔══════════════════════════════════════════════════════════════════════════════╗
║  【模型5/4 - 全DCN版本】本模型特点：                                           ║
║  ✓ 全面DCN增强 - 所有特征提取卷积都使用DCN可变形卷积                           ║
║  ✓ CEDBlock核心DCN - dwconv替换为DCN，自适应断层几何形变                      ║
║  ✓ 编码器全DCN - 所有下采样和特征提取使用DCN                                   ║
║  ✓ 3个级联Stage - 编码-解码级联，特征逐步精炼                                 ║
║  ✓ 4层特征金字塔 - [c2(64³), c3(32³), c4(16³), c5(8³)]                     ║
║  ✓ UPerNet拼接融合 - 所有特征上采样到统一分辨率后拼接                         ║
║  ✓ 固定64通道输出 - 融合后统一为64通道                                        ║
║  ✓ 输出64³后上采样 - 最后上采样到128³                                        ║
║                                                                              ║
║  DCN改进位置（相比CEDNet_Unet.py）：                                          ║
║  • Stem: 2层全部DCN（128³→64³）                                             ║
║  • P2下采样: DCN（64³→32³）                                                  ║
║  • CEDBlock的dwconv: 替换为DCN（核心改进，约24处）                           ║
║  • Encoder下采样: 全部DCN（2次下采样×3个Stage）                              ║
║  • Decoder上采样: 保持标准卷积（DCN不适用于上采样）                            ║
║                                                                              ║
║  与其他版本的区别：                                                           ║
║  vs CEDNet_Unet.py: 所有特征提取卷积都使用DCN                                ║
║  vs CEDNet_Unet_DCN.py: 仅Stem和P2用DCN，本版本全面DCN                       ║
║                                                                              ║
║  适用场景：断层几何形变复杂，需要最强自适应能力，参数量和计算量最大           ║
╚══════════════════════════════════════════════════════════════════════════════╝

完整实现了 CEDNet 的级联编码-解码结构，包括：
- Stem + P2 初始特征提取 (全DCN)
- 3 个级联 Stage (编码-解码对，编码器全DCN)
- CEDBlock 核心块使用DCN
- PPM 金字塔池化模块
- UPerNet 多尺度融合
- 分割头

DCN 改进：
- Stem: 2层卷积全部替换为DCN
- P2下采样: DCN
- CEDBlock的dwconv: 替换为DCN（约24个CEDBlock × 3个Stage = 72次DCN调用）
- Encoder下采样: 2次下采样 × 3个Stage = 6次DCN

输入: (B, 1, 128, 128, 128) - 地震数据
输出: (B, 2, 128, 128, 128) - 二分类分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from tvdcn.ops import deform_conv3d
from torch.utils.checkpoint import checkpoint


_CHECKPOINT_SUPPORTS_USE_REENTRANT = "use_reentrant" in inspect.signature(checkpoint).parameters


# ===== 基础组件 =====

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm3d(nn.Module):
    """3D Layer Normalization"""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, D, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[None, :, None, None, None] * x + self.bias[None, :, None, None, None]
        return x


class DeformConv3dBlock(nn.Module):
    """
    3D 可变形卷积块
    
    支持不同的 stride 和 kernel_size
    结构: Offset/Mask 预测 → DCN → Norm → Activation
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小（默认3）
        stride: 步长（默认1）
        padding: 填充（默认1）
        use_gelu: 是否使用GELU激活（默认True，与CEDNet一致）
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_gelu=True):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        num_points = kernel_size * kernel_size * kernel_size
        offset_mask_channels = 4 * num_points  # 3*K^3 (offset) + 1*K^3 (mask)
        
        # Offset 和 Mask 预测卷积
        # 关键：offset 预测需要使用相同的 stride，确保输出空间维度匹配
        self.offset_mask_conv = nn.Conv3d(
            in_channels, offset_mask_channels, 
            kernel_size=kernel_size, 
            stride=stride,  # 使用相同的 stride
            padding=padding
        )
        
        # DCN 权重参数
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # 初始化权重
        nn.init.kaiming_uniform_(self.weight, a=0)
        self.bias.data.zero_()
        
        # 归一化和激活
        self.norm = LayerNorm3d(out_channels)
        self.act = nn.GELU() if use_gelu else nn.ReLU(inplace=True)
        
    def forward(self, x):
        num_points = self.kernel_size ** 3
        
        # 预测 offset 和 mask
        offset_mask = self.offset_mask_conv(x)
        offset = offset_mask[:, :3 * num_points, :, :, :]
        mask = torch.sigmoid(offset_mask[:, 3 * num_points:, :, :, :])
        
        # 执行可变形卷积
        x = deform_conv3d(
            x, self.weight, offset, mask, self.bias,
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            dilation=(1, 1, 1),
            groups=1
        )
        
        x = self.norm(x)
        x = self.act(x)
        
        return x


class DepthwiseDeformConv3d(nn.Module):
    """
    深度可变形卷积 - 用于替换 CEDBlock 中的 dwconv
    
    每个通道独立进行可变形卷积，保持深度可分离特性
    结构: Offset/Mask 预测 (标准卷积) → DCN (depthwise)
    
    注意：
    1. offset/mask预测使用标准卷积（不是depthwise），因为需要输出固定的offset_mask_channels
    2. 实际的DCN操作是depthwise的（groups=channels）
    3. 不包含归一化和激活，由 CEDBlock 控制
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        num_points = kernel_size * kernel_size * kernel_size
        offset_mask_channels = 4 * num_points  # 3*K^3 (offset) + 1*K^3 (mask)
        
        # Offset 和 Mask 预测卷积（标准卷积，不是depthwise）
        # 原因：offset_mask_channels 可能无法被 channels 整除
        self.offset_mask_conv = nn.Conv3d(
            channels, offset_mask_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            groups=1  # 标准卷积（修复：原来是groups=channels导致错误）
        )
        
        # DCN 权重参数（深度可分离: 每个输出通道只连接一个输入通道）
        self.weight = nn.Parameter(
            torch.Tensor(channels, 1, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(channels))
        
        # 初始化权重
        nn.init.kaiming_uniform_(self.weight, a=0)
        self.bias.data.zero_()
        
    def forward(self, x):
        num_points = self.kernel_size ** 3
        
        # 预测 offset 和 mask
        offset_mask = self.offset_mask_conv(x)
        offset = offset_mask[:, :3 * num_points, :, :, :]
        mask = torch.sigmoid(offset_mask[:, 3 * num_points:, :, :, :])
        
        # 执行深度可变形卷积（groups=channels）
        x = deform_conv3d(
            x, self.weight, offset, mask, self.bias,
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            dilation=(1, 1, 1),
            groups=self.channels  # 深度可分离
        )
        
        return x


class CEDBlock(nn.Module):
    """
    CEDNet 基础块 - 3D 版本（全DCN增强）
    
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║  CEDBlock 是 CEDNet 的核心组件，本版本使用DCN增强                           ║
    ║                                                                            ║
    ║  设计思想：                                                                 ║
    ║  1. Depthwise DCN - 深度可变形卷积，自适应空间特征变形                      ║
    ║  2. Inverted Bottleneck - 先扩张后压缩（C→4C→C），增强表达能力             ║
    ║  3. Residual Connection - 残差连接，缓解梯度消失                           ║
    ║                                                                            ║
    ║  结构流程：                                                                 ║
    ║    输入(C) → DW-DCN(空间特征,自适应) → LayerNorm → PWConv(4C,扩张)         ║
    ║           → GELU → PWConv(C,还原) → LayerScale → DropPath                  ║
    ║           → 残差相加 → 输出(C)                                             ║
    ║                                                                            ║
    ║  DCN改进：dwconv → DepthwiseDeformConv3d                                   ║
    ╚════════════════════════════════════════════════════════════════════════════╝

    参数:
        dim: 输入/输出通道数
        drop_path: DropPath 概率（随机深度正则化）
        layer_scale_init_value: Layer Scale 初始值
        kernel_size: DWConv 卷积核大小（通常为3）
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=3):
        super().__init__()

        # ===== 1. Depthwise 可变形卷积 - 自适应空间特征提取 =====
        # 替换原来的标准 dwconv，使用 DCN 增强几何变形感知能力
        self.dwconv = DepthwiseDeformConv3d(
            dim, kernel_size=kernel_size, 
            stride=1, padding=kernel_size // 2
        )
        
        # ===== 2. 归一化层 =====
        self.norm = LayerNorm3d(dim)

        # ===== 3. Pointwise 卷积 (MLP) - 通道特征混合 =====
        # 1×1×1 卷积，只在通道维度上混合信息
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)  # 扩张4倍
        self.act = nn.GELU()                                    # 平滑的非线性激活
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)  # 压缩回原始通道数

        # ===== 4. Layer Scale - 可学习的缩放因子 =====
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        # ===== 5. DropPath - 随机深度正则化 =====
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        前向传播
        
        x: (B, C, D, H, W) 输入特征
        return: (B, C, D, H, W) 输出特征
        """
        input = x  # 保存输入，用于残差连接
        
        # 特征提取流程
        x = self.dwconv(x)      # 1. 深度可变形卷积 - 自适应提取空间特征
        x = self.norm(x)        # 2. 归一化 - 稳定训练
        x = self.pwconv1(x)     # 3. 通道扩张 (C → 4C)
        x = self.act(x)         # 4. 非线性激活
        x = self.pwconv2(x)     # 5. 通道压缩 (4C → C)

        # Layer Scale - 缩放残差分支
        if self.gamma is not None:
            x = self.gamma[None, :, None, None, None] * x

        # 残差连接: output = input + 特征变换(input)
        x = input + self.drop_path(x)
        return x


# ===== 编码器-解码器组件 =====

class Encoder(nn.Module):
    """
    编码器模块 - 使用 CEDBlock (DCN增强) 进行特征提取
    
    ╔═══════════════════════════════════════════════════════════════╗
    ║  本模块是 CEDBlock 的主要使用位置，全面使用DCN              ║
    ║  通过堆叠多个 DCN-CEDBlock，逐层提取更抽象的语义特征        ║
    ║                                                               ║
    ║  DCN-CEDBlock 分布：                                          ║
    ║  • Layer1 (32³): 2个 DCN-CEDBlock                           ║
    ║  • Downsample1: DCN 下采样（32³→16³）                       ║
    ║  • Layer2 (16³): 4个 DCN-CEDBlock（主力层）                 ║
    ║  • Downsample2: DCN 下采样（16³→8³）                        ║
    ║  • Layer3 (8³):  2个 DCN-CEDBlock                           ║
    ║                                                               ║
    ║  总计每个Encoder：8个DCN-CEDBlock + 2个DCN下采样 = 10次DCN  ║
    ║       3个Stage × 10 = 30次DCN（编码器部分）                 ║
    ╚═══════════════════════════════════════════════════════════════╝

    结构: 3个层级，2次下采样（全DCN）
    输入: (dim0, S, S, S)
    输出: (dim0, S, S, S), (dim1, S/2, S/2, S/2), (dim2, S/4, S/4, S/4)
    
    参数:
        dims: 各层通道数 [32, 64, 128]
        blocks: 各层CEDBlock数量 [2, 4, 2]
        dp_rates: 各CEDBlock的DropPath率
    """

    def __init__(self, dims=[32, 64, 128], blocks=[2, 4, 2], dp_rates=None):
        super().__init__()

        if dp_rates is None:
            dp_rates = [0.] * sum(blocks)

        # ===== Layer 1: 不下采样，堆叠 DCN-CEDBlock =====
        self.layer1 = nn.Sequential(
            *[CEDBlock(dims[0], drop_path=dp_rates[i]) for i in range(blocks[0])]
        )

        # ===== Downsample 1: DCN 下采样 32³ → 16³ =====
        # 使用 kernel_size=3 而不是 2，DCN 需要足够的感受野
        self.down1 = nn.Sequential(
            LayerNorm3d(dims[0]),
            DeformConv3dBlock(dims[0], dims[1], kernel_size=3, stride=2, padding=1, use_gelu=True)
        )

        # ===== Layer 2: 不下采样，堆叠 DCN-CEDBlock =====
        start_idx = blocks[0]
        self.layer2 = nn.Sequential(
            *[CEDBlock(dims[1], drop_path=dp_rates[start_idx + i])
              for i in range(blocks[1])]
        )

        # ===== Downsample 2: DCN 下采样 16³ → 8³ =====
        self.down2 = nn.Sequential(
            LayerNorm3d(dims[1]),
            DeformConv3dBlock(dims[1], dims[2], kernel_size=3, stride=2, padding=1, use_gelu=True)
        )

        # ===== Layer 3: 不下采样，堆叠 DCN-CEDBlock =====
        start_idx = blocks[0] + blocks[1]
        self.layer3 = nn.Sequential(
            *[CEDBlock(dims[2], drop_path=dp_rates[start_idx + i], kernel_size=3)
              for i in range(blocks[2])]
        )

    def forward(self, x):
        # Layer 1
        c3 = self.layer1(x)

        # Downsample + Layer 2
        x = self.down1(c3)
        c4 = self.layer2(x)

        # Downsample + Layer 3
        x = self.down2(c4)
        c5 = self.layer3(x)

        return c3, c4, c5


class Decoder(nn.Module):
    """
    解码器模块
    
    注意：上采样不使用DCN，因为DCN主要适用于下采样和特征提取
          上采样使用标准卷积 + 三线性插值
    
    结构: 2次上采样，融合编码器特征
    """

    def __init__(self, dims=[32, 64, 128]):
        super().__init__()

        # Upsample 1: c5 → c4
        self.up1 = nn.Sequential(
            nn.Conv3d(dims[2], dims[1], kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )
        self.norm1 = LayerNorm3d(dims[1])

        # Upsample 2: c4 → c3
        self.up2 = nn.Sequential(
            nn.Conv3d(dims[1], dims[0], kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )
        self.norm2 = LayerNorm3d(dims[0])

    def forward(self, c3, c4, c5):
        # 融合 c5 → c4
        c4_refined = self.norm1(c4 + self.up1(c5))

        # 融合 c4 → c3
        c3_refined = self.norm2(c3 + self.up2(c4_refined))

        return c3_refined, c4_refined, c5


# ===== PPM 金字塔池化模块 =====

class PPM3D(nn.Module):
    """
    3D Pyramid Pooling Module

    在最深层特征上进行多尺度池化，聚合全局上下文信息
    使用标准卷积（1×1×1），不需要DCN
    """

    def __init__(self, in_channels, out_channels=32, pool_scales=(1, 2, 3)):
        super().__init__()

        self.pool_scales = pool_scales
        self.stages = nn.ModuleList()

        for scale in pool_scales:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool3d(scale),
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                LayerNorm3d(out_channels),
                nn.GELU()
            ))

    def forward(self, x):
        """
        x: (B, C, D, H, W) 例如 (B, 128, 8, 8, 8)
        return: [(B, out_channels, D, H, W), ...] for each scale
        """
        ppm_outs = []
        target_size = x.shape[2:]  # (D, H, W)

        for stage in self.stages:
            # 池化 + 卷积
            pooled = stage(x)
            # 上采样回原始尺寸
            upsampled = F.interpolate(pooled, size=target_size,
                                      mode='trilinear', align_corners=False)
            ppm_outs.append(upsampled)

        return ppm_outs


# ===== UPerNet 3D 多尺度融合 =====

class UPerNet3D(nn.Module):
    """
    3D UPerNet 解码头

    功能：
    1. PPM 增强最深层特征
    2. 侧边卷积统一通道数（使用标准1×1卷积，不需要DCN）
    3. 上采样到统一分辨率
    4. 多尺度拼接融合
    """

    def __init__(self, in_channels=[16, 32, 64, 128], channels=64,
                 pool_scales=(1, 2, 3), ppm_channels=32):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels

        # PPM 模块（只对最深层 c5）
        self.ppm = PPM3D(in_channels[-1], ppm_channels, pool_scales)

        # PPM Bottleneck
        ppm_out_channels = in_channels[-1] + len(pool_scales) * ppm_channels
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv3d(ppm_out_channels, in_channels[-1], kernel_size=3, padding=1),
            LayerNorm3d(in_channels[-1]),
            nn.GELU()
        )

        # 侧边卷积（统一通道数）
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv3d(in_ch, channels, kernel_size=1),
                LayerNorm3d(channels),
                nn.GELU()
            ))

        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * len(in_channels), channels, kernel_size=3, padding=1),
            LayerNorm3d(channels),
            nn.GELU()
        )

    def forward(self, features):
        """
        features: [c2, c3, c4, c5]
        c2: (B, 16, 64, 64, 64)
        c3: (B, 32, 32, 32, 32)
        c4: (B, 64, 16, 16, 16)
        c5: (B, 128, 8, 8, 8)
        """
        # 1. PPM 增强 c5
        c5 = features[-1]
        ppm_outs = self.ppm(c5)
        c5_enhanced = torch.cat([c5] + ppm_outs, dim=1)
        c5_enhanced = self.ppm_bottleneck(c5_enhanced)

        # 更新特征列表
        features = list(features[:-1]) + [c5_enhanced]

        # 2. 侧边卷积（统一通道数）
        laterals = []
        for i, (feature, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            laterals.append(lateral_conv(feature))

        # 3. 上采样到统一分辨率（c2 的尺寸）
        target_size = laterals[0].shape[2:]  # (64, 64, 64)

        upsampled = []
        for i, lateral in enumerate(laterals):
            if lateral.shape[2:] != target_size:
                upsampled.append(F.interpolate(lateral, size=target_size,
                                               mode='trilinear', align_corners=False))
            else:
                upsampled.append(lateral)

        # 4. 拼接所有特征
        fused = torch.cat(upsampled, dim=1)

        # 5. 融合卷积
        output = self.fusion_conv(fused)

        return output


# ===== 主模型 =====

class FaultSeg3D(nn.Module):
    """
    FaultSeg3D - 基于 CEDNet 架构的 3D 断层分割网络（全DCN增强版）

    架构流程:
    输入 (1, 128³)
      → Stem (DCN): 128³ → 64³
      → P2: 特征提取(DCN-CEDBlock), 保存c2, DCN下采样到32³
      → Stage 1: 编码(全DCN) + 解码
      → Stage 2: 编码(全DCN) + 解码
      → Stage 3: 编码(全DCN，最终特征)
      → UPerNet: PPM + 多尺度融合 → 64³
      → Head: 上采样 + 分类 → 128³
    输出 (2, 128³)

    DCN使用统计:
      - Stem: 2次DCN
      - P2下采样: 1次DCN
      - 3个Stage编码器: 每个10次DCN × 3 = 30次DCN
      - 总计: 约33次DCN操作

    参数:
        n_channels: 输入通道数 (默认1)
        n_classes: 输出类别数 (默认2)
        dims: 各阶段通道数配置
        depths: 各阶段block数量
        num_stages: 级联Stage数量
        drop_path_rate: DropPath率
        upernet_channels: UPerNet统一通道数
        ppm_scales: PPM池化尺度
    """

    def __init__(
            self,
            n_channels=1,
            n_classes=2,
            dims=[16, 32, 64, 128],  # 通道配置
            depths=[2, 2, 4, 2],  # block数量
            num_stages=3,  # Stage数量
            drop_path_rate=0.1,  # DropPath率
            upernet_channels=64,  # UPerNet通道数
            ppm_scales=(1, 2, 3),  # PPM池化尺度
            layer_scale_init_value=1e-6,
            use_checkpointing=True
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_stages = num_stages
        self.use_checkpointing = use_checkpointing

        # ===== Stem: 128³ → 64³ (全DCN) =====
        self.stem = nn.Sequential(
            DeformConv3dBlock(n_channels, dims[0] // 2, kernel_size=3, stride=1, padding=1, use_gelu=True),
            DeformConv3dBlock(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1, use_gelu=True),
        )

        # ===== P2 Stage: 64³ 特征提取 + 下采样到 32³ =====
        # DropPath 率线性增加
        total_blocks = depths[0] + num_stages * sum(depths[1:])
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # P2 blocks (DCN-CEDBlock)
        p2_blocks = []
        for i in range(depths[0]):
            p2_blocks.append(CEDBlock(dims[0], drop_path=dp_rates[i],
                                      layer_scale_init_value=layer_scale_init_value))
        self.p2_blocks = nn.Sequential(*p2_blocks)

        # P2 下采样 (DCN)
        self.p2_downsample = DeformConv3dBlock(
            dims[0], dims[1], kernel_size=3, stride=2, padding=1, use_gelu=True
        )

        # ===== 级联 Stages (全DCN编码器) =====
        self.stages = nn.ModuleList()
        cur_dp = depths[0]

        for stage_idx in range(num_stages):
            # 计算当前 stage 的 DropPath 率
            stage_dp_rates = dp_rates[cur_dp: cur_dp + sum(depths[1:])]

            # 编码器（全DCN）
            encoder = Encoder(
                dims=dims[1:],  # [32, 64, 128]
                blocks=depths[1:],  # [2, 4, 2]
                dp_rates=stage_dp_rates
            )

            # 解码器（最后一个 Stage 不需要解码器）
            if stage_idx < num_stages - 1:
                decoder = Decoder(dims=dims[1:])
                self.stages.append(nn.ModuleList([encoder, decoder]))
            else:
                self.stages.append(nn.ModuleList([encoder]))

            cur_dp += sum(depths[1:])

        # ===== UPerNet 多尺度融合 =====
        # 输入: [c2, c3, c4, c5] = [(16,64³), (32,32³), (64,16³), (128,8³)]
        self.upernet = UPerNet3D(
            in_channels=dims,
            channels=upernet_channels,
            pool_scales=ppm_scales,
            ppm_channels=32
        )

        # ===== 分割头 =====
        self.seg_head = nn.Sequential(
            nn.Conv3d(upernet_channels, upernet_channels, kernel_size=3, padding=1),
            LayerNorm3d(upernet_channels),
            nn.GELU(),
            nn.Conv3d(upernet_channels, n_classes, kernel_size=1)
        )

        self.softmax = nn.Softmax(dim=1)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm3d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def _checkpoint_enabled(self):
        return self.use_checkpointing and self.training and torch.is_grad_enabled()

    def _run_checkpointed(self, function, *inputs):
        if not self._checkpoint_enabled():
            return function(*inputs)
        if _CHECKPOINT_SUPPORTS_USE_REENTRANT:
            return checkpoint(function, *inputs, use_reentrant=True)
        return checkpoint(function, *inputs)

    @staticmethod
    def _make_checkpoint_input(x):
        if x.requires_grad:
            return x
        return x.detach().requires_grad_(True)

    def forward(self, x):
        """
        前向传播

        x: (B, 1, 128, 128, 128)
        return: (B, 2, 128, 128, 128)
        """
        # Stem: 128³ → 64³ (DCN)
        if self._checkpoint_enabled():
            x = self._make_checkpoint_input(x)
        for stem_layer in self.stem:
            x = self._run_checkpointed(stem_layer, x)

        # P2: 特征提取(DCN-CEDBlock) + 保存 c2
        c2 = self._run_checkpointed(self.p2_blocks, x)  # (B, 16, 64, 64, 64)
        x = self._run_checkpointed(self.p2_downsample, c2)  # (B, 32, 32, 32, 32) - DCN下采样

        # 级联 Stages（全DCN编码器）
        for stage_idx, stage in enumerate(self.stages):
            if len(stage) == 2:  # 有解码器
                encoder, decoder = stage
                c3, c4, c5 = self._run_checkpointed(encoder, x)
                x, _, _ = self._run_checkpointed(decoder, c3, c4, c5)  # 恢复到输入尺寸
            else:  # 最后一个 Stage，只有编码器
                encoder = stage[0]
                c3, c4, c5 = self._run_checkpointed(encoder, x)

        # 特征金字塔
        features = [c2, c3, c4, c5]
        # c2: (B, 16, 64, 64, 64)
        # c3: (B, 32, 32, 32, 32)
        # c4: (B, 64, 16, 16, 16)
        # c5: (B, 128, 8, 8, 8)

        # UPerNet 多尺度融合: → 64³
        def _upernet_forward(c2_feature, c3_feature, c4_feature, c5_feature):
            return self.upernet([c2_feature, c3_feature, c4_feature, c5_feature])

        fused = self._run_checkpointed(_upernet_forward, c2, c3, c4, c5)  # (B, 64, 64, 64, 64)

        # 上采样到原始分辨率: 64³ → 128³
        def _head_forward(fused_feature):
            upsampled = F.interpolate(fused_feature, scale_factor=2, mode='trilinear',
                                      align_corners=False)  # (B, 64, 128, 128, 128)
            logits = self.seg_head(upsampled)  # (B, 2, 128, 128, 128)
            return self.softmax(logits)

        # 分类
        output = self._run_checkpointed(_head_forward, fused)

        return output

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'FaultSeg3D (CEDNet + 全DCN)',
            'total_params': f'{total_params / 1e6:.2f}M',
            'trainable_params': f'{trainable_params / 1e6:.2f}M',
            'dims': [16, 32, 64, 128],
            'num_stages': self.num_stages,
            'dcn_enabled': 'Stem + P2 + All Encoders (Full DCN)',
            'activation_checkpointing': self.use_checkpointing,
        }


# ===== 测试代码 =====

if __name__ == '__main__':
    print("=" * 70)
    print("FaultSeg3D (CEDNet 全DCN增强架构) - 模型测试")
    print("=" * 70)

    # 创建模型
    model = FaultSeg3D(n_channels=1, n_classes=2)

    # 打印模型信息
    info = model.get_model_info()
    print("\n模型信息:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # 测试前向传播
    print("\n" + "=" * 70)
    print("前向传播测试")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 创建测试输入
    x = torch.randn(1, 1, 128, 128, 128).to(device)

    print(f"\n输入尺寸: {x.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(x)

    print(f"输出尺寸: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"概率和检查: {output[0, :, 64, 64, 64].sum():.4f} (应该≈1.0)")

    # 验证输出
    assert output.shape == (1, 2, 128, 128, 128), "输出尺寸错误!"
    assert torch.allclose(output.sum(dim=1), torch.ones_like(output.sum(dim=1)),
                          atol=1e-5), "Softmax 概率和不为1!"

    print("\n✓ 所有测试通过!")

    # 显存占用
    if torch.cuda.is_available():
        print(f"\n显存占用: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    # DCN使用统计
    print("\n" + "=" * 70)
    print("DCN使用统计")
    print("=" * 70)
    print("""
DCN操作分布:
  • Stem: 2次 DeformConv3dBlock
  • P2下采样: 1次 DeformConv3dBlock
  • P2 CEDBlock: 2次 DepthwiseDeformConv3d
  • Stage 1 编码器: 
    - Layer1: 2次 DepthwiseDeformConv3d (CEDBlock)
    - Down1: 1次 DeformConv3dBlock
    - Layer2: 4次 DepthwiseDeformConv3d (CEDBlock)
    - Down2: 1次 DeformConv3dBlock
    - Layer3: 2次 DepthwiseDeformConv3d (CEDBlock)
    小计: 10次DCN
  • Stage 2 编码器: 10次DCN
  • Stage 3 编码器: 10次DCN
  
总计: 2(Stem) + 1(P2下采样) + 2(P2块) + 30(3个Stage编码器) = 35次DCN操作

对比CEDNet_Unet.py (无DCN): 0次DCN
对比CEDNet_Unet_DCN.py: 仅3次DCN (Stem + P2)
本模型: 35次DCN (全面增强)
    """)

    # 详细的中间特征尺寸（调试用）
    print("\n" + "=" * 70)
    print("详细特征尺寸流动（调试信息）")
    print("=" * 70)

    # 手动追踪特征尺寸
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, 128, 128, 128).to(device)

        # Stem
        x_stem = model.stem(x)
        print(f"Stem 输出 (DCN): {x_stem.shape}")

        # P2
        c2 = model.p2_blocks(x_stem)
        print(f"P2 块输出 (c2, DCN-CEDBlock): {c2.shape}")

        x_p2 = model.p2_downsample(c2)
        print(f"P2 下采样输出 (DCN): {x_p2.shape}")

        # Stage 1 Encoder
        c3, c4, c5 = model.stages[0][0](x_p2)
        print(f"Stage1 Encoder输出 (全DCN):")
        print(f"  c3: {c3.shape}")
        print(f"  c4: {c4.shape}")
        print(f"  c5: {c5.shape}")

        # Stage 1 Decoder
        if len(model.stages[0]) == 2:
            dec_out, _, _ = model.stages[0][1](c3, c4, c5)
            print(f"Stage1 Decoder输出: {dec_out.shape}")

        # 特征金字塔
        print(f"\n最终特征金字塔:")
        print(f"  c2: (16, 64, 64, 64)   - 1/2  分辨率")
        print(f"  c3: (32, 32, 32, 32)   - 1/4  分辨率")
        print(f"  c4: (64, 16, 16, 16)   - 1/8  分辨率")
        print(f"  c5: (128, 8, 8, 8)     - 1/16 分辨率")

    print("\n" + "=" * 70)
    print("🎉 FaultSeg3D (CEDNet 全DCN) 测试完成!")
    print("=" * 70)

    # 使用说明
    print("\n" + "=" * 70)
    print("使用说明")
    print("=" * 70)
    print("""
训练命令:
python main.py --mode train --exp CEDNet_FullDCN_400_50 \\
    --train_path ./data/data_3D_400/train/ \\
    --valid_path ./data/data_3D_400/valid/ \\
    --epochs 50 \\
    --batch_size 2 \\
    --loss_func dice_plus_ce \\
    --optim_lr 1e-4

预测命令:
python main.py --mode pred --exp CEDNet_FullDCN_400_50 \\
    --pred_data_name f3 \\
    --pred_path /path/to/data/

推荐超参数:
  - batch_size: 2 (显存足够可以用4，全DCN显存占用较大)
  - learning_rate: 1e-4
  - epochs: 50-100
  - loss: dice_plus_ce
  
注意事项:
  1. 全DCN版本参数量和计算量较大，训练时间较长
  2. 建议从较小的learning rate开始 (1e-4)
  3. 显存占用约为标准版本的1.5-2倍
  4. 特别适合断层几何形变复杂的场景
    """)

