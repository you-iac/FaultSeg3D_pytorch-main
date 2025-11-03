
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           FaultSeg3D (CEDNet-UNet Hybrid) - 简化版架构                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

输入: (B, 1, 128³)
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段 1: UNet 风格初始特征 (First Conv)                                       │
│  DoubleConv: 1 → 16 通道, 保持 128³ 分辨率                                   │
│  → 保存 c1 (16, 128³)                                                        │
└───────┬─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段 2: P2 Stage (合并的下采样+特征提取)                                     │
│                                                                              │
│  第一步：Conv 下采样 128³ → 64³ (16 → 16)                                    │
│         ↓                                                                    │
│  第二步：CEDBlock × 2 特征提取 (16, 64³)                                     │
│         → 保存 c2 (16, 64³)                                                  │
│         ↓                                                                    │
│  第三步：Conv 下采样 64³ → 32³ (16 → 32)                                     │
└───────┬─────────────────────────────────────────────────────────────────────┘
        │
        ▼ (32, 32³)
        │
┌───────┴─────────────────────────────────────────────────────────────────────┐
│  阶段 3-5: 级联 Stages (CEDNet 主干)                                          │
│  Stage 1, 2: 编码-解码                                                        │
│  Stage 3: 仅编码，输出最终特征                                                │
│  → 生成 c3 (32, 32³), c4 (64, 16³), c5 (128, 8³)                            │
└───────┬─────────────────────────────────────────────────────────────────────┘
        │
        ▼ 特征金字塔: [c1, c2, c3, c4, c5]
        │
┌───────┴─────────────────────────────────────────────────────────────────────┐
│  阶段 6: UPerNet 多尺度融合 (5层，通道数逐步递减)                             │
│  PPM(c5) → 逐步融合: C5(64)→C4(48)→C3(32)→C2(24)→C1(16)                    │
│  → 输出 (16, 128³)                                                           │
└───────┬─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────┴─────────────────────────────────────────────────────────────────────┐
│  阶段 7: 分类头 (保持 128³)                                                   │
│  直接分类: 16 → 2 通道                                                        │
│  Softmax                                                                     │
└───────┬─────────────────────────────────────────────────────────────────────┘
        │
        ▼
输出: (B, 2, 128³)
FaultSeg3D - 基于 CEDNet 架构的 3D 地震断层分割网络 (UNet Hybrid 版本)

混合架构设计：
- UNet 风格的初始特征提取 (保持 128³ 分辨率，生成 c1)
- P2 Stage: 下采样(128³→64³) → 特征提取 → 下采样(64³→32³)
- 3 个级联 Stage (编码-解码对)
- PPM 金字塔池化模块
- UPerNet 多尺度融合 (5层: C5→C4→C3→C2→C1)
- 分割头 (直接在 128³ 分辨率输出)

架构特点:
1. 在 128³ 分辨率提取初始特征 (类似 UNet)
2. P2 统一处理下采样和特征提取
3. 通过 CEDNet 主干提取多尺度特征
4. UPerNet 逐步融合回 128³ 分辨率（通道数逐步递减）
5. 渐进式降维: 64 → 48 → 32 → 24 → 16 → 2 通道

输入: (B, 1, 128, 128, 128) - 地震数据
输出: (B, 2, 128, 128, 128) - 二分类分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DoubleConv(nn.Module):
    """UNet 风格的双卷积块"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class CEDBlock(nn.Module):
    """
    CEDNet 基础块 - 3D 版本

    结构: DWConv → Norm → PWConv(扩张4倍) → GELU → PWConv(还原) → 残差连接
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=3):
        super().__init__()

        # Depthwise 卷积
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm3d(dim)

        # Pointwise 卷积 (MLP)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)

        # Layer Scale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

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


# ===== 编码器-解码器组件 =====

class Encoder(nn.Module):
    """
    编码器模块

    结构: 3个层级，2次下采样
    输入: (dim0, S, S, S)
    输出: (dim0, S, S, S), (dim1, S/2, S/2, S/2), (dim2, S/4, S/4, S/4)
    """

    def __init__(self, dims=[32, 64, 128], blocks=[2, 4, 2], dp_rates=None):
        super().__init__()

        if dp_rates is None:
            dp_rates = [0.] * sum(blocks)

        # Layer 1: 不下采样
        self.layer1 = nn.Sequential(
            *[CEDBlock(dims[0], drop_path=dp_rates[i]) for i in range(blocks[0])]
        )

        # Downsample 1
        self.down1 = nn.Sequential(
            LayerNorm3d(dims[0]),
            nn.Conv3d(dims[0], dims[1], kernel_size=2, stride=2)
        )

        # Layer 2: 不下采样
        start_idx = blocks[0]
        self.layer2 = nn.Sequential(
            *[CEDBlock(dims[1], drop_path=dp_rates[start_idx + i])
              for i in range(blocks[1])]
        )

        # Downsample 2
        self.down2 = nn.Sequential(
            LayerNorm3d(dims[1]),
            nn.Conv3d(dims[1], dims[2], kernel_size=2, stride=2)
        )

        # Layer 3: 不下采样（使用扩张卷积增强）
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


# ===== UPerNet 3D 多尺度融合（UNet风格逐步融合）=====

class UPerNet3D(nn.Module):
    """
    3D UPerNet 解码头 - UNet风格逐步融合（扩展到5层）

    功能：
    1. PPM 增强最深层特征 C5
    2. 逐步融合: C5→C4→C3→C2→C1
    3. 每一步: 上采样 + UNet拼接 + 卷积融合

    流程:
        C5 (128, 8³) → PPM增强 → 1×1 Conv → (64, 8³)
          ↓ 上采样×2 + UNet拼接
        C4 (64, 16³) → 1×1 Conv(48) → cat → Conv 3×3×3 → (48, 16³)
          ↓ 上采样×2 + UNet拼接
        C3 (32, 32³) → 1×1 Conv(32) → cat → Conv 3×3×3 → (32, 32³)
          ↓ 上采样×2 + UNet拼接
        C2 (16, 64³) → 1×1 Conv(24) → cat → Conv 3×3×3 → (24, 64³)
          ↓ 上采样×2 + UNet拼接
        C1 (16, 128³) → 1×1 Conv(16) → cat → Conv 3×3×3 → (16, 128³)
    """

    def __init__(self, in_channels=[16, 16, 32, 64, 128], 
                 pool_scales=(1, 2, 3), ppm_channels=32):
        super().__init__()

        self.in_channels = in_channels
        
        # 逐步递减的通道数配置: 64 → 48 → 32 → 24 → 16
        self.decode_channels = [64, 48, 32, 24, 16]

        # PPM 模块（只对最深层 c5）
        self.ppm = PPM3D(in_channels[-1], ppm_channels, pool_scales)

        # PPM Bottleneck: 增强后统一通道数
        ppm_out_channels = in_channels[-1] + len(pool_scales) * ppm_channels
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv3d(ppm_out_channels, self.decode_channels[0], kernel_size=3, padding=1),
            LayerNorm3d(self.decode_channels[0]),
            nn.GELU()
        )

        # 侧边卷积（统一通道数到对应的decode_channels）
        self.lateral_convs = nn.ModuleList()
        for i, in_ch in enumerate(in_channels[:-1]):  # 不包括c5，c5已经在ppm_bottleneck处理
            # c1, c2, c3, c4 分别对应 decode_channels[4], [3], [2], [1]
            out_ch = self.decode_channels[4 - i]  # 反向索引
            self.lateral_convs.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1),
                LayerNorm3d(out_ch),
                nn.GELU()
            ))

        # 上采样模块（UNet拼接风格融合，通道数逐步递减）
        self.upsample_convs = nn.ModuleList()
        for i in range(len(in_channels) - 1):  # 4次上采样融合
            # 拼接后的输入通道数 + 输出通道数（逐步递减）
            in_ch = self.decode_channels[i] + self.decode_channels[i + 1]
            out_ch = self.decode_channels[i + 1]
            self.upsample_convs.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                LayerNorm3d(out_ch),
                nn.GELU()
            ))

    def forward(self, features):
        """
        features: [c1, c2, c3, c4, c5]
        c1: (B, 16, 128, 128, 128)
        c2: (B, 16, 64, 64, 64)
        c3: (B, 32, 32, 32, 32)
        c4: (B, 64, 16, 16, 16)
        c5: (B, 128, 8, 8, 8)

        返回: (B, 16, 128, 128, 128) - 逐步递减到16通道
        """
        c1, c2, c3, c4, c5 = features

        # 1. PPM 增强 c5 并统一通道数
        ppm_outs = self.ppm(c5)
        c5_enhanced = torch.cat([c5] + ppm_outs, dim=1)
        c5_fused = self.ppm_bottleneck(c5_enhanced)  # (B, 64, 8, 8, 8)

        # 2. 侧边卷积: 统一通道数（逐步递减）
        c1_lateral = self.lateral_convs[0](c1)  # (B, 16, 128, 128, 128)
        c2_lateral = self.lateral_convs[1](c2)  # (B, 24, 64, 64, 64)
        c3_lateral = self.lateral_convs[2](c3)  # (B, 32, 32, 32, 32)
        c4_lateral = self.lateral_convs[3](c4)  # (B, 48, 16, 16, 16)

        # 3. 逐步融合 (从深到浅, UNet拼接风格, 通道数逐步递减): 
        #    C5(64) → C4(48) → C3(32) → C2(24) → C1(16)

        # C5 → C4 融合: 64 → 48
        c5_up = F.interpolate(c5_fused, scale_factor=2, mode='trilinear',
                              align_corners=False)  # (B, 64, 16, 16, 16)
        c4_fused = torch.cat([c4_lateral, c5_up], dim=1)  # UNet拼接 (B, 112, 16, 16, 16)
        c4_fused = self.upsample_convs[0](c4_fused)  # 融合卷积 (B, 48, 16, 16, 16)

        # C4 → C3 融合: 48 → 32
        c4_up = F.interpolate(c4_fused, scale_factor=2, mode='trilinear',
                              align_corners=False)  # (B, 48, 32, 32, 32)
        c3_fused = torch.cat([c3_lateral, c4_up], dim=1)  # UNet拼接 (B, 80, 32, 32, 32)
        c3_fused = self.upsample_convs[1](c3_fused)  # (B, 32, 32, 32, 32)

        # C3 → C2 融合: 32 → 24
        c3_up = F.interpolate(c3_fused, scale_factor=2, mode='trilinear',
                              align_corners=False)  # (B, 32, 64, 64, 64)
        c2_fused = torch.cat([c2_lateral, c3_up], dim=1)  # UNet拼接 (B, 56, 64, 64, 64)
        c2_fused = self.upsample_convs[2](c2_fused)  # (B, 24, 64, 64, 64)

        # C2 → C1 融合: 24 → 16
        c2_up = F.interpolate(c2_fused, scale_factor=2, mode='trilinear',
                              align_corners=False)  # (B, 24, 128, 128, 128)
        c1_fused = torch.cat([c1_lateral, c2_up], dim=1)  # UNet拼接 (B, 40, 128, 128, 128)
        c1_fused = self.upsample_convs[3](c1_fused)  # (B, 16, 128, 128, 128)

        return c1_fused  # (B, 16, 128, 128, 128)


# ===== 主模型 =====

class FaultSeg3D(nn.Module):
    """
    FaultSeg3D - 基于 CEDNet 架构的 3D 断层分割网络 (UNet Hybrid)

    架构流程:
    输入 (1, 128³)
      → First Conv: UNet风格特征提取, 保存c1 (128³)
      → P2: 下采样(128³→64³) → 特征提取, 保存c2 → 下采样(64³→32³)
      → Stage 1: 编码(32³→16³→8³) + 解码(8³→16³→32³)
      → Stage 2: 编码-解码
      → Stage 3: 编码(最终特征)
      → UPerNet: PPM + 多尺度融合(5层) → 128³
      → Head: 特征精炼(64→32→16) + 分类 → 128³
    输出 (2, 128³)

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
            layer_scale_init_value=1e-6
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_stages = num_stages

        # ===== 第一阶段: UNet 风格特征提取 (保持 128³) =====
        self.first_conv = DoubleConv(n_channels, dims[0])  # 1 → 16, 128³

        # ===== P2 Stage: 128³ → 64³ → 特征提取 → 32³ (合并原 Stem + P2) =====
        # DropPath 率线性增加
        total_blocks = depths[0] + num_stages * sum(depths[1:])
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # P2 第一步：下采样 128³ → 64³
        self.p2_downsample1 = nn.Sequential(
            nn.Conv3d(dims[0], dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm3d(dims[0]),
            nn.GELU(),
        )

        # P2 第二步：特征提取（64³）
        p2_blocks = []
        for i in range(depths[0]):
            p2_blocks.append(CEDBlock(dims[0], drop_path=dp_rates[i],
                                      layer_scale_init_value=layer_scale_init_value))
        self.p2_blocks = nn.Sequential(*p2_blocks)

        # P2 第三步：下采样 64³ → 32³
        self.p2_downsample2 = nn.Sequential(
            LayerNorm3d(dims[0]),
            nn.Conv3d(dims[0], dims[1], kernel_size=2, stride=2)
        )

        # ===== 级联 Stages =====
        self.stages = nn.ModuleList()
        cur_dp = depths[0]

        for stage_idx in range(num_stages):
            # 计算当前 stage 的 DropPath 率
            stage_dp_rates = dp_rates[cur_dp: cur_dp + sum(depths[1:])]

            # 编码器
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

        # ===== UPerNet 多尺度融合（逐步递减通道数）=====
        # 输入: [c1, c2, c3, c4, c5] = [(16,128³), (16,64³), (32,32³), (64,16³), (128,8³)]
        # 输出: (16, 128³) - 通道数从64逐步递减到16
        self.upernet = UPerNet3D(
            in_channels=[dims[0]] + dims,  # [16, 16, 32, 64, 128]
            pool_scales=ppm_scales,
            ppm_channels=32
        )

        # ===== 分割头（直接从 16 通道分类）=====
        # UPerNet已经输出16通道(128³)，直接分类即可
        self.classifier = nn.Conv3d(dims[0], n_classes, kernel_size=1)

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

    def forward(self, x):
        """
        前向传播

        x: (B, 1, 128, 128, 128)
        return: (B, 2, 128, 128, 128)
        """
        # 第一阶段: UNet 风格特征提取 (保持 128³)
        c1 = self.first_conv(x)  # (B, 16, 128, 128, 128)

        # P2 阶段 (合并的 Stem + P2)
        # 第一步：下采样 128³ → 64³
        x = self.p2_downsample1(c1)  # (B, 16, 64, 64, 64)

        # 第二步：特征提取 + 保存 c2
        c2 = self.p2_blocks(x)  # (B, 16, 64, 64, 64)

        # 第三步：下采样 64³ → 32³
        x = self.p2_downsample2(c2)  # (B, 32, 32, 32, 32)

        # 级联 Stages
        for stage_idx, stage in enumerate(self.stages):
            if len(stage) == 2:  # 有解码器
                encoder, decoder = stage
                c3, c4, c5 = encoder(x)
                x, _, _ = decoder(c3, c4, c5)  # 恢复到输入尺寸
            else:  # 最后一个 Stage，只有编码器
                encoder = stage[0]
                c3, c4, c5 = encoder(x)

        # 特征金字塔（5层）
        features = [c1, c2, c3, c4, c5]
        # c1: (B, 16, 128, 128, 128)
        # c2: (B, 16, 64, 64, 64)
        # c3: (B, 32, 32, 32, 32)
        # c4: (B, 64, 16, 16, 16)
        # c5: (B, 128, 8, 8, 8)

        # UPerNet 逐步多尺度融合: C5(64)→C4(48)→C3(32)→C2(24)→C1(16) → 128³
        fused = self.upernet(features)  # (B, 16, 128, 128, 128)

        # 直接分类 (16 → n_classes)
        logits = self.classifier(fused)  # (B, 2, 128, 128, 128)
        output = self.softmax(logits)

        return output

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'FaultSeg3D (CEDNet-UNet Hybrid)',
            'total_params': f'{total_params / 1e6:.2f}M',
            'trainable_params': f'{trainable_params / 1e6:.2f}M',
            'dims': [16, 32, 64, 128],
            'num_stages': self.num_stages,
            'architecture': 'UNet初始特征 + CEDNet主干 + UPerNet融合',
        }


# ===== 测试代码 =====

if __name__ == '__main__':
    print("=" * 70)
    print("FaultSeg3D (CEDNet-UNet Hybrid 架构) - 模型测试")
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

    # 详细的中间特征尺寸（调试用）
    print("\n" + "=" * 70)
    print("详细特征尺寸流动（调试信息）")
    print("=" * 70)

    # 手动追踪特征尺寸
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, 128, 128, 128).to(device)

        # First Conv
        c1 = model.first_conv(x)
        print(f"First Conv 输出 (c1): {c1.shape}")

        # P2 阶段（合并的 Stem + P2）
        x_p2_down1 = model.p2_downsample1(c1)
        print(f"P2 第一步下采样输出: {x_p2_down1.shape}")

        c2 = model.p2_blocks(x_p2_down1)
        print(f"P2 特征提取输出 (c2): {c2.shape}")

        x_p2_down2 = model.p2_downsample2(c2)
        print(f"P2 第二步下采样输出: {x_p2_down2.shape}")

        # Stage 1 Encoder
        c3, c4, c5 = model.stages[0][0](x_p2_down2)
        print(f"Stage1 Encoder输出:")
        print(f"  c3: {c3.shape}")
        print(f"  c4: {c4.shape}")
        print(f"  c5: {c5.shape}")

        # Stage 1 Decoder
        if len(model.stages[0]) == 2:
            dec_out, _, _ = model.stages[0][1](c3, c4, c5)
            print(f"Stage1 Decoder输出: {dec_out.shape}")

        # 特征金字塔
        print(f"\n最终特征金字塔:")
        print(f"  c1: (16, 128, 128, 128) - 原始分辨率")
        print(f"  c2: (16, 64, 64, 64)    - 1/2  分辨率")
        print(f"  c3: (32, 32, 32, 32)    - 1/4  分辨率")
        print(f"  c4: (64, 16, 16, 16)    - 1/8  分辨率")
        print(f"  c5: (128, 8, 8, 8)      - 1/16 分辨率")

    print("\n" + "=" * 70)
    print("🎉 FaultSeg3D (CEDNet-UNet Hybrid) 测试完成!")
    print("=" * 70)

    # 使用说明
    print("\n" + "=" * 70)
    print("使用说明")
    print("=" * 70)
    print("""
训练命令:
python main.py --mode train --exp CEDNet_400_50 \\
    --train_path ./data/data_3D_400/train/ \\
    --valid_path ./data/data_3D_400/valid/ \\
    --epochs 50 \\
    --batch_size 2 \\
    --loss_func dice_plus_ce \\
    --optim_lr 1e-4

预测命令:
python main.py --mode pred --exp CEDNet_400_50 \\
    --pred_data_name f3 \\
    --pred_path /path/to/data/

推荐超参数:
  - batch_size: 2 (显存足够可以用4)
  - learning_rate: 1e-4
  - epochs: 50-100
  - loss: dice_plus_ce
    """)

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              FaultSeg3D (CEDNet-UNet Hybrid 混合架构)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

输入: (B, 1, 128×128×128) 地震数据
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 第一阶段: UNet 风格特征提取 (128³)                             │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │  Conv3d(1→16, 3×3×3) + BN + ReLU                          │             │
│  │  Conv3d(16→16, 3×3×3) + BN + ReLU                         │             │
│  └────────────────────────────────────────────────────────────┘             │
│                        保持 128³ 分辨率                                       │
└───────┬──────────────────────────────────────────────────────────────────────┘
        │
        ▼ c1 (B, 16, 128³) ← 保存用于 UPerNet 融合
        │
┌───────┴──────────────────────────────────────────────────────────────────────┐
│                   P2 Stage (合并的下采样+特征提取)                             │
│                                                                              │
│  第一步：下采样 128³ → 64³                                                    │
│  ┌────────────────────┐                                                      │
│  │ Conv3d(16→16)      │                                                      │
│  │ 3×3×3, stride=2    │                                                      │
│  │ + LayerNorm + GELU │                                                      │
│  └────────────────────┘                                                      │
│         ↓ (B, 16, 64³)                                                       │
│                                                                              │
│  第二步：特征提取                                                             │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  CEDBlock(16) × 2                                       │                │
│  │  (DWConv → Norm → PWConv 4× → GELU → PWConv → Residual)│                │
│  └─────────────────────────────────────────────────────────┘                │
│         │                                                                    │
│         ├──────────────────────────┐ 保存 c2 (16, 64³)                      │
│         │                          │                                        │
│         ▼                          │                                        │
│                                    │                                        │
│  第三步：下采样 64³ → 32³          │                                         │
│  ┌─────────────────┐               │                                        │
│  │ Conv 2×2×2      │               │                                        │
│  │ stride=2 (16→32)│               │                                        │
│  └─────────────────┘               │                                        │
└───────│────────────────────────────┼────────────────────────────────────────┘
        │                            │
        ▼ (B, 32, 32³)                │
        │
┌───────┴─────────────────────────────────────────────────────────────────────┐
│                          级联 Stage 1 (编码-解码)                             │
│                                                                             │
│   编码器:                                                                    │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │ Layer1: CEDBlock(32)×2          →  c3 (32, 32³) ────┐            │      │
│   │    ↓ Downsample (2×2×2, 32→64)                      │            │      │
│   │ Layer2: CEDBlock(64)×4          →  c4 (64, 16³) ────┼───┐        │      │
│   │    ↓ Downsample (2×2×2, 64→128)                     │   │        │      │
│   │ Layer3: CEDBlock(128)×2         →  c5 (128, 8³) ────┼───┼───┐    │      │
│   └─────────────────────────────────────────────────────┼───┼───┼──-─┘      │
│                                                          │   │   │          │
│   解码器:                                                 │   │   │          │
│   ┌──────────────────────────────────────────────────────┼───┼───┼───┐      │
│   │                                         c5 (128, 8³) │   │   │   │      │
│   │                                            ↓ Up×2    │   │   │   │      │
│   │                                c4 (64,16³) + ────────┘   │   │   │      │
│   │                                            ↓ Up×2        │   │   │      │
│   │                                c3 (32,32³) + ────────────┘   │   │      │
│   │                                            ↓                 │   │      │
│   │                                 输出 (32, 32³) ───────── ─────┘   │      │
│   └──────────────────────────────────────────────────────────────────┘      │
└───────┬─────────────────────────────────────────────────────────────────────┘
        │
        ▼ (B, 32, 32³)
        │
┌───────┴─────────────────────────────────────────────────────────────────────┐
│                          级联 Stage 2 (编码-解码)                             │
│   [结构同 Stage 1]                                                           │
└───────┬─────────────────────────────────────────────────────────────────────┘
        │
        ▼ (B, 32, 32³)
        │
┌───────┴─────────────────────────────────────────────────────────────────────┐
│                          级联 Stage 3 (仅编码)                                │
│   编码器: 32³ → 16³ → 8³                                                     │
│   输出最终特征金字塔:                                                          │
│      c3: (32, 32³)                                                          │
│      c4: (64, 16³)                                                          │
│      c5: (128, 8³)                                                          │
└───────┬─────────────────────────────────────────────────────────────────────┘
        │
        │  特征金字塔: [c1, c2, c3, c4, c5]
        │              ↓    ↓    ↓    ↓    ↓
        │             16   16   32   64  128 (通道数)
        │            128³  64³  32³  16³  8³ (空间尺寸)
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UPerNet 多尺度融合 (扩展到5层)                             │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │  1. PPM 模块增强 c5                                         │             │
│  │     ┌─────┐  ┌─────┐  ┌─────┐                              │             │
│  │     │ 1×1 │  │ 2×2 │  │ 3×3 │  AdaptiveAvgPool3d           │             │
│  │     └──┬──┘  └──┬──┘  └──┬──┘                              │             │
│  │        └────────┴────────┘  Concat                         │             │
│  │              ↓                                             │             │
│  │         Bottleneck Conv (128+96 → 64)                      │             │
│  │              ↓ c5_fused (64, 8³)                           │             │
│  └────────────────────────────────────────────────────────────┘             │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │  2. 逐步融合 (UNet 拼接风格, 融合到 128³)                   │             │
│  │                                                             │             │
│  │     c1 (16,128³) → Lateral Conv → 16 ──────────────────┐   │             │
│  │                                                         │   │             │
│  │     c2 (16,64³)  → Lateral Conv → 24 ──────────────┐   │   │             │
│  │                                                     │   │   │             │
│  │     c3 (32,32³)  → Lateral Conv → 32 ──────────┐   │   │   │             │
│  │                                                 │   │   │   │             │
│  │     c4 (64,16³)  → Lateral Conv → 48 ──────┐   │   │   │   │             │
│  │                                             │   │   │   │   │             │
│  │     c5_fused (64,8³)                        │   │   │   │   │             │
│  │            ↓ Up×2                           │   │   │   │   │             │
│  │     c4 (48,16³) cat ────────────────────────┘   │   │   │   │             │
│  │            ↓ Fusion Conv (112→48)               │   │   │   │             │
│  │            ↓ Up×2                               │   │   │   │             │
│  │     c3 (32,32³) cat ────────────────────────────┘   │   │   │             │
│  │            ↓ Fusion Conv (80→32)                    │   │   │             │
│  │            ↓ Up×2                                   │   │   │             │
│  │     c2 (24,64³) cat ────────────────────────────────┘   │   │             │
│  │            ↓ Fusion Conv (56→24)                        │   │             │
│  │            ↓ Up×2                                       │   │             │
│  │     c1 (16,128³) cat ───────────────────────────────────┘   │             │
│  │            ↓ Fusion Conv (40→16)                            │             │
│  │         输出 (16, 128³)                                     │             │
│  └────────────────────────────────────────────────────────────┘             │
└───────┬───────────────────────────────────────────────────────────────────────┘
        │
        ▼ (B, 16, 128³)
┌─────────────────────────────────────────────────────────────────────────────┐
│                        分类头 (保持 128³)                                     │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  直接分类: Conv3d(16→2, 1×1×1)                          │                │
│  └─────────────────────────────────────────────────────────┘                │
│         ↓ (B, 2, 128³)                                                       │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  Softmax(dim=1)                                         │                │
│  └─────────────────────────────────────────────────────────┘                │
└───────┬───────────────────────────────────────────────────────────────────────┘
        │
        ▼
输出: (B, 2, 128×128×128) 分割概率图


╔══════════════════════════════════════════════════════════════════════════════╗
║                              关键特性总结                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ✓ UNet 风格初始特征提取 (在 128³ 分辨率，生成 c1)                            ║
║ ✓ 级联编码-解码结构 (3个Stage)                                               ║
║ ✓ CEDBlock: DWConv + MLP + Residual + DropPath                               ║
║ ✓ 5层特征金字塔: 128³, 64³, 32³, 16³, 8³                                   ║
║ ✓ PPM 金字塔池化: 全局上下文聚合                                             ║
║ ✓ UPerNet 风格融合: UNet拼接 + 通道数逐步递减 (64→48→32→24→16)             ║
║ ✓ 渐进式降维: 64 → 48 → 32 → 24 → 16 → 2 通道                               ║
║ ✓ 简洁分类头: 直接从16通道分类，无需额外特征精炼                             ║
║ ✓ 参数量: ~3-5M (优化后的通道设计)                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""