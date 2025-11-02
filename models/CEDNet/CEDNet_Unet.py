"""

FaultSeg3D - 基于 CEDNet 架构的 3D 地震断层分割网络

完整实现了 CEDNet 的级联编码-解码结构，包括：
- Stem + P2 初始特征提取
- 3 个级联 Stage (编码-解码对)
- PPM 金字塔池化模块
- UPerNet 多尺度融合
- 分割头

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


class CEDBlock(nn.Module):
    """
    CEDNet 基础块 - 3D 版本

    结构: DWConv → Norm → PWConv(扩张4倍) → GELU → PWConv(还原) → 残差连接
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=3):
        super().__init__()

        # Depthwise 卷积
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size,
                               padding=kernel_size//2, groups=dim)
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
    3D UPerNet 解码头 - UNet风格逐步融合

    功能：
    1. PPM 增强最深层特征 C5
    2. 逐步融合: C5→C4, C4→C3, C3→C2
    3. 每一步: 上采样 + 相加 + 卷积融合

    流程:
        C5 (128, 8³) → PPM增强 → 1×1 Conv → (64, 8³)
          ↓ 上采样×2 + 融合
        C4 (64, 16³) → 1×1 Conv → + → Conv 3×3×3 → (64, 16³)
          ↓ 上采样×2 + 融合
        C3 (32, 32³) → 1×1 Conv → + → Conv 3×3×3 → (64, 32³)
          ↓ 上采样×2 + 融合
        C2 (16, 64³) → 1×1 Conv → + → Conv 3×3×3 → (64, 64³)
    """
    def __init__(self, in_channels=[16, 32, 64, 128], channels=64,
                 pool_scales=(1, 2, 3), ppm_channels=32):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels

        # PPM 模块（只对最深层 c5）
        self.ppm = PPM3D(in_channels[-1], ppm_channels, pool_scales)

        # PPM Bottleneck: 增强后统一通道数
        ppm_out_channels = in_channels[-1] + len(pool_scales) * ppm_channels
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv3d(ppm_out_channels, channels, kernel_size=3, padding=1),
            LayerNorm3d(channels),
            nn.GELU()
        )

        # 侧边卷积（统一通道数到 channels）
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels[:-1]:  # 不包括c5，c5已经在ppm_bottleneck处理
            self.lateral_convs.append(nn.Sequential(
                nn.Conv3d(in_ch, channels, kernel_size=1),
                LayerNorm3d(channels),
                nn.GELU()
            ))

        # 上采样模块（逐步融合）
        self.upsample_convs = nn.ModuleList()
        for i in range(len(in_channels) - 1):  # 3次上采样融合
            self.upsample_convs.append(nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                LayerNorm3d(channels),
                nn.GELU()
            ))

    def forward(self, features):
        """
        features: [c2, c3, c4, c5]
        c2: (B, 16, 64, 64, 64)
        c3: (B, 32, 32, 32, 32)
        c4: (B, 64, 16, 16, 16)
        c5: (B, 128, 8, 8, 8)

        返回: (B, channels, 64, 64, 64)
        """
        c2, c3, c4, c5 = features

        # 1. PPM 增强 c5 并统一通道数
        ppm_outs = self.ppm(c5)
        c5_enhanced = torch.cat([c5] + ppm_outs, dim=1)
        c5_fused = self.ppm_bottleneck(c5_enhanced)  # (B, channels, 8, 8, 8)

        # 2. 侧边卷积: 统一通道数
        c2_lateral = self.lateral_convs[0](c2)  # (B, channels, 64, 64, 64)
        c3_lateral = self.lateral_convs[1](c3)  # (B, channels, 32, 32, 32)
        c4_lateral = self.lateral_convs[2](c4)  # (B, channels, 16, 16, 16)

        # 3. 逐步融合 (从深到浅): C5 → C4 → C3 → C2

        # C5 → C4 融合
        c5_up = F.interpolate(c5_fused, scale_factor=2, mode='trilinear',
                             align_corners=False)  # (B, channels, 16, 16, 16)
        c4_fused = c4_lateral + c5_up  # 残差相加
        c4_fused = self.upsample_convs[0](c4_fused)  # 融合卷积

        # C4 → C3 融合
        c4_up = F.interpolate(c4_fused, scale_factor=2, mode='trilinear',
                             align_corners=False)  # (B, channels, 32, 32, 32)
        c3_fused = c3_lateral + c4_up
        c3_fused = self.upsample_convs[1](c3_fused)

        # C3 → C2 融合
        c3_up = F.interpolate(c3_fused, scale_factor=2, mode='trilinear',
                             align_corners=False)  # (B, channels, 64, 64, 64)
        c2_fused = c2_lateral + c3_up
        c2_fused = self.upsample_convs[2](c2_fused)

        return c2_fused  # (B, channels, 64, 64, 64)


# ===== 主模型 =====

class FaultSeg3D(nn.Module):
    """
    FaultSeg3D - 基于 CEDNet 架构的 3D 断层分割网络

    架构流程:
    输入 (1, 128³)
      → Stem: 128³ → 64³
      → P2: 特征提取, 保存c2, 下采样到32³
      → Stage 1: 编码(32³→16³→8³) + 解码(8³→16³→32³)
      → Stage 2: 编码-解码
      → Stage 3: 编码(最终特征)
      → UPerNet: PPM + 多尺度融合 → 64³
      → Head: 上采样 + 分类 → 128³
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
        dims=[16, 32, 64, 128],          # 通道配置
        depths=[2, 2, 4, 2],              # block数量
        num_stages=3,                     # Stage数量
        drop_path_rate=0.1,               # DropPath率
        upernet_channels=64,              # UPerNet通道数
        ppm_scales=(1, 2, 3),            # PPM池化尺度
        layer_scale_init_value=1e-6
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_stages = num_stages

        # ===== Stem: 128³ → 64³ =====
        self.stem = nn.Sequential(
            nn.Conv3d(n_channels, dims[0]//2, kernel_size=3, stride=1, padding=1),
            LayerNorm3d(dims[0]//2),
            nn.GELU(),
            nn.Conv3d(dims[0]//2, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm3d(dims[0]),
            nn.GELU(),
        )

        # ===== P2 Stage: 64³ 特征提取 + 下采样到 32³ =====
        # DropPath 率线性增加
        total_blocks = depths[0] + num_stages * sum(depths[1:])
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # P2 blocks
        p2_blocks = []
        for i in range(depths[0]):
            p2_blocks.append(CEDBlock(dims[0], drop_path=dp_rates[i],
                                     layer_scale_init_value=layer_scale_init_value))
        self.p2_blocks = nn.Sequential(*p2_blocks)

        # P2 下采样
        self.p2_downsample = nn.Sequential(
            LayerNorm3d(dims[0]),
            nn.Conv3d(dims[0], dims[1], kernel_size=2, stride=2)
        )

        # ===== 级联 Stages =====
        self.stages = nn.ModuleList()
        cur_dp = depths[0]

        for stage_idx in range(num_stages):
            # 计算当前 stage 的 DropPath 率
            stage_dp_rates = dp_rates[cur_dp : cur_dp + sum(depths[1:])]

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

        # ===== UPerNet 多尺度融合 =====
        # 输入: [c2, c3, c4, c5] = [(16,64³), (32,32³), (64,16³), (128,8³)]
        self.upernet = UPerNet3D(
            in_channels=dims,
            channels=upernet_channels,
            pool_scales=ppm_scales,
            ppm_channels=32
        )

        # ===== 分割头（渐进式特征恢复）=====
        # 方案1: 简单上采样 + 卷积
        # self.seg_head = nn.Sequential(
        #     nn.Conv3d(upernet_channels, upernet_channels, kernel_size=3, padding=1),
        #     LayerNorm3d(upernet_channels),
        #     nn.GELU(),
        #     nn.Conv3d(upernet_channels, n_classes, kernel_size=1)
        # )

        # 方案2: 渐进式上采样（64³ → 128³）
        self.seg_head = nn.Sequential(
            # 第一阶段：特征精炼
            nn.Conv3d(upernet_channels, upernet_channels, kernel_size=3, padding=1),
            LayerNorm3d(upernet_channels),
            nn.GELU(),

            # 第二阶段：上采样前的降维
            nn.Conv3d(upernet_channels, upernet_channels // 2, kernel_size=3, padding=1),
            LayerNorm3d(upernet_channels // 2),
            nn.GELU(),
        )

        # 最终分类层（在上采样之后）
        self.classifier = nn.Conv3d(upernet_channels // 2, n_classes, kernel_size=1)

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
        # Stem: 128³ → 64³
        x = self.stem(x)  # (B, 16, 64, 64, 64)

        # P2: 特征提取 + 保存 c2
        c2 = self.p2_blocks(x)  # (B, 16, 64, 64, 64)
        x = self.p2_downsample(c2)  # (B, 32, 32, 32, 32)

        # 级联 Stages
        for stage_idx, stage in enumerate(self.stages):
            if len(stage) == 2:  # 有解码器
                encoder, decoder = stage
                c3, c4, c5 = encoder(x)
                x, _, _ = decoder(c3, c4, c5)  # 恢复到输入尺寸
            else:  # 最后一个 Stage，只有编码器
                encoder = stage[0]
                c3, c4, c5 = encoder(x)

        # 特征金字塔
        features = [c2, c3, c4, c5]
        # c2: (B, 16, 64, 64, 64)
        # c3: (B, 32, 32, 32, 32)
        # c4: (B, 64, 16, 16, 16)
        # c5: (B, 128, 8, 8, 8)

        # UPerNet 逐步多尺度融合: C5→C4→C3→C2 → 64³
        fused = self.upernet(features)  # (B, 64, 64, 64, 64)

        # 分割头：特征精炼
        refined = self.seg_head(fused)  # (B, 32, 64, 64, 64)

        # 上采样到原始分辨率: 64³ → 128³
        upsampled = F.interpolate(refined, scale_factor=2, mode='trilinear',
                                 align_corners=False)  # (B, 32, 128, 128, 128)

        # 最终分类
        logits = self.classifier(upsampled)  # (B, 2, 128, 128, 128)
        output = self.softmax(logits)

        return output

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'FaultSeg3D (CEDNet)',
            'total_params': f'{total_params / 1e6:.2f}M',
            'trainable_params': f'{trainable_params / 1e6:.2f}M',
            'dims': [16, 32, 64, 128],
            'num_stages': self.num_stages,
        }


# ===== 测试代码 =====

if __name__ == '__main__':
    print("="*70)
    print("FaultSeg3D (CEDNet 3D架构) - 模型测试")
    print("="*70)

    # 创建模型
    model = FaultSeg3D(n_channels=1, n_classes=2)

    # 打印模型信息
    info = model.get_model_info()
    print("\n模型信息:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # 测试前向传播
    print("\n" + "="*70)
    print("前向传播测试")
    print("="*70)

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
        print(f"\n显存占用: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # 详细的中间特征尺寸（调试用）
    print("\n" + "="*70)
    print("详细特征尺寸流动（调试信息）")
    print("="*70)

    # 手动追踪特征尺寸
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, 128, 128, 128).to(device)

        # Stem
        x_stem = model.stem(x)
        print(f"Stem 输出: {x_stem.shape}")

        # P2
        c2 = model.p2_blocks(x_stem)
        print(f"P2 块输出 (c2): {c2.shape}")

        x_p2 = model.p2_downsample(c2)
        print(f"P2 下采样输出: {x_p2.shape}")

        # Stage 1 Encoder
        c3, c4, c5 = model.stages[0][0](x_p2)
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
        print(f"  c2: (16, 64, 64, 64)   - 1/2  分辨率")
        print(f"  c3: (32, 32, 32, 32)   - 1/4  分辨率")
        print(f"  c4: (64, 16, 16, 16)   - 1/8  分辨率")
        print(f"  c5: (128, 8, 8, 8)     - 1/16 分辨率")

    print("\n" + "="*70)
    print("🎉 FaultSeg3D (CEDNet) 测试完成!")
    print("="*70)

    # 使用说明
    print("\n" + "="*70)
    print("使用说明")
    print("="*70)
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
║                    FaultSeg3D (CEDNet-UNet 3D 架构)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

输入: (B, 1, 128×128×128) 地震数据
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STEM 模块                                       │
│  ┌────────────────────┐        ┌────────────────────┐                       │
│  │ Conv3d(1→8)        │   →    │ Conv3d(8→16)       │                       │
│  │ 3×3×3, stride=1    │        │ 3×3×3, stride=2    │                       │
│  │ + LayerNorm + GELU │        │ + LayerNorm + GELU │                       │
│  └────────────────────┘        └────────────────────┘                       │
│       128³                             64³                                  │
└─────────────────────────────────────────────────────────────────────────────┘
  │
  ▼ (B, 16, 64³)
┌─────────────────────────────────────────────────────────────────────────────┐
│                           P2 Stage (特征提取)                                 │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  CEDBlock(16) × 2                                       │                │
│  │  (DWConv → Norm → PWConv 4× → GELU → PWConv → Residual)│                 │
│  └─────────────────────────────────────────────────────────┘                │
│       │                                                                     │
│       ├───────────────────────────────────────┐ 保存 c2 (16, 64³)            │
│       │                                        │                            │
│       ▼                                        │                            │
│  ┌─────────────────┐                          │                             │
│  │ P2 Downsample   │                          │                             │
│  │ Conv 2×2×2      │                          │                             │
│  │ stride=2 (16→32)│                          │                             │
│  └─────────────────┘                          │                             │
└───────│──────────────────────────────────────────────────────────────────────┘
        │
        ▼ (B, 32, 32³)
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
        │  特征金字塔: [c2, c3, c4, c5]
        │              ↓    ↓    ↓    ↓
        │             16   32   64  128 (通道数)
        │             64³  32³  16³  8³ (空间尺寸)
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          UPerNet 多尺度融合                                   │
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
│  │  2. 逐步融合 (FPN 风格)                                       │             │
│  │                                                             │             │
│  │     c2 (16,64³) → Lateral Conv → 64 ─────────——─┐           │             │
│  │                                                 │           │             │
│  │     c3 (32,32³) → Lateral Conv → 64 ───────┐    │           │             │
│  │                                              │  │           │             │
│  │     c4 (64,16³) → Lateral Conv → 64 ────┐    │  │           │             │
│  │                                           │  │  │           │             │
│  │     c5_fused (64,8³)                      │  │  │           │             │
│  │            ↓ Up×2                         │  │  │           │             │
│  │     c4 (64,16³) + ────────────────────────┘  │  │           │             │
│  │            ↓ Fusion Conv                     │  │           │             │
│  │            ↓ Up×2                            │  │           │             │
│  │     c3 (64,32³) + ───────────────────────────┘  │           │             │
│  │            ↓ Fusion Conv                        │           │             │
│  │            ↓ Up×2                               │           │             │
│  │     c2 (64,64³) + ──────────────────────────────┘           │             │
│  │            ↓ Fusion Conv                                    │             │
│  │         输出 (64, 64³)                                      │             │
│  └────────────────────────────────────────────────────────────┘             │
└───────┬───────────────────────────────────────────────────────────────────────┘
        │
        ▼ (B, 64, 64³)
┌─────────────────────────────────────────────────────────────────────────────┐
│                          分割头 (Segmentation Head)                           │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  特征精炼:                                              │                │
│  │    Conv3d(64→64) + LayerNorm + GELU                    │                │
│  │    Conv3d(64→32) + LayerNorm + GELU                    │                │
│  └─────────────────────────────────────────────────────────┘                │
│         ↓ (B, 32, 64³)                                                       │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  上采样: 64³ → 128³ (scale_factor=2, trilinear)        │                │
│  └─────────────────────────────────────────────────────────┘                │
│         ↓ (B, 32, 128³)                                                      │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  分类器: Conv3d(32→2, 1×1×1)                           │                │
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
║ ✓ 级联编码-解码结构 (3个Stage)                                               ║
║ ✓ CEDBlock: DWConv + MLP + Residual + DropPath                               ║
║ ✓ 多尺度特征金字塔: 64³, 32³, 16³, 8³                                       ║
║ ✓ PPM 金字塔池化: 全局上下文聚合                                             ║
║ ✓ UPerNet 风格融合: 逐步上采样 + 侧边连接                                    ║
║ ✓ 参数量: ~3-5M                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""