
"""
Fault3DNnet - A lightweight 3D seismic fault detection network with bidirectional decoding

论文: Fault3DNnet: A lightweight 3D seismic fault detection network with bidirectional decoding
      (Tang et al., 2024)

模型架构：
- 输入模块 (Input Module): 128³ → 32³，两次下采样
- 编码器 (Encoder): 从32³特征进一步提取
- 双解码器 (Dual Decoders):
  * Forward Decoder: 低分辨率 → 高分辨率
  * Backward Decoder: 高分辨率 → 低分辨率
- 输出模块 (Output Module): 融合特征并恢复到128³

输入: (B, 1, 128, 128, 128) - 地震数据
输出: (B, 1, 128, 128, 128) - 断层概率图 (Sigmoid输出)
"""
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 辅助函数：卷积块 ---
def conv_block_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.ReLU):
    """标准的 3D 卷积 + BatchNorm + 激活函数 块"""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_channels),
        activation(inplace=True)
    )


class DoubleConv(nn.Module):
    """双层卷积块，参考 faultseg3d_.py 的实现，增强特征提取能力"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class InputModule(nn.Module):
    """
    输入模块：实现 128³ → 32³ (两次下采样)

    通道配置遵循：1 → 8 → 16 → 32
    """

    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        # 第一次下采样: 128³ → 64³ (通道: 1 → 8)
        self.conv1 = conv_block_3d(in_channels, 8, kernel_size=5, stride=2, padding=2)

        # 中间层: 64³ → 64³ (通道: 8 → 16) - 使用DoubleConv增强特征提取
        self.conv2 = DoubleConv(8, 16)

        # 第二次下采样: 64³ → 32³ (通道: 16 → 32)
        # 注意：这里使用 stride=2 的卷积进行下采样，然后使用 DoubleConv 增强特征
        self.conv3_down = conv_block_3d(16, base_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = DoubleConv(base_channels, base_channels)  # 下采样后增强特征提取

    def forward(self, x):
        """
        x: (B, 1, 128, 128, 128)
        return: (B, 32, 32, 32, 32)
        """
        x = self.conv1(x)  # 128³ → 64³
        x = self.conv2(x)  # 64³ → 64³
        x = self.conv3_down(x)  # 64³ → 32³ (下采样)
        x = self.conv3(x)  # 32³ → 32³ (增强特征提取)
        return x


class Encoder(nn.Module):
    """
    编码器：在 32³ 尺寸上进行特征提取 (瓶颈层)

    注意：此模块不包含下采样 (stride=1)，保持 32³ 尺寸。
    编码器包含两层（E1和E2/E3），需要输出F_E1用于解码器的跳跃连接。
    """

    def __init__(self, in_channels=32, out_channels=32):
        super().__init__()

        # E1: 编码器第一阶段 (接收32³特征，无尺寸变化) - 使用DoubleConv增强特征提取
        self.block1 = DoubleConv(in_channels, out_channels)

        # E2/E3: 编码器第二阶段/瓶颈层 (编码器最终输出FE3) - 使用DoubleConv增强特征提取
        self.block2 = DoubleConv(out_channels, out_channels)

        self.block3 = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        """
        x: (B, 32, 32, 32, 32) - 输入模块输出
        return:
            e1: (B, 32, 32, 32, 32) - 编码器输入（用于跳跃连接）
            e2: (B, 32, 32, 32, 32) - 编码器第一阶段输出
            e3: (B, 32, 32, 32, 32) - 编码器第二阶段输出
            e4: (B, 32, 32, 32, 32) - 编码器瓶颈层输出
        """
        e1 = x  # (B, 32, 32, 32, 32)
        e2 = self.block1(x)  # (B, 32, 32, 32, 32)
        e3 = self.block2(e1)  # (B, 32, 32, 32, 32)
        e4 = self.block3(e2)  # (B, 32, 32, 32, 32)

        return e1, e2, e3, e4


class ForwardDecoder(nn.Module):
    """
    前向解码器 (Forward Decoder): 从低分辨率到高分辨率的解码
    """

    def __init__(self, in_channels=32, out_channels=16):
        super().__init__()

        # 解码器需要处理拼接后的通道数
        # e1+e2: 32+32=64 → 32
        self.block1 = DoubleConv(in_channels * 2, in_channels)  # 64 → 32
        # x+e3: 32+32=64 → 32
        self.block2 = DoubleConv(in_channels * 2, in_channels)  # 64 → 32
        # x+e4: 32+32=64 → 32
        self.block3 = DoubleConv(in_channels * 2, in_channels)  # 64 → 32
        
        # 最终输出层：32 → 16
        self.final_conv = DoubleConv(in_channels, out_channels)  # 32 → 16

    def forward(self, e1, e2, e3, e4):
        # e1, e2, e3, e4 都是 (B, 32, 32, 32, 32)
        x = torch.cat([e1, e2], dim=1)  # (B, 64, 32, 32, 32)
        x = self.block1(x)  # (B, 32, 32, 32, 32)

        x = torch.cat([x, e3], dim=1)  # (B, 64, 32, 32, 32)
        x = self.block2(x)  # (B, 32, 32, 32, 32)

        x = torch.cat([x, e4], dim=1)  # (B, 64, 32, 32, 32)
        x = self.block3(x)  # (B, 32, 32, 32, 32)
        
        # 最终输出 16 通道
        x = self.final_conv(x)  # (B, 16, 32, 32, 32)

        return x


class BackwardDecoder(nn.Module):
    """
    反向解码器 (Backward Decoder): 处理相反方向的特征信息，提供互补性
    """

    def __init__(self, in_channels=32, out_channels=16):
        super().__init__()

        # 解码器需要处理拼接后的通道数
        # e3+e4: 32+32=64 → 32
        self.block3 = DoubleConv(in_channels * 2, in_channels)  # 64 → 32
        # e2+x: 32+32=64 → 32
        self.block2 = DoubleConv(in_channels * 2, in_channels)  # 64 → 32
        # e4+x: 32+32=64 → 32
        self.block1 = DoubleConv(in_channels * 2, in_channels)  # 64 → 32
        
        # 最终输出层：32 → 16
        self.final_conv = DoubleConv(in_channels, out_channels)  # 32 → 16

    def forward(self, e1, e2, e3, e4):
        # e1, e2, e3, e4 都是 (B, 32, 32, 32, 32)
        x = torch.cat([e3, e4], dim=1)  # (B, 64, 32, 32, 32)
        x = self.block3(x)  # (B, 32, 32, 32, 32)

        x = torch.cat([e2, x], dim=1)  # (B, 64, 32, 32, 32)
        x = self.block2(x)  # (B, 32, 32, 32, 32)

        x = torch.cat([e4, x], dim=1)  # (B, 64, 32, 32, 32)
        x = self.block1(x)  # (B, 32, 32, 32, 32)
        
        # 最终输出 16 通道
        x = self.final_conv(x)  # (B, 16, 32, 32, 32)
        
        return x


class OutputModule(nn.Module):
    """
    输出模块：融合三个 32³ 特征，并两次上采样恢复到 128³
    """

    def __init__(self, n_classes=2):
        super().__init__()

        # 三个输入：FD (16) + BD (16) + e4 (32) = 64 通道
        in_channels = 64

        # 第一次上采样: 32³ → 64³ (通道: 64 → 32) - 使用trilinear插值，参考faultseg3d_.py
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            DoubleConv(in_channels, in_channels // 2)  # 64 → 32
        )

        # 第二次上采样: 64³ → 128³ (通道: 32 → 16) - 使用trilinear插值
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            DoubleConv(in_channels // 2, 16)  # 32 → 16
        )

        # 最终输出层: 1×1×1卷积，输出n_classes个通道 (16 → n_classes)
        self.final_conv = nn.Conv3d(16, n_classes, kernel_size=1)

    def forward(self, forward_out, backward_out, encoder_bottleneck):
        """
        forward_out: (B, 16, 32, 32, 32)
        backward_out: (B, 16, 32, 32, 32)
        encoder_bottleneck: (B, 32, 32, 32, 32) - e4
        """


        # 2. 拼接三个特征图 (16+16+32 = 64 通道)
        fused = torch.cat([forward_out, backward_out, encoder_bottleneck], dim=1)  # (B, 64, 32, 32, 32)

        # 3. 两次上采样: 32³ → 64³ → 128³
        x = self.up1(fused)  # 32³ → 64³ (64 → 32)
        x = self.up2(x)  # 64³ → 128³ (32 → 16)

        # 4. 最终输出 logits
        logits = self.final_conv(x)  # (B, n_classes, 128, 128, 128)

        return logits


class FaultSeg3D(nn.Module):
    """
    Fault3DNnet - 轻量级3D地震断层检测网络
    """

    def __init__(self, in_channels=1, n_classes=2, base_channels=32):
        super().__init__()

        # --- 通道配置 ---
        # Input/Encoder: 32 channels
        # Decoders Output: 16 channels

        # 1. 输入模块: 128³ → 32³
        self.input_module = InputModule(in_channels, base_channels)  # 输出 32 通道

        # 2. 编码器: 32³ → 32³ (输出e1, e2, e3, e4)
        self.encoder = Encoder(base_channels, base_channels)  # 输出 e1, e2, e3, e4，都是 32 通道

        # 3. 双解码器: 独立且并行 (都输出 16 通道)
        # Forward Decoder: 32 → 16，使用跳跃连接F_E1
        self.forward_decoder = ForwardDecoder(base_channels, base_channels // 2)

        # Backward Decoder: 32 → 16，使用跳跃连接F_E1（权重独立）
        self.backward_decoder = BackwardDecoder(base_channels, base_channels // 2)

        # 4. 输出模块
        self.output_module = OutputModule(n_classes)

        # 最终激活层：参考 faultseg3d_.py 的实现，使用 Softmax
        # 当 n_classes=2 时，使用 Softmax 进行归一化，与 U-Net 保持一致
        self.softmax = nn.Softmax(dim=1)

        # 权重初始化（可选）
        self._init_weights()

    def _init_weights(self):
        """权重初始化 (Kaiming/He initialization for ReLU)"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Conv3d后接BatchNorm时，bias=False
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        x: (B, 1, 128, 128, 128)
        return: (B, n_classes, 128, 128, 128) - 最终 logits 或概率
        """
        # 1. 输入模块: 128³ → 32³
        x_input = self.input_module(x)  # (B, 32, 32, 32, 32)

        # 2. 编码器: 输出e1, e2, e3, e4
        e1, e2, e3, e4 = self.encoder(x_input)  # 都是 (B, 32, 32, 32, 32)

        # 3. 双解码器: 并行处理，使用e1, e2, e3, e4作为跳跃连接
        forward_out = self.forward_decoder(e1, e2, e3, e4)  # (B, 16, 32, 32, 32)
        backward_out = self.backward_decoder(e1, e2, e3, e4)  # (B, 16, 32, 32, 32)

        # 4. 输出模块: 融合并恢复到 128³
        logits = self.output_module(forward_out, backward_out, e4)  # (B, n_classes, 128, 128, 128)

        # 5. 最终激活 - 参考 faultseg3d_.py，使用 Softmax
        outputs = self.softmax(logits)

        return outputs


# ===== 测试代码和参数总结 =====

if __name__ == '__main__':
    # 查看网络参数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))

# ================================================================
# Total params: 700,338
# Trainable params: 700,338
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 8385452.00
# Params size (MB): 2.67
# Estimated Total Size (MB): 8385462.67
# ----------------------------------------------------------------



