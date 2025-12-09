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


class InputModule(nn.Module):
    """
    输入模块 - 将128³输入降采样到32³
    
    结构:
    1. Conv3d 5×5×5, stride=2: 128³ → 64³
    2. Conv3d 3×3×3, stride=1: 中间层处理
    3. Conv3d 3×3×3, stride=2: 64³ → 32³
    
    所有卷积后接 BatchNorm3d + ReLU
    使用卷积下采样而非池化，保留边缘信息
    """
    
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        
        # 第一次下采样: 128³ → 64³
        # 使用 padding=2 保证输出尺寸正确
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 中间层: 64³ → 64³
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 第二次下采样: 64³ → 32³
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        x: (B, 1, 128, 128, 128)
        return: (B, base_channels, 32, 32, 32)
        """
        x = self.conv1(x)  # 128³ → 64³
        x = self.conv2(x)  # 64³ → 64³
        x = self.conv3(x)  # 64³ → 32³
        return x


class Encoder(nn.Module):
    """
    编码器 - 从输入模块的32³特征中提取多尺度特征
    
    结构: 3个阶段，逐步提取抽象特征
    每个阶段包含多个卷积块
    """
    
    def __init__(self, in_channels=16, channels=[16, 32, 64]):
        super().__init__()
        
        self.channels = channels
        
        # Stage 1: 32³, 通道数不变或增加
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: 32³ → 16³ (可选下采样)
        self.stage2_conv = nn.Sequential(
            nn.Conv3d(channels[0], channels[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(channels[1]),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(channels[1], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels[1], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[1]),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: 16³ → 8³
        self.stage3_conv = nn.Sequential(
            nn.Conv3d(channels[1], channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(channels[2]),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv3d(channels[2], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels[2], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[2]),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        x: (B, in_channels, 32, 32, 32)
        return:
            c1: (B, channels[0], 32, 32, 32) - Stage 1输出
            c2: (B, channels[1], 16, 16, 16) - Stage 2输出
            c3: (B, channels[2], 8, 8, 8) - Stage 3输出
        """
        c1 = self.stage1(x)  # 32³
        
        x = self.stage2_conv(c1)  # 32³ → 16³
        c2 = self.stage2(x)  # 16³
        
        x = self.stage3_conv(c2)  # 16³ → 8³
        c3 = self.stage3(x)  # 8³
        
        return c1, c2, c3


class ForwardDecoder(nn.Module):
    """
    前向解码器 - 从低分辨率特征解码到高分辨率
    
    从编码器的输出开始，逐步上采样恢复空间分辨率
    使用最近邻插值进行上采样
    """
    
    def __init__(self, channels=[64, 32, 16], out_channels=16):
        super().__init__()
        
        # 从 c3 (8³) 上采样到 c2 (16³)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(channels[2], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[1]),
            nn.ReLU(inplace=True)
        )
        
        # 从 16³ 上采样到 32³
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(channels[1], out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合（如果有跳跃连接）
        self.fusion1 = nn.Sequential(
            nn.Conv3d(channels[1] * 2, channels[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[1]),
            nn.ReLU(inplace=True)
        )
        
        self.fusion2 = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, c1, c2, c3):
        """
        c1: (B, channels[0], 32, 32, 32) - 编码器Stage 1
        c2: (B, channels[1], 16, 16, 16) - 编码器Stage 2
        c3: (B, channels[2], 8, 8, 8) - 编码器Stage 3
        
        return:
            out: (B, out_channels, 32, 32, 32) - 32³高分辨率特征
        """
        # 从 c3 上采样并与 c2 融合
        up_c3 = self.up1(c3)  # 8³ → 16³
        # 注意：如果 c2 和 up_c3 通道数不同，需要调整
        if up_c3.shape[1] == c2.shape[1]:
            fused_c2 = torch.cat([c2, up_c3], dim=1)
            fused_c2 = self.fusion1(fused_c2)  # (B, channels[1], 16, 16, 16)
        else:
            # 如果通道数不同，只使用上采样的特征
            fused_c2 = up_c3
        
        # 从 16³ 上采样到 32³并与 c1 融合
        up_c2 = self.up2(fused_c2)  # 16³ → 32³
        if up_c2.shape[1] == c1.shape[1]:
            fused_c1 = torch.cat([c1, up_c2], dim=1)
            out = self.fusion2(fused_c1)  # (B, out_channels, 32, 32, 32)
        else:
            out = up_c2
        
        return out


class BackwardDecoder(nn.Module):
    """
    反向解码器 - 从高分辨率特征解码到低分辨率
    
    从前向解码器的高分辨率特征开始，逐步下采样
    生成低分辨率特征图，形成互补的特征表示
    """
    
    def __init__(self, in_channels=16, channels=[16, 32, 64]):
        super().__init__()
        
        # 从 32³ 下采样到 16³
        self.down1 = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[1]),
            nn.ReLU(inplace=True)
        )
        
        # 从 16³ 下采样到 8³
        self.down2 = nn.Sequential(
            nn.Conv3d(channels[1], channels[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[2]),
            nn.ReLU(inplace=True)
        )
        
        # 从 8³ 上采样回 32³（用于输出模块融合）
        self.up_back = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8³ → 16³
            nn.Conv3d(channels[2], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16³ → 32³
            nn.Conv3d(channels[1], in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        x: (B, in_channels, 32, 32, 32) - 前向解码器输出
        
        return:
            out: (B, in_channels, 32, 32, 32) - 反向解码后恢复到32³
        """
        # 下采样路径
        x = self.down1(x)  # 32³ → 16³
        x = self.down2(x)  # 16³ → 8³
        
        # 上采样回32³
        out = self.up_back(x)  # 8³ → 16³ → 32³
        
        return out


class OutputModule(nn.Module):
    """
    输出模块 - 融合双解码器和编码器的特征，恢复到原始分辨率

    融合三个特征图:
    1. 前向解码器输出
    2. 反向解码器输出
    3. 编码器第三阶段输出（上采样到32³）

    然后两次上采样恢复到128³
    """

    def __init__(self, in_channels=48, base_channels=16, encoder_stage3_channels=64, n_classes=2):
        super().__init__()

        # 将编码器第三阶段特征上采样到32³
        self.encoder_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='nearest'),  # 8³ → 32³
            nn.Conv3d(encoder_stage3_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 第一次上采样: 32³ → 64³
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        # 第二次上采样: 64³ → 128³
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        # 最终输出层: 1×1×1卷积，输出n_classes个通道
        self.final_conv = nn.Conv3d(in_channels // 4, n_classes, kernel_size=1)
    
    def forward(self, forward_out, backward_out, encoder_stage3):
        """
        forward_out: (B, base_channels, 32, 32, 32) - 前向解码器输出
        backward_out: (B, base_channels, 32, 32, 32) - 反向解码器输出
        encoder_stage3: (B, encoder_stage3_channels, 8, 8, 8) - 编码器第三阶段

        return:
            out: (B, n_classes, 128, 128, 128) - 最终分类logits
        """
        # 将编码器第三阶段特征上采样到32³
        encoder_32 = self.encoder_up(encoder_stage3)  # 8³ → 32³

        # 拼接三个特征图 (假设都是16通道)
        # 如果通道数不同，需要调整
        if forward_out.shape[1] == backward_out.shape[1] == encoder_32.shape[1]:
            fused = torch.cat([forward_out, backward_out, encoder_32], dim=1)  # (B, 48, 32, 32, 32)
        else:
            # 如果通道数不一致，先统一通道数
            min_channels = min(forward_out.shape[1], backward_out.shape[1], encoder_32.shape[1])
            forward_out = forward_out[:, :min_channels, :, :, :]
            backward_out = backward_out[:, :min_channels, :, :, :]
            encoder_32 = encoder_32[:, :min_channels, :, :, :]
            fused = torch.cat([forward_out, backward_out, encoder_32], dim=1)

        # 两次上采样: 32³ → 64³ → 128³
        x = self.up1(fused)  # 32³ → 64³
        x = self.up2(x)  # 64³ → 128³

        # 最终输出logits（不包含Sigmoid，让训练代码决定是否使用Softmax）
        out = self.final_conv(x)  # (B, n_classes, 128, 128, 128)

        return out


class FaultSeg3D(nn.Module):
    """
    Fault3DNnet - 轻量级3D地震断层检测网络，采用双向解码结构

    架构流程:
    输入 (1, 128³)
      → 输入模块: 128³ → 32³
      → 编码器: 提取多尺度特征 [32³, 16³, 8³]
      → 前向解码器: 8³ → 16³ → 32³
      → 反向解码器: 32³ → 16³ → 8³ → 32³
      → 输出模块: 融合三个特征，32³ → 64³ → 128³
    输出 (n_classes, 128³) - 分类概率图（Softmax输出）
    """

    def __init__(self, in_channels=1, n_classes=2, base_channels=16,
                 encoder_channels=[16, 32, 64]):
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.encoder_channels = encoder_channels

        # 输入模块
        self.input_module = InputModule(in_channels, base_channels)

        # 编码器
        self.encoder = Encoder(base_channels, encoder_channels)

        # 前向解码器
        self.forward_decoder = ForwardDecoder(encoder_channels, base_channels)

        # 反向解码器
        self.backward_decoder = BackwardDecoder(base_channels, encoder_channels)

        # 输出模块
        self.output_module = OutputModule(
            in_channels=base_channels * 3,  # 48通道
            base_channels=base_channels,
            encoder_stage3_channels=encoder_channels[2],
            n_classes=n_classes
        )

        # Softmax层（用于输出概率）
        self.softmax = nn.Softmax(dim=1)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播

        x: (B, 1, 128, 128, 128)
        return: (B, n_classes, 128, 128, 128) - 分类logits（需要Softmax）
        """
        # 输入模块: 128³ → 32³
        x_input = self.input_module(x)  # (B, 16, 32, 32, 32)

        # 编码器: 提取多尺度特征
        c1, c2, c3 = self.encoder(x_input)
        # c1: (B, 16, 32, 32, 32)
        # c2: (B, 32, 16, 16, 16)
        # c3: (B, 64, 8, 8, 8)

        # 前向解码器: 从低分辨率解码到高分辨率
        forward_out = self.forward_decoder(c1, c2, c3)  # (B, 16, 32, 32, 32)

        # 反向解码器: 从高分辨率解码到低分辨率
        backward_out = self.backward_decoder(forward_out)  # (B, 16, 32, 32, 32)

        # 输出模块: 融合并恢复到128³
        logits = self.output_module(forward_out, backward_out, c3)  # (B, n_classes, 128, 128, 128)

        # 应用Softmax得到概率
        output = self.softmax(logits)  # (B, n_classes, 128, 128, 128)

        return output


# ===== 测试代码 =====

if __name__ == '__main__':
    # 查看网络参数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(in_channels=1, n_classes=2).to(device)
    summary(net, input_size=(1, 128, 128, 128))

# ================================================================
# Total params: 626,571
# Trainable params: 626,571
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 17179869475.38
# Params size (MB): 2.39
# Estimated Total Size (MB): 17179869485.77
# ----------------------------------------------------------------