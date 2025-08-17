#实现一个3D Transformer模型嵌入到Unet的跳跃连接中，
# 输入输出保持大小一致一致为B C L H W，
# 最大输入为 2 16 128 128 128 ,最小为2 128 16 16 16。实现这个模块

from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer


class Transformer3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, num_layers=4):
        super(Transformer3D, self).__init__()

        # Transformer要求输入形状为 (seq_len, batch_size, feature_dim)
        # 所以我们需要把输入从 [B, C, L, H, W] 变为 [L*H*W, B, C] 这样才可以进入Transformer
        self.input_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.transformer = Transformer(d_model=out_channels,
                                       nhead=num_heads,
                                       num_encoder_layers=num_layers,
                                       num_decoder_layers=num_layers)

        self.output_proj = nn.Conv3d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x shape: [B, C, L, H, W]

        # 第一部分：卷积层
        B, C, L, H, W = x.shape
        x = self.input_proj(x)  # [B, out_channels, L, H, W]

        # Transformer需要的输入格式 (L*H*W, B, C)
        x = x.view(B, C, -1).permute(2, 0, 1)  # [L*H*W, B, C]

        # 使用Transformer进行处理
        x = self.transformer(x, x)  # 使用相同的输入作为encoder和decoder

        # 将输出变回 [B, out_channels, L, H, W]
        x = x.permute(1, 2, 0).view(B, C, L, H, W)

        # 使用卷积层得到最终输出
        x = self.output_proj(x)

        return x


class UNet3DWithTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, num_layers=4):
        super(UNet3DWithTransformer, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, kernel_size=3, padding=1)
        )

        # Transformer嵌入到跳跃连接
        self.transformer = Transformer3D(128, 128, num_heads=num_heads, num_layers=num_layers)

    def forward(self, x):
        # 编码器部分
        enc = self.encoder(x)

        # Transformer嵌入到跳跃连接中
        transformer_output = self.transformer(enc)

        # 解码器部分
        dec = self.decoder(transformer_output)

        return dec

if __name__ == '__main__':

    # 创建模型实例
    # model = UNet3DWithTransformer(in_channels=64, out_channels=32, num_heads=8, num_layers=4)
    #
    # # 测试模型
    # input_tensor = torch.randn(2, 64, 16, 16, 16)  # 2 samples, 16 channels, 128x128x128
    # output_tensor = model(input_tensor)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet3DWithTransformer(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))
    # print(f"Output shape: {output_tensor.shape}")  # 应该输出 [2, 1, 128, 128, 128]
