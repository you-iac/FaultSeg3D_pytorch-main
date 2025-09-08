import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

class EMA3D(nn.Module):  # 定义一个继承自 nn.Module 的 EMA 类
    def __init__(self, channels, c2=None, factor=16):  # 构造函数，初始化对象
        super(EMA3D, self).__init__()  # 调用父类的构造函数
        self.groups = factor  # 定义组的数量为 factor，默认值为 32
        assert channels // self.groups > 0  # 确保通道数可以被组数整除

        self.softmax = nn.Softmax(-1)  # 定义 softmax 层，用于最后一个维度
        self.agp = nn.AdaptiveAvgPool3d((1, 1, 1))  # 定义自适应平均池化层，输出大小为 1x1
        self.pool_h = nn.AdaptiveAvgPool3d((1,None, None))  # 定义自适应平均池化层，只在宽度上池化
        self.pool_w = nn.AdaptiveAvgPool3d((None, 1, None))  # 定义自适应平均池化层，只在高度上池化
        self.pool_l = nn.AdaptiveAvgPool3d((None,None,1))  # 定义自适应平均池化层，只在高度上池化

        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 定义组归一化层
        self.conv1x1x1 = nn.Conv3d(
            channels // self.groups,
            channels // self.groups,
            kernel_size=1,
            stride=1,
            padding=0
        )  # 定义 1x1 卷积层
        self.conv3x3x3 = nn.Conv3d(
            channels // self.groups,
            channels // self.groups,
            kernel_size=3,
            stride=1,
            padding=1
        )  # 定义 3x3 卷积层

    def forward(self, x):  # 定义前向传播函数
        # 2 x 16 x 128 x 128 x 128
        b, c, h, w, l = x.size()  # 获取输入张量的大小：批次、通道、高度和宽度
                            #32 x 1 x 128 x 128 x128
        group_x = x.reshape(b * self.groups, -1, h, w,l)  # 将输入张量重新形状为 (b * 组数, c // 组数, 高度, 宽度)

        x_h = self.pool_h(group_x)  # 在高度上进行池化        #32 1 1  w l
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2, 4)  #32 1 h  1 l
        x_l = self.pool_l(group_x).permute(0, 1, 4, 2, 3)  #32 1 h  w 1

        hwl = self.conv1x1x1(torch.cat([x_h, x_w, x_l], dim=3))  # 将池化结果拼接并通过 1x1 卷积层
        x_h, x_w, x_l  = torch.split(hwl, [h, w, l], dim=3)  # 将卷积结果按高度和宽度长度

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2, 4).sigmoid() * x_l.permute(0, 1, 4, 2, 3))  # 进行组归一化，并结合高度和宽度的激活结果

        x2 = self.conv3x3x3(group_x)  # 通过 3x3 卷积层

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x1 进行池化、形状变换、并应用 softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将 x2 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x2 进行池化、形状变换、并应用 softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 将 x1 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w,l)  # 计算权重
        return (group_x * weights.sigmoid()).reshape(b, c, h, w, l)  # 应用权重并将形状恢复为原始大小

class DoubleConv(nn.Module):
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


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.EMA3D = EMA3D(in_channels)

    def forward(self, x):
        return self.maxpool_conv(self.EMA3D(x))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
        # 加入CA模块
        self.EMA3D = EMA3D(out_channels)
    def forward(self, x1, x2):
        x1 = self.up( x1 )
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.EMA3D(self.conv(x))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FaultSeg3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FaultSeg3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        self.up2 = Up(192, 64)
        self.up3 = Up(96, 32)
        self.up4 = Up(48, 16)
        self.outc = OutConv(16, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder部分
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # decoder部分
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        outputs = self.softmax(logits)
        return outputs


if __name__ == '__main__':
    # 查看网络参数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FaultSeg3D(1, 2).to(device)
    summary(net, input_size=(1, 128, 128, 128))



# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 6720.63
# Params size (MB): 5.58
# Estimated Total Size (MB): 6734.20
# ----------------------------------------------------------------