"""
FaultSeg3D - CEDNet with dynamic surface convolution.

This mirrors the replacement scope in CEDNet_Unet_FullDCN.py:
- Stem feature convolutions
- P2 downsample convolution
- Encoder downsample convolutions
- CEDBlock depthwise spatial convolutions

Pointwise 1x1 projections, decoder, PPM, UPerNet fusion and classifier are kept
as standard Conv3d layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..surface_conv3d import SurfaceConv3d
except ImportError:
    try:
        from models.surface_conv3d import SurfaceConv3d
    except ImportError:
        import os
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from surface_conv3d import SurfaceConv3d

try:
    from .CEDNet import DropPath, LayerNorm3d, Decoder, UPerNet3D
except ImportError:
    from CEDNet import DropPath, LayerNorm3d, Decoder, UPerNet3D


class SurfaceConv3dLayer(nn.Module):
    """XZ/YZ surface branches plus a normal Conv3d branch, fused by 1x1 Conv3d."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mode="accum",
        surface_kernel_size=5,
        offset_scale=1.0,
        stride=1,
        normal_kernel_size=3,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if normal_kernel_size % 2 == 0:
            raise ValueError("normal_kernel_size must be odd.")
        if groups <= 0:
            raise ValueError("groups must be positive.")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups.")

        normal_padding = normal_kernel_size // 2

        self.xz_surface = SurfaceConv3d(
            in_channels,
            out_channels,
            kernel_size=surface_kernel_size,
            plane="xz",
            mode=mode,
            offset_scale=offset_scale,
            stride=stride,
            groups=groups,
            bias=False,
        )
        self.yz_surface = SurfaceConv3d(
            in_channels,
            out_channels,
            kernel_size=surface_kernel_size,
            plane="yz",
            mode=mode,
            offset_scale=offset_scale,
            stride=stride,
            groups=groups,
            bias=False,
        )
        self.normal_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=normal_kernel_size,
            stride=stride,
            padding=normal_padding,
            groups=groups,
            bias=False,
        )
        self.fuse = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1, groups=groups, bias=bias)

    def forward(self, x):
        xz = self.xz_surface(x)
        yz = self.yz_surface(x)
        normal = self.normal_conv(x)
        return self.fuse(torch.cat([xz, yz, normal], dim=1))


class SurfaceConv3dBlock(nn.Module):
    """SurfaceConv3dLayer + LayerNorm3d + activation, matching DeformConv3dBlock."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mode="accum",
        surface_kernel_size=5,
        offset_scale=1.0,
        stride=1,
        normal_kernel_size=3,
        use_gelu=True,
    ):
        super().__init__()
        self.conv = SurfaceConv3dLayer(
            in_channels,
            out_channels,
            mode=mode,
            surface_kernel_size=surface_kernel_size,
            offset_scale=offset_scale,
            stride=stride,
            normal_kernel_size=normal_kernel_size,
            groups=1,
            bias=True,
        )
        self.norm = LayerNorm3d(out_channels)
        self.act = nn.GELU() if use_gelu else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class DepthwiseSurfaceConv3d(nn.Module):
    """Depthwise surface convolution for replacing CEDBlock.dwconv."""

    def __init__(
        self,
        channels,
        mode="accum",
        surface_kernel_size=5,
        offset_scale=1.0,
        normal_kernel_size=3,
    ):
        super().__init__()
        self.conv = SurfaceConv3dLayer(
            channels,
            channels,
            mode=mode,
            surface_kernel_size=surface_kernel_size,
            offset_scale=offset_scale,
            stride=1,
            normal_kernel_size=normal_kernel_size,
            groups=channels,
            bias=True,
        )

    def forward(self, x):
        return self.conv(x)


class CEDBlock(nn.Module):
    """CEDNet block with depthwise dynamic surface convolution."""

    def __init__(
        self,
        dim,
        drop_path=0.,
        layer_scale_init_value=1e-6,
        kernel_size=3,
        surface_mode="accum",
        surface_kernel_size=5,
        offset_scale=1.0,
    ):
        super().__init__()

        self.dwconv = DepthwiseSurfaceConv3d(
            dim,
            mode=surface_mode,
            surface_kernel_size=surface_kernel_size,
            offset_scale=offset_scale,
            normal_kernel_size=kernel_size,
        )
        self.norm = LayerNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma[None, :, None, None, None] * x

        return residual + self.drop_path(x)


class Encoder(nn.Module):
    """CEDNet encoder with surface-conv CEDBlocks and surface-conv downsampling."""

    def __init__(
        self,
        dims=[32, 64, 128],
        blocks=[2, 4, 2],
        dp_rates=None,
        surface_mode="accum",
        surface_kernel_size=5,
        offset_scale=1.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        if dp_rates is None:
            dp_rates = [0.] * sum(blocks)

        self.layer1 = nn.Sequential(
            *[
                CEDBlock(
                    dims[0],
                    drop_path=dp_rates[i],
                    layer_scale_init_value=layer_scale_init_value,
                    surface_mode=surface_mode,
                    surface_kernel_size=surface_kernel_size,
                    offset_scale=offset_scale,
                )
                for i in range(blocks[0])
            ]
        )

        self.down1 = nn.Sequential(
            LayerNorm3d(dims[0]),
            SurfaceConv3dBlock(
                dims[0],
                dims[1],
                mode=surface_mode,
                surface_kernel_size=surface_kernel_size,
                offset_scale=offset_scale,
                stride=2,
                normal_kernel_size=3,
                use_gelu=True,
            ),
        )

        start_idx = blocks[0]
        self.layer2 = nn.Sequential(
            *[
                CEDBlock(
                    dims[1],
                    drop_path=dp_rates[start_idx + i],
                    layer_scale_init_value=layer_scale_init_value,
                    surface_mode=surface_mode,
                    surface_kernel_size=surface_kernel_size,
                    offset_scale=offset_scale,
                )
                for i in range(blocks[1])
            ]
        )

        self.down2 = nn.Sequential(
            LayerNorm3d(dims[1]),
            SurfaceConv3dBlock(
                dims[1],
                dims[2],
                mode=surface_mode,
                surface_kernel_size=surface_kernel_size,
                offset_scale=offset_scale,
                stride=2,
                normal_kernel_size=3,
                use_gelu=True,
            ),
        )

        start_idx = blocks[0] + blocks[1]
        self.layer3 = nn.Sequential(
            *[
                CEDBlock(
                    dims[2],
                    drop_path=dp_rates[start_idx + i],
                    layer_scale_init_value=layer_scale_init_value,
                    kernel_size=3,
                    surface_mode=surface_mode,
                    surface_kernel_size=surface_kernel_size,
                    offset_scale=offset_scale,
                )
                for i in range(blocks[2])
            ]
        )

    def forward(self, x):
        c3 = self.layer1(x)
        x = self.down1(c3)
        c4 = self.layer2(x)
        x = self.down2(c4)
        c5 = self.layer3(x)
        return c3, c4, c5


class FaultSeg3D(nn.Module):
    """CEDNet segmentation model with surface-conv feature extraction."""

    def __init__(
        self,
        n_channels=1,
        n_classes=2,
        dims=[16, 32, 64, 128],
        depths=[2, 2, 4, 2],
        num_stages=3,
        drop_path_rate=0.1,
        upernet_channels=64,
        ppm_scales=(1, 2, 3),
        layer_scale_init_value=1e-6,
        surface_mode="accum",
        surface_kernel_size=5,
        offset_scale=1.0,
    ):
        super().__init__()
        if surface_mode not in ("accum", "equation"):
            raise ValueError("surface_mode must be 'accum' or 'equation'.")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_stages = num_stages
        self.dims = list(dims)
        self.surface_mode = surface_mode
        self.surface_kernel_size = surface_kernel_size
        self.offset_scale = offset_scale

        self.stem = nn.Sequential(
            SurfaceConv3dBlock(
                n_channels,
                dims[0] // 2,
                mode=surface_mode,
                surface_kernel_size=surface_kernel_size,
                offset_scale=offset_scale,
                stride=1,
                normal_kernel_size=3,
                use_gelu=True,
            ),
            SurfaceConv3dBlock(
                dims[0] // 2,
                dims[0],
                mode=surface_mode,
                surface_kernel_size=surface_kernel_size,
                offset_scale=offset_scale,
                stride=2,
                normal_kernel_size=3,
                use_gelu=True,
            ),
        )

        total_blocks = depths[0] + num_stages * sum(depths[1:])
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        p2_blocks = []
        for i in range(depths[0]):
            p2_blocks.append(
                CEDBlock(
                    dims[0],
                    drop_path=dp_rates[i],
                    layer_scale_init_value=layer_scale_init_value,
                    surface_mode=surface_mode,
                    surface_kernel_size=surface_kernel_size,
                    offset_scale=offset_scale,
                )
            )
        self.p2_blocks = nn.Sequential(*p2_blocks)

        self.p2_downsample = SurfaceConv3dBlock(
            dims[0],
            dims[1],
            mode=surface_mode,
            surface_kernel_size=surface_kernel_size,
            offset_scale=offset_scale,
            stride=2,
            normal_kernel_size=3,
            use_gelu=True,
        )

        self.stages = nn.ModuleList()
        cur_dp = depths[0]
        for stage_idx in range(num_stages):
            stage_dp_rates = dp_rates[cur_dp: cur_dp + sum(depths[1:])]
            encoder = Encoder(
                dims=dims[1:],
                blocks=depths[1:],
                dp_rates=stage_dp_rates,
                surface_mode=surface_mode,
                surface_kernel_size=surface_kernel_size,
                offset_scale=offset_scale,
                layer_scale_init_value=layer_scale_init_value,
            )

            if stage_idx < num_stages - 1:
                decoder = Decoder(dims=dims[1:])
                self.stages.append(nn.ModuleList([encoder, decoder]))
            else:
                self.stages.append(nn.ModuleList([encoder]))

            cur_dp += sum(depths[1:])

        self.upernet = UPerNet3D(
            in_channels=dims,
            channels=upernet_channels,
            pool_scales=ppm_scales,
            ppm_channels=32,
        )

        self.seg_head = nn.Sequential(
            nn.Conv3d(upernet_channels, upernet_channels, kernel_size=3, padding=1),
            LayerNorm3d(upernet_channels),
            nn.GELU(),
            nn.Conv3d(upernet_channels, n_classes, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, SurfaceConv3d):
            m.reset_parameters()
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm3d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.p2_blocks(x)
        x = self.p2_downsample(c2)

        for stage in self.stages:
            if len(stage) == 2:
                encoder, decoder = stage
                c3, c4, c5 = encoder(x)
                x, _, _ = decoder(c3, c4, c5)
            else:
                encoder = stage[0]
                c3, c4, c5 = encoder(x)

        fused = self.upernet([c2, c3, c4, c5])
        upsampled = F.interpolate(fused, scale_factor=2, mode="trilinear", align_corners=False)
        logits = self.seg_head(upsampled)
        return self.softmax(logits)

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "FaultSeg3D (CEDNet + SurfaceConv)",
            "total_params": f"{total_params / 1e6:.2f}M",
            "trainable_params": f"{trainable_params / 1e6:.2f}M",
            "dims": self.dims,
            "num_stages": self.num_stages,
            "surface_mode": self.surface_mode,
            "surface_kernel_size": self.surface_kernel_size,
            "offset_scale": self.offset_scale,
            "surface_conv_enabled": "Stem + P2 + All Encoders",
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaultSeg3D(n_channels=1, n_classes=2).to(device)
    x = torch.randn(1, 1, 128, 128, 128, device=device)
    with torch.no_grad():
        y = model(x)
    print(model.get_model_info())
    print(y.shape)
