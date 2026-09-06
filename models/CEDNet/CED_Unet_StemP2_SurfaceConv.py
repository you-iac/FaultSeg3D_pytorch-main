"""CED_Unet ablation: dynamic surface convolution only in Stem and P2.

The reference architecture is ``CED_Unet.py``, not ``CEDNet.py``.  The model
therefore preserves all characteristic CED_Unet paths:

* a full-resolution c1 feature from a UNet-style double convolution;
* P2 at half resolution and two P2 downsampling operations;
* three standard cascaded CEDNet encoder/decoder stages;
* five-level UPerNet fusion over [c1, c2, c3, c4, c5];
* a direct 1x1 classifier at the original input resolution.

Only the Stem and P2 convolutions are changed to SurfaceConv3d.  Surface points
are evaluated in vectorized chunks (five points by default), downsampling grids
are built at their output resolution, and activation checkpointing is enabled
during training by default.  The public ``FaultSeg3D`` API is compatible with
the repository's train/test utilities.
This file works in its normal location and when copied verbatim to
``models/faultseg3d.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# Copying this file to models/faultseg3d.py changes its package depth by one.
if __package__ and __package__.endswith(".CEDNet"):
    from ..surface_conv3d import SurfaceConv3d
    from .CED_Unet import CEDBlock as StandardCEDBlock
    from .CED_Unet import Decoder, DropPath, Encoder, LayerNorm3d, UPerNet3D
elif __package__:
    from .surface_conv3d import SurfaceConv3d
    from .CEDNet.CED_Unet import CEDBlock as StandardCEDBlock
    from .CEDNet.CED_Unet import Decoder, DropPath, Encoder, LayerNorm3d, UPerNet3D
else:
    # Support: python models/CEDNet/CED_Unet_StemP2_SurfaceConv.py
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from models.surface_conv3d import SurfaceConv3d
    from models.CEDNet.CED_Unet import CEDBlock as StandardCEDBlock
    from models.CEDNet.CED_Unet import Decoder, DropPath, Encoder, LayerNorm3d, UPerNet3D


__all__ = ["FaultSeg3D"]


class ChunkedSurfaceConv3d(SurfaceConv3d):
    """Memory-aware SurfaceConv3d with vectorized point chunks.

    All surface points participate in every forward pass.  ``point_chunk_size``
    only controls how many points are packed into one ``grid_sample`` call; it
    does not randomly drop points or freeze any weights.
    """

    def __init__(self, *args, point_chunk_size: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if point_chunk_size <= 0:
            raise ValueError("point_chunk_size must be positive.")
        self.point_chunk_size = min(point_chunk_size, self.num_points)
        self.register_buffer(
            "_point_offsets",
            torch.tensor(self.points, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _normalize_coordinate(coordinate: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 1:
            return torch.zeros_like(coordinate)
        return coordinate * (2.0 / float(size - 1)) - 1.0

    def _sample_chunk(
        self,
        x: torch.Tensor,
        surface_height: torch.Tensor,
        start: int,
        end: int,
        base_z: torch.Tensor,
        base_y: torch.Tensor,
        base_x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, in_depth, in_height, in_width = x.shape
        stride_d, stride_h, stride_w = self.stride
        normal_offset = surface_height[
            :, start:end, ::stride_d, ::stride_h, ::stride_w
        ]

        offsets = self._point_offsets[start:end].to(dtype=x.dtype)
        point_a = offsets[:, 0].view(1, -1, 1, 1, 1)
        point_b = offsets[:, 1].view(1, -1, 1, 1, 1)

        if self.plane == "xz":
            grid_z = base_z + point_a
            grid_y = base_y + normal_offset
            grid_x = base_x + point_b
        else:
            grid_z = base_z + point_a
            grid_y = base_y + point_b
            grid_x = base_x + normal_offset

        grid_z, grid_y, grid_x = torch.broadcast_tensors(grid_z, grid_y, grid_x)
        grid = torch.stack(
            (
                self._normalize_coordinate(grid_x, in_width),
                self._normalize_coordinate(grid_y, in_height),
                self._normalize_coordinate(grid_z, in_depth),
            ),
            dim=-1,
        )

        chunk_size = end - start
        out_depth, out_height, out_width = normal_offset.shape[-3:]
        # grid_sample has no point dimension.  Pack the point dimension into
        # output depth, perform one call for the whole chunk, then unpack it.
        packed_grid = grid.reshape(
            batch_size,
            chunk_size * out_depth,
            out_height,
            out_width,
            3,
        )
        sampled = F.grid_sample(
            x,
            packed_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.reshape(
            batch_size,
            channels,
            chunk_size,
            out_depth,
            out_height,
            out_width,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        surface_height = self._surface_height(x)
        batch_size, _, in_depth, in_height, in_width = x.shape
        stride_d, stride_h, stride_w = self.stride

        # Construct the strided output grid once.  This is numerically
        # equivalent to computing the full output and slicing it afterwards,
        # but downsampling layers avoid roughly 7/8 of the sampled voxels.
        base_z = torch.arange(0, in_depth, stride_d, device=x.device, dtype=x.dtype)
        base_y = torch.arange(0, in_height, stride_h, device=x.device, dtype=x.dtype)
        base_x = torch.arange(0, in_width, stride_w, device=x.device, dtype=x.dtype)
        base_z = base_z.view(1, 1, -1, 1, 1)
        base_y = base_y.view(1, 1, 1, -1, 1)
        base_x = base_x.view(1, 1, 1, 1, -1)

        output = None
        for start in range(0, self.num_points, self.point_chunk_size):
            end = min(start + self.point_chunk_size, self.num_points)
            sampled = self._sample_chunk(
                x,
                surface_height,
                start,
                end,
                base_z,
                base_y,
                base_x,
            )

            if self.groups == 1:
                partial = torch.einsum(
                    "bcqdhw,ocq->bodhw",
                    sampled,
                    self.weight[:, :, start:end],
                )
            else:
                chunk_size = end - start
                out_depth, out_height, out_width = sampled.shape[-3:]
                sampled = sampled.reshape(
                    batch_size,
                    self.groups,
                    self.in_channels_per_group,
                    chunk_size,
                    out_depth,
                    out_height,
                    out_width,
                )
                weight = self.weight.reshape(
                    self.groups,
                    self.out_channels_per_group,
                    self.in_channels_per_group,
                    self.num_points,
                )
                partial = torch.einsum(
                    "bgcqdhw,gocq->bgodhw",
                    sampled,
                    weight[:, :, :, start:end],
                ).reshape(
                    batch_size,
                    self.out_channels,
                    out_depth,
                    out_height,
                    out_width,
                )

            output = partial if output is None else output + partial

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output


class SurfaceConv3dLayer(nn.Module):
    """Fuse dynamic XZ/YZ surfaces with a regular 3D convolution branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "accum",
        surface_kernel_size: int = 5,
        offset_scale: float = 1.0,
        stride: int = 1,
        normal_kernel_size: int = 3,
        groups: int = 1,
        bias: bool = True,
        point_chunk_size: int = 5,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if normal_kernel_size % 2 == 0:
            raise ValueError("normal_kernel_size must be odd.")
        if groups <= 0:
            raise ValueError("groups must be positive.")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups.")

        surface_kwargs = dict(
            kernel_size=surface_kernel_size,
            mode=mode,
            offset_scale=offset_scale,
            stride=stride,
            groups=groups,
            bias=False,
            point_chunk_size=point_chunk_size,
        )
        self.xz_surface = ChunkedSurfaceConv3d(
            in_channels,
            out_channels,
            plane="xz",
            **surface_kwargs,
        )
        self.yz_surface = ChunkedSurfaceConv3d(
            in_channels,
            out_channels,
            plane="yz",
            **surface_kwargs,
        )
        self.normal_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=normal_kernel_size,
            stride=stride,
            padding=normal_kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.fuse = nn.Conv3d(
            out_channels * 3,
            out_channels,
            kernel_size=1,
            groups=groups,
            bias=bias,
        )
        self.use_checkpointing = use_checkpointing

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        xz = self.xz_surface(x)
        yz = self.yz_surface(x)
        normal = self.normal_conv(x)
        return self.fuse(torch.cat((xz, yz, normal), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class SurfaceDoubleConv(nn.Module):
    """CED_Unet DoubleConv with only its Conv3d operations replaced."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        surface_mode: str = "accum",
        surface_kernel_size: int = 5,
        offset_scale: float = 1.0,
        point_chunk_size: int = 5,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        surface_kwargs = dict(
            mode=surface_mode,
            surface_kernel_size=surface_kernel_size,
            offset_scale=offset_scale,
            point_chunk_size=point_chunk_size,
            use_checkpointing=use_checkpointing,
        )
        # Keep CED_Unet's BatchNorm3d + ReLU sequence exactly.
        self.double_conv = nn.Sequential(
            SurfaceConv3dLayer(in_channels, mid_channels, **surface_kwargs),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            SurfaceConv3dLayer(mid_channels, out_channels, **surface_kwargs),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class P2SurfaceCEDBlock(nn.Module):
    """CED_Unet CEDBlock with its depthwise Conv3d replaced by SurfaceConv."""

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        kernel_size: int = 3,
        surface_mode: str = "accum",
        surface_kernel_size: int = 5,
        offset_scale: float = 1.0,
        point_chunk_size: int = 5,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.dwconv = SurfaceConv3dLayer(
            dim,
            dim,
            mode=surface_mode,
            surface_kernel_size=surface_kernel_size,
            offset_scale=offset_scale,
            normal_kernel_size=kernel_size,
            groups=dim,
            point_chunk_size=point_chunk_size,
            use_checkpointing=use_checkpointing,
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
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma[None, :, None, None, None] * x
        return residual + self.drop_path(x)


class FaultSeg3D(nn.Module):
    """CED_Unet with SurfaceConv3d restricted to full-resolution Stem and P2."""

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        dims=(16, 32, 64, 128),
        depths=(2, 2, 4, 2),
        num_stages: int = 3,
        drop_path_rate: float = 0.1,
        upernet_channels: int = 64,
        ppm_scales=(1, 2, 3),
        layer_scale_init_value: float = 1e-6,
        surface_mode: str = "accum",
        surface_kernel_size: int = 5,
        offset_scale: float = 1.0,
        point_chunk_size: int = 5,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if surface_mode not in ("accum", "equation"):
            raise ValueError("surface_mode must be 'accum' or 'equation'.")
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("dims and depths must each contain four values.")
        if num_stages < 1:
            raise ValueError("num_stages must be at least 1.")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_stages = num_stages
        self.dims = list(dims)
        self.depths = list(depths)
        self.upernet_channels = upernet_channels  # Kept for CED_Unet API compatibility.
        self.surface_mode = surface_mode
        self.surface_kernel_size = surface_kernel_size
        self.offset_scale = offset_scale
        self.point_chunk_size = min(point_chunk_size, surface_kernel_size ** 2)
        self.use_checkpointing = use_checkpointing

        surface_kwargs = dict(
            surface_mode=surface_mode,
            surface_kernel_size=surface_kernel_size,
            offset_scale=offset_scale,
            point_chunk_size=point_chunk_size,
            use_checkpointing=use_checkpointing,
        )

        # CED_Unet Stem: full-resolution c1, with its two convolutions replaced.
        self.first_conv = SurfaceDoubleConv(
            n_channels,
            dims[0],
            **surface_kwargs,
        )

        total_blocks = depths[0] + num_stages * sum(depths[1:])
        dp_rates = [value.item() for value in torch.linspace(0, drop_path_rate, total_blocks)]

        # CED_Unet P2 step 1: 128^3 -> 64^3, preserving Norm + GELU order.
        self.p2_downsample1 = nn.Sequential(
            SurfaceConv3dLayer(
                dims[0],
                dims[0],
                mode=surface_mode,
                surface_kernel_size=surface_kernel_size,
                offset_scale=offset_scale,
                stride=2,
                normal_kernel_size=3,
                point_chunk_size=point_chunk_size,
                use_checkpointing=use_checkpointing,
            ),
            LayerNorm3d(dims[0]),
            nn.GELU(),
        )

        # CED_Unet P2 step 2: two feature extraction blocks at 64^3.
        self.p2_blocks = nn.Sequential(
            *[
                P2SurfaceCEDBlock(
                    dims[0],
                    drop_path=dp_rates[index],
                    layer_scale_init_value=layer_scale_init_value,
                    surface_mode=surface_mode,
                    surface_kernel_size=surface_kernel_size,
                    offset_scale=offset_scale,
                    point_chunk_size=point_chunk_size,
                    use_checkpointing=use_checkpointing,
                )
                for index in range(depths[0])
            ]
        )

        # CED_Unet P2 step 3: 64^3 -> 32^3, preserving pre-normalization.
        # SurfaceConv uses an odd 3x3 normal branch instead of CED_Unet's 2x2
        # Conv3d; for even feature sizes both produce the same half resolution.
        self.p2_downsample2 = nn.Sequential(
            LayerNorm3d(dims[0]),
            SurfaceConv3dLayer(
                dims[0],
                dims[1],
                mode=surface_mode,
                surface_kernel_size=surface_kernel_size,
                offset_scale=offset_scale,
                stride=2,
                normal_kernel_size=3,
                point_chunk_size=point_chunk_size,
                use_checkpointing=use_checkpointing,
            ),
        )

        # Exact CED_Unet backbone: no surface convolution in the cascaded stages.
        self.stages = nn.ModuleList()
        current_dp = depths[0]
        blocks_per_encoder = sum(depths[1:])
        for stage_index in range(num_stages):
            stage_dp_rates = dp_rates[current_dp : current_dp + blocks_per_encoder]
            encoder = Encoder(
                dims=dims[1:],
                blocks=depths[1:],
                dp_rates=stage_dp_rates,
            )
            if stage_index < num_stages - 1:
                self.stages.append(nn.ModuleList((encoder, Decoder(dims=dims[1:]))))
            else:
                self.stages.append(nn.ModuleList((encoder,)))
            current_dp += blocks_per_encoder

        # Exact CED_Unet five-level progressive fusion and direct classifier.
        self.upernet = UPerNet3D(
            in_channels=[dims[0], *dims],
            pool_scales=ppm_scales,
            ppm_channels=32,
        )
        self.classifier = nn.Conv3d(dims[0], n_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, SurfaceConv3d):
            module.reset_parameters()
        elif isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNorm3d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Enable or disable checkpointing for every surface-convolution layer."""
        self.use_checkpointing = bool(enabled)
        for module in self.modules():
            if isinstance(module, SurfaceConv3dLayer):
                module.use_checkpointing = self.use_checkpointing

    # Alias for training entry points that use the more common method name.
    set_gradient_checkpointing = set_activation_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.first_conv(x)
        x = self.p2_downsample1(c1)
        c2 = self.p2_blocks(x)
        x = self.p2_downsample2(c2)

        for stage in self.stages:
            if len(stage) == 2:
                encoder, decoder = stage
                c3, c4, c5 = encoder(x)
                x, _, _ = decoder(c3, c4, c5)
            else:
                c3, c4, c5 = stage[0](x)

        fused = self.upernet((c1, c2, c3, c4, c5))
        return self.softmax(self.classifier(fused))

    def get_model_info(self) -> dict[str, object]:
        total_params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        surface_layers = sum(
            isinstance(module, SurfaceConv3dLayer) for module in self.modules()
        )
        stage_surface_layers = sum(
            isinstance(module, SurfaceConv3dLayer) for module in self.stages.modules()
        )
        standard_stage_blocks = sum(
            isinstance(module, StandardCEDBlock) for module in self.stages.modules()
        )
        return {
            "model_name": "FaultSeg3D (CED_Unet Stem+P2 SurfaceConv ablation)",
            "reference_architecture": "CED_Unet.py",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "dims": self.dims,
            "depths": self.depths,
            "num_stages": self.num_stages,
            "surface_mode": self.surface_mode,
            "surface_kernel_size": self.surface_kernel_size,
            "offset_scale": self.offset_scale,
            "point_chunk_size": self.point_chunk_size,
            "activation_checkpointing": self.use_checkpointing,
            "grid_sample_calls_per_surface": (
                self.surface_kernel_size ** 2 + self.point_chunk_size - 1
            )
            // self.point_chunk_size,
            "surface_conv_placement": "first_conv + P2 only",
            "surface_conv_layers": surface_layers,
            "stage_surface_conv_layers": stage_surface_layers,
            "standard_stage_ced_blocks": standard_stage_blocks,
            "fusion_features": "[c1, c2, c3, c4, c5]",
        }


def _smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaultSeg3D().to(device).eval()
    sample = torch.randn(1, 1, 32, 32, 32, device=device)
    with torch.inference_mode():
        prediction = model(sample)

    expected_shape = (1, 2, 32, 32, 32)
    if tuple(prediction.shape) != expected_shape:
        raise RuntimeError(f"Expected output shape {expected_shape}, got {tuple(prediction.shape)}")

    print(model.get_model_info())
    print("input:", tuple(sample.shape))
    print("output:", tuple(prediction.shape))
    print("probability sum:", prediction.sum(dim=1).mean().item())


if __name__ == "__main__":
    _smoke_test()
