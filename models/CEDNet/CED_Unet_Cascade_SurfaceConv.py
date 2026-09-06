"""CED_Unet ablation with 3x3 SurfaceConv in the cascaded encoders.

The reference network is CED_Unet.py.  Stem, P2, decoder and UPerNet paths
remain unchanged.  Only the depthwise Conv3d in every CEDBlock of the three
cascaded encoders is replaced by a two-plane SurfaceConv3d layer.

All 3x3 surface points (nine points) participate in every forward pass.  They
are packed into one grid_sample call per plane by default.  Activation
checkpointing is enabled for these layers during training to limit memory.
This module works in its current package and when copied to models/faultseg3d.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# Copying this file to models/faultseg3d.py changes the package depth by one.
if __package__ and __package__.endswith(".CEDNet"):
    from ..surface_conv3d import SurfaceConv3d
    from .CED_Unet import Decoder, DropPath, FaultSeg3D as BaseFaultSeg3D, LayerNorm3d
elif __package__:
    from .surface_conv3d import SurfaceConv3d
    from .CEDNet.CED_Unet import (
        Decoder,
        DropPath,
        FaultSeg3D as BaseFaultSeg3D,
        LayerNorm3d,
    )
else:
    # Support: python models/CEDNet/CED_Unet_Cascade_SurfaceConv.py
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from models.surface_conv3d import SurfaceConv3d
    from models.CEDNet.CED_Unet import (
        Decoder,
        DropPath,
        FaultSeg3D as BaseFaultSeg3D,
        LayerNorm3d,
    )


__all__ = ["FaultSeg3D"]


class ChunkedSurfaceConv3d(SurfaceConv3d):
    """SurfaceConv3d that evaluates all points in vectorized chunks."""

    def __init__(self, *args, point_chunk_size: int = 9, **kwargs) -> None:
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
        batch_size, channels, depth, height, width = x.shape
        normal_offset = surface_height[:, start:end]

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
                self._normalize_coordinate(grid_x, width),
                self._normalize_coordinate(grid_y, height),
                self._normalize_coordinate(grid_z, depth),
            ),
            dim=-1,
        )

        chunk_size = end - start
        packed_grid = grid.reshape(
            batch_size,
            chunk_size * depth,
            height,
            width,
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
            depth,
            height,
            width,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride != (1, 1, 1):
            raise RuntimeError("Cascade SurfaceConv only supports stride=1.")

        surface_height = self._surface_height(x)
        batch_size, _, depth, height, width = x.shape
        base_z = torch.arange(depth, device=x.device, dtype=x.dtype).view(
            1, 1, depth, 1, 1
        )
        base_y = torch.arange(height, device=x.device, dtype=x.dtype).view(
            1, 1, 1, height, 1
        )
        base_x = torch.arange(width, device=x.device, dtype=x.dtype).view(
            1, 1, 1, 1, width
        )

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
                sampled = sampled.reshape(
                    batch_size,
                    self.groups,
                    self.in_channels_per_group,
                    chunk_size,
                    depth,
                    height,
                    width,
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
                    depth,
                    height,
                    width,
                )

            output = partial if output is None else output + partial

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output


class SurfaceDepthwiseConv3d(nn.Module):
    """XZ/YZ surface branches plus a regular depthwise 3D branch."""

    def __init__(
        self,
        channels: int,
        surface_mode: str = "accum",
        surface_kernel_size: int = 3,
        offset_scale: float = 1.0,
        point_chunk_size: int = 9,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if surface_kernel_size != 3:
            raise ValueError(
                "This ablation is fixed to surface_kernel_size=3 (nine points)."
            )

        kwargs = dict(
            kernel_size=surface_kernel_size,
            mode=surface_mode,
            offset_scale=offset_scale,
            groups=channels,
            bias=False,
            point_chunk_size=point_chunk_size,
        )
        self.xz_surface = ChunkedSurfaceConv3d(
            channels,
            channels,
            plane="xz",
            **kwargs,
        )
        self.yz_surface = ChunkedSurfaceConv3d(
            channels,
            channels,
            plane="yz",
            **kwargs,
        )
        self.normal_conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.fuse = nn.Conv3d(
            channels * 3,
            channels,
            kernel_size=1,
            groups=channels,
            bias=True,
        )
        self.use_checkpointing = bool(use_checkpointing)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        xz = self.xz_surface(x)
        yz = self.yz_surface(x)
        normal = self.normal_conv(x)
        return self.fuse(torch.cat((xz, yz, normal), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class CascadeSurfaceCEDBlock(nn.Module):
    """Original CEDBlock with only its depthwise Conv3d replaced."""

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        surface_mode: str = "accum",
        surface_kernel_size: int = 3,
        offset_scale: float = 1.0,
        point_chunk_size: int = 9,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.dwconv = SurfaceDepthwiseConv3d(
            dim,
            surface_mode=surface_mode,
            surface_kernel_size=surface_kernel_size,
            offset_scale=offset_scale,
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


class CascadeSurfaceEncoder(nn.Module):
    """CED_Unet Encoder with SurfaceConv in all of its CEDBlocks."""

    def __init__(
        self,
        dims: tuple[int, int, int] | list[int],
        blocks: tuple[int, int, int] | list[int],
        dp_rates: list[float],
        surface_mode: str,
        surface_kernel_size: int,
        offset_scale: float,
        point_chunk_size: int,
        use_checkpointing: bool,
        layer_scale_init_value: float,
    ) -> None:
        super().__init__()
        common = dict(
            surface_mode=surface_mode,
            surface_kernel_size=surface_kernel_size,
            offset_scale=offset_scale,
            point_chunk_size=point_chunk_size,
            use_checkpointing=use_checkpointing,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.layer1 = nn.Sequential(
            *[
                CascadeSurfaceCEDBlock(dims[0], drop_path=dp_rates[i], **common)
                for i in range(blocks[0])
            ]
        )
        self.down1 = nn.Sequential(
            LayerNorm3d(dims[0]),
            nn.Conv3d(dims[0], dims[1], kernel_size=2, stride=2),
        )

        start = blocks[0]
        self.layer2 = nn.Sequential(
            *[
                CascadeSurfaceCEDBlock(
                    dims[1],
                    drop_path=dp_rates[start + i],
                    **common,
                )
                for i in range(blocks[1])
            ]
        )
        self.down2 = nn.Sequential(
            LayerNorm3d(dims[1]),
            nn.Conv3d(dims[1], dims[2], kernel_size=2, stride=2),
        )

        start += blocks[1]
        self.layer3 = nn.Sequential(
            *[
                CascadeSurfaceCEDBlock(
                    dims[2],
                    drop_path=dp_rates[start + i],
                    **common,
                )
                for i in range(blocks[2])
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3 = self.layer1(x)
        c4 = self.layer2(self.down1(c3))
        c5 = self.layer3(self.down2(c4))
        return c3, c4, c5


class FaultSeg3D(BaseFaultSeg3D):
    """CED_Unet with nine-point SurfaceConv in all cascaded CEDBlocks."""

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        dims: tuple[int, int, int, int] | list[int] = (16, 32, 64, 128),
        depths: tuple[int, int, int, int] | list[int] = (2, 2, 4, 2),
        num_stages: int = 3,
        drop_path_rate: float = 0.1,
        upernet_channels: int = 64,
        ppm_scales: tuple[int, ...] = (1, 2, 3),
        layer_scale_init_value: float = 1e-6,
        surface_mode: str = "accum",
        surface_kernel_size: int = 3,
        offset_scale: float = 1.0,
        point_chunk_size: int = 9,
        use_checkpointing: bool = True,
    ) -> None:
        if surface_kernel_size != 3:
            raise ValueError(
                "CED_Unet_Cascade_SurfaceConv uses a fixed 3x3, nine-point surface."
            )
        if point_chunk_size <= 0:
            raise ValueError("point_chunk_size must be positive.")

        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            dims=list(dims),
            depths=list(depths),
            num_stages=num_stages,
            drop_path_rate=drop_path_rate,
            upernet_channels=upernet_channels,
            ppm_scales=ppm_scales,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.dims = tuple(dims)
        self.depths = tuple(depths)
        self.surface_mode = surface_mode
        self.surface_kernel_size = surface_kernel_size
        self.offset_scale = offset_scale
        self.point_chunk_size = min(point_chunk_size, surface_kernel_size**2)
        self.use_checkpointing = bool(use_checkpointing)

        total_blocks = depths[0] + num_stages * sum(depths[1:])
        dp_rates = [
            value.item()
            for value in torch.linspace(0, drop_path_rate, total_blocks)
        ]

        self.stages = nn.ModuleList()
        current_dp = depths[0]
        blocks_per_encoder = sum(depths[1:])
        for stage_index in range(num_stages):
            stage_dp_rates = dp_rates[current_dp : current_dp + blocks_per_encoder]
            encoder = CascadeSurfaceEncoder(
                dims=dims[1:],
                blocks=depths[1:],
                dp_rates=stage_dp_rates,
                surface_mode=surface_mode,
                surface_kernel_size=surface_kernel_size,
                offset_scale=offset_scale,
                point_chunk_size=self.point_chunk_size,
                use_checkpointing=use_checkpointing,
                layer_scale_init_value=layer_scale_init_value,
            )
            if stage_index < num_stages - 1:
                self.stages.append(nn.ModuleList((encoder, Decoder(dims=dims[1:]))))
            else:
                self.stages.append(nn.ModuleList((encoder,)))
            current_dp += blocks_per_encoder

        # Initialize only the replacement stages.  Surface offset heads remain
        # zero-initialized, so training begins from an undeformed regular grid.
        self.stages.apply(self._init_cascade_weights)

    @staticmethod
    def _init_cascade_weights(module: nn.Module) -> None:
        if isinstance(module, SurfaceConv3d):
            module.reset_parameters()
        elif isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNorm3d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def set_activation_checkpointing(self, enabled: bool) -> None:
        self.use_checkpointing = bool(enabled)
        for module in self.modules():
            if isinstance(module, SurfaceDepthwiseConv3d):
                module.use_checkpointing = self.use_checkpointing

    set_gradient_checkpointing = set_activation_checkpointing

    def get_model_info(self) -> dict[str, object]:
        total_params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        surface_layers = sum(
            isinstance(module, SurfaceDepthwiseConv3d)
            for module in self.modules()
        )
        surface_blocks_by_level = [
            sum(
                isinstance(module, CascadeSurfaceCEDBlock)
                for stage in self.stages
                for module in stage[0].__getattr__(level).modules()
            )
            for level in ("layer1", "layer2", "layer3")
        ]
        calls_per_plane = (
            self.surface_kernel_size**2 + self.point_chunk_size - 1
        ) // self.point_chunk_size
        return {
            "model_name": "FaultSeg3D (CED_Unet Cascade 3x3 SurfaceConv)",
            "reference_architecture": "CED_Unet.py",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "dims": self.dims,
            "depths": self.depths,
            "num_stages": self.num_stages,
            "surface_mode": self.surface_mode,
            "surface_kernel_size": self.surface_kernel_size,
            "surface_points": self.surface_kernel_size**2,
            "point_chunk_size": self.point_chunk_size,
            "activation_checkpointing": self.use_checkpointing,
            "surface_conv_placement": "all CEDBlocks in cascaded encoders",
            "surface_layers": surface_layers,
            "surface_blocks_c3_c4_c5": surface_blocks_by_level,
            "grid_sample_calls_per_plane_per_layer": calls_per_plane,
            "grid_sample_calls_per_forward": surface_layers * 2 * calls_per_plane,
        }


def _smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaultSeg3D().to(device).eval()
    sample = torch.randn(1, 1, 32, 32, 32, device=device)
    with torch.inference_mode():
        prediction = model(sample)

    expected_shape = (1, 2, 32, 32, 32)
    if tuple(prediction.shape) != expected_shape:
        raise RuntimeError(
            f"Expected output shape {expected_shape}, got {tuple(prediction.shape)}"
        )
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Model output contains NaN or Inf.")

    print(model.get_model_info())
    print("input:", tuple(sample.shape))
    print("output:", tuple(prediction.shape))
    print("probability sum:", prediction.sum(dim=1).mean().item())


if __name__ == "__main__":
    _smoke_test()
