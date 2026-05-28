"""Reusable 3D FFC/DFF frequency-domain blocks.

The core block is :class:`FFC3D_DFF`, a 3D Fast Fourier Convolution style
module with a DFF-style spatial attention gate in the frequency branch.

Expected input shape: ``(B, C, D, H, W)``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "LayerNorm3D",
    "make_norm3d",
    "ConvNormAct3D",
    "DFFSpatialAttention3D",
    "FourierUnit3D_DFF",
    "SpectralTransform3D_DFF",
    "FFC3D_DFF",
    "FFCResidualBlock3D",
    "FFC3DBlockDFF",
]


class LayerNorm3D(nn.Module):
    """Channels-first LayerNorm for tensors shaped ``(B, C, D, H, W)``."""

    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            weight = self.weight.view(1, -1, 1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1, 1)
            x = x * weight + bias
        return x


def make_norm3d(norm_type: str, num_channels: int) -> nn.Module:
    """Create a 3D normalization layer.

    Args:
        norm_type: ``"bn"``, ``"in"``, ``"ln"``, or ``"none"``.
        num_channels: Number of channels in the normalized tensor.
    """

    norm_type = norm_type.lower()
    if norm_type == "bn":
        return nn.BatchNorm3d(num_channels)
    if norm_type == "in":
        return nn.InstanceNorm3d(num_channels, affine=True)
    if norm_type == "ln":
        return LayerNorm3D(num_channels, affine=True)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvNormAct3D(nn.Module):
    """3D convolution followed by normalization and optional ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
        norm_type: str = "bn",
        act: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = make_norm3d(norm_type, out_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DFFSpatialAttention3D(nn.Module):
    """DFF-style instance-adaptive spatial attention for 3D frequency features.

    The input is normally the concatenated real/imaginary FFT representation.
    The module builds a single attention map from channel average and maximum
    projections, then gates the input feature map.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class FourierUnit3D_DFF(nn.Module):
    """3D Fourier unit with optional DFF attention.

    Flow:
        ``rFFTN -> concat(real, imag) -> 1x1x1 conv -> norm -> ReLU
        -> DFF attention -> split(real, imag) -> iRFFTN``.
    """

    def __init__(
        self,
        channels: int,
        norm_type: str = "bn",
        use_dff: bool = True,
        dff_kernel_size: int = 3,
        fft_norm: str = "ortho",
    ):
        super().__init__()
        self.channels = channels
        self.fft_norm = fft_norm
        self.freq_conv = nn.Conv3d(channels * 2, channels * 2, kernel_size=1, bias=False)
        self.freq_norm = make_norm3d(norm_type, channels * 2)
        self.freq_act = nn.ReLU(inplace=True)
        self.dff = DFFSpatialAttention3D(kernel_size=dff_kernel_size) if use_dff else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, depth, height, width = x.shape
        x_fft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm=self.fft_norm)

        freq = torch.cat([x_fft.real, x_fft.imag], dim=1)
        freq = self.freq_act(self.freq_norm(self.freq_conv(freq)))
        freq = self.dff(freq)

        real, imag = torch.chunk(freq, 2, dim=1)
        x_fft = torch.complex(real, imag)
        return torch.fft.irfftn(
            x_fft,
            s=(depth, height, width),
            dim=(-3, -2, -1),
            norm=self.fft_norm,
        )


class SpectralTransform3D_DFF(nn.Module):
    """Reduced-channel spectral transform used by the global FFC branch."""

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        stride: int = 1,
        norm_type: str = "bn",
        use_dff: bool = True,
        dff_kernel_size: int = 3,
        fft_norm: str = "ortho",
    ):
        super().__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        out_channels = channels if out_channels is None else out_channels
        mid_channels = max(channels // 2, 1)

        self.stride = stride
        self.reduce = ConvNormAct3D(
            channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type,
            act=True,
        )
        self.fourier = FourierUnit3D_DFF(
            mid_channels,
            norm_type=norm_type,
            use_dff=use_dff,
            dff_kernel_size=dff_kernel_size,
            fft_norm=fft_norm,
        )
        self.project = ConvNormAct3D(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type,
            act=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = F.avg_pool3d(x, kernel_size=self.stride, stride=self.stride, ceil_mode=True)
        x = self.reduce(x)
        x = x + self.fourier(x)
        return self.project(x)


class FFC3D_DFF(nn.Module):
    """3D Fast Fourier Convolution block with DFF attention.

    The input channels are split into local and global groups. Local paths use
    spatial ``Conv3d``; the global-to-global path uses ``SpectralTransform3D_DFF``.

    Args:
        ratio_gin: Fraction of input channels assigned to the global branch.
        ratio_gout: Fraction of output channels assigned to the global branch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        ratio_gin: float = 0.5,
        ratio_gout: float = 0.5,
        norm_type: str = "bn",
        use_dff: bool = True,
        dff_kernel_size: int = 3,
        fft_norm: str = "ortho",
    ):
        super().__init__()
        if not 0.0 <= ratio_gin <= 1.0:
            raise ValueError(f"ratio_gin must be in [0, 1], got {ratio_gin}")
        if not 0.0 <= ratio_gout <= 1.0:
            raise ValueError(f"ratio_gout must be in [0, 1], got {ratio_gout}")

        in_global = int(round(in_channels * ratio_gin))
        in_local = in_channels - in_global
        out_global = int(round(out_channels * ratio_gout))
        out_local = out_channels - out_global

        self.in_local = in_local
        self.in_global = in_global
        self.out_local = out_local
        self.out_global = out_global

        padding = kernel_size // 2
        self.local_to_local = (
            ConvNormAct3D(in_local, out_local, kernel_size, stride, padding, norm_type=norm_type, act=False)
            if in_local > 0 and out_local > 0
            else None
        )
        self.local_to_global = (
            ConvNormAct3D(in_local, out_global, kernel_size, stride, padding, norm_type=norm_type, act=False)
            if in_local > 0 and out_global > 0
            else None
        )
        self.global_to_local = (
            ConvNormAct3D(in_global, out_local, kernel_size, stride, padding, norm_type=norm_type, act=False)
            if in_global > 0 and out_local > 0
            else None
        )
        self.global_to_global = (
            SpectralTransform3D_DFF(
                in_global,
                out_channels=out_global,
                stride=stride,
                norm_type=norm_type,
                use_dff=use_dff,
                dff_kernel_size=dff_kernel_size,
                fft_norm=fft_norm,
            )
            if in_global > 0 and out_global > 0
            else None
        )

        self.out_norm_local = make_norm3d(norm_type, out_local) if out_local > 0 else None
        self.out_norm_global = make_norm3d(norm_type, out_global) if out_global > 0 else None
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _add_term(acc: Optional[torch.Tensor], term: torch.Tensor) -> torch.Tensor:
        if acc is None:
            return term
        if term.shape[-3:] != acc.shape[-3:]:
            term = F.interpolate(term, size=acc.shape[-3:], mode="trilinear", align_corners=False)
        return acc + term

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_global == 0:
            x_local, x_global = x, None
        elif self.in_local == 0:
            x_local, x_global = None, x
        else:
            x_local, x_global = torch.split(x, [self.in_local, self.in_global], dim=1)

        y_local = None
        y_global = None

        if self.out_local > 0:
            if self.local_to_local is not None and x_local is not None:
                y_local = self._add_term(y_local, self.local_to_local(x_local))
            if self.global_to_local is not None and x_global is not None:
                y_local = self._add_term(y_local, self.global_to_local(x_global))
            if y_local is None:
                raise RuntimeError("Local output branch has no valid input path.")
            y_local = self.act(self.out_norm_local(y_local))

        if self.out_global > 0:
            if self.local_to_global is not None and x_local is not None:
                y_global = self._add_term(y_global, self.local_to_global(x_local))
            if self.global_to_global is not None and x_global is not None:
                y_global = self._add_term(y_global, self.global_to_global(x_global))
            if y_global is None:
                raise RuntimeError("Global output branch has no valid input path.")
            y_global = self.act(self.out_norm_global(y_global))

        if y_local is not None and y_global is not None:
            if y_global.shape[-3:] != y_local.shape[-3:]:
                y_global = F.interpolate(y_global, size=y_local.shape[-3:], mode="trilinear", align_corners=False)
            return torch.cat([y_local, y_global], dim=1)
        if y_local is not None:
            return y_local
        return y_global


class FFCResidualBlock3D(nn.Module):
    """Residual block using two ``FFC3D_DFF`` layers."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_type: str = "bn",
        ratio_g: float = 0.5,
        use_dff: bool = True,
        dff_kernel_size: int = 3,
        fft_norm: str = "ortho",
    ):
        super().__init__()
        self.conv1 = FFC3D_DFF(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            ratio_gin=ratio_g,
            ratio_gout=ratio_g,
            norm_type=norm_type,
            use_dff=use_dff,
            dff_kernel_size=dff_kernel_size,
            fft_norm=fft_norm,
        )
        self.conv2 = FFC3D_DFF(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            ratio_gin=ratio_g,
            ratio_gout=ratio_g,
            norm_type=norm_type,
            use_dff=use_dff,
            dff_kernel_size=dff_kernel_size,
            fft_norm=fft_norm,
        )
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_norm3d(norm_type, out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv2(self.conv1(x))
        if identity.shape[-3:] != out.shape[-3:]:
            identity = F.interpolate(identity, size=out.shape[-3:], mode="trilinear", align_corners=False)
        return self.relu(out + identity)


FFC3DBlockDFF = FFC3D_DFF


if __name__ == "__main__":
    block = FFC3D_DFF(16, 32, ratio_gin=0.5, ratio_gout=0.5)
    x = torch.randn(1, 16, 32, 32, 32)
    y = block(x)
    print(tuple(y.shape))
