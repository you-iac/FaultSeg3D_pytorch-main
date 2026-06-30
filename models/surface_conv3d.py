import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_plane_points(kernel_size):
    radius = kernel_size // 2
    return [(i, j) for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)]


def _axis_values(kernel_size, device, dtype):
    radius = kernel_size // 2
    values = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    return values / float(max(radius, 1))


def _normalize_grid_coord(coord, size):
    if size <= 1:
        return torch.zeros_like(coord)
    return coord * (2.0 / float(size - 1)) - 1.0


def _as_3tuple(value, name):
    if isinstance(value, int):
        value = (value, value, value)
    if len(value) != 3:
        raise ValueError(f"{name} must be an int or a 3-tuple.")
    if any(v <= 0 for v in value):
        raise ValueError(f"{name} values must be positive.")
    return tuple(value)


def _stride_slice(x, stride):
    if stride == (1, 1, 1):
        return x
    return x[:, :, ::stride[0], ::stride[1], ::stride[2]]


def _base_mesh(batch_size, depth, height, width, device, dtype):
    z = torch.arange(depth, device=device, dtype=dtype)
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x)
    zz = zz.unsqueeze(0).expand(batch_size, -1, -1, -1)
    yy = yy.unsqueeze(0).expand(batch_size, -1, -1, -1)
    xx = xx.unsqueeze(0).expand(batch_size, -1, -1, -1)
    return zz, yy, xx


def _sample_3d(x, dz, dy, dx, padding_mode="zeros", align_corners=True):
    batch_size, _, depth, height, width = x.shape
    zz, yy, xx = _base_mesh(batch_size, depth, height, width, x.device, x.dtype)

    zz = zz + dz
    yy = yy + dy
    xx = xx + dx

    grid = torch.stack(
        (
            _normalize_grid_coord(xx, width),
            _normalize_grid_coord(yy, height),
            _normalize_grid_coord(zz, depth),
        ),
        dim=-1,
    )
    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def _build_accumulated_height(delta):
    # delta: [B, K*K, D, H, W], K must be odd. Heights are built from the
    # center cross first, then each quadrant is propagated from known neighbors.
    batch_size, num_points, depth, height, width = delta.shape
    kernel_size = int(num_points ** 0.5)
    if kernel_size * kernel_size != num_points or kernel_size % 2 == 0:
        raise ValueError("accumulated surface offset expects an odd square kernel.")

    center = kernel_size // 2
    delta_grid = delta.view(batch_size, kernel_size, kernel_size, depth, height, width)
    out = [[None for _ in range(kernel_size)] for _ in range(kernel_size)]
    zero = delta_grid[:, center, center].new_zeros(batch_size, depth, height, width)
    out[center][center] = zero

    # Center row: 11-12-13-14-15 for K=5.
    for col in range(center + 1, kernel_size):
        out[center][col] = out[center][col - 1] + delta_grid[:, center, col]
    for col in range(center - 1, -1, -1):
        out[center][col] = out[center][col + 1] + delta_grid[:, center, col]

    # Center column: 3-8-13-18-23 for K=5.
    for row in range(center - 1, -1, -1):
        out[row][center] = out[row + 1][center] + delta_grid[:, row, center]
    for row in range(center + 1, kernel_size):
        out[row][center] = out[row - 1][center] + delta_grid[:, row, center]

    # Fill the four quadrants. Each point depends on already built vertical and
    # horizontal neighbors, matching examples like 9 <- 8 and 14.
    row_ranges = (range(center - 1, -1, -1), range(center + 1, kernel_size))
    col_ranges = (range(center - 1, -1, -1), range(center + 1, kernel_size))
    for rows in row_ranges:
        for cols in col_ranges:
            for row in rows:
                for col in cols:
                    row_neighbor = row + (1 if row < center else -1)
                    col_neighbor = col + (1 if col < center else -1)
                    out[row][col] = (
                        0.5 * (out[row_neighbor][col] + out[row][col_neighbor])
                        + delta_grid[:, row, col]
                    )

    return torch.stack([out[row][col] for row in range(kernel_size) for col in range(kernel_size)], dim=1)


class SurfaceConv3d(nn.Module):
    """A 3D dynamic surface convolution over one axial plane.

    plane="xz" samples a deformable XZ surface and moves points along Y.
    plane="yz" samples a deformable YZ surface and moves points along X.
    mode="accum" builds offsets from a center-propagated 5x5 surface.
    mode="equation" builds offsets from low-dimensional surface basis weights.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        plane="xz",
        mode="accum",
        offset_scale=1.0,
        stride=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if plane not in ("xz", "yz"):
            raise ValueError("plane must be 'xz' or 'yz'.")
        if mode not in ("accum", "equation"):
            raise ValueError("mode must be 'accum' or 'equation'.")
        if groups <= 0:
            raise ValueError("groups must be positive.")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.plane = plane
        self.mode = mode
        self.offset_scale = offset_scale
        self.stride = _as_3tuple(stride, "stride")
        self.groups = groups
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups
        self.points = _make_plane_points(kernel_size)
        self.num_points = kernel_size * kernel_size

        if mode == "accum":
            offset_channels = self.num_points
        else:
            offset_channels = 5

        self.offset_head = nn.Conv3d(
            in_channels,
            offset_channels,
            kernel_size=3,
            padding=1,
        )
        self.weight = nn.Parameter(torch.empty(out_channels, self.in_channels_per_group, self.num_points))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def _height_from_equation(self, x, control):
        values = _axis_values(self.kernel_size, x.device, x.dtype)
        uu, vv = torch.meshgrid(values, values)
        basis = torch.stack(
            (
                uu,
                vv,
                uu.pow(2),
                vv.pow(2),
                uu * vv,
            ),
            dim=0,
        )
        return torch.einsum("bmdhw,mij->bijdhw", control, basis).flatten(1, 2)

    def _surface_height(self, x):
        raw = torch.tanh(self.offset_head(x)) * float(self.offset_scale)
        if self.mode == "accum":
            return _build_accumulated_height(raw)
        return self._height_from_equation(x, raw)

    def forward(self, x):
        height = self._surface_height(x)
        samples = []
        for point_index, (a, b) in enumerate(self.points):
            normal_offset = height[:, point_index]
            if self.plane == "xz":
                dz = x.new_tensor(float(a))
                dy = normal_offset
                dx = x.new_tensor(float(b))
            else:
                dz = x.new_tensor(float(a))
                dy = x.new_tensor(float(b))
                dx = normal_offset
            samples.append(_sample_3d(x, dz, dy, dx))

        sampled = torch.stack(samples, dim=2)
        if self.groups == 1:
            out = torch.einsum("bcpdhw,ocp->bodhw", sampled, self.weight)
        else:
            batch_size, _, _, depth, height, width = sampled.shape
            sampled = sampled.view(
                batch_size,
                self.groups,
                self.in_channels_per_group,
                self.num_points,
                depth,
                height,
                width,
            )
            weight = self.weight.view(
                self.groups,
                self.out_channels_per_group,
                self.in_channels_per_group,
                self.num_points,
            )
            out = torch.einsum("bgcpdhw,gocp->bgodhw", sampled, weight)
            out = out.reshape(batch_size, self.out_channels, depth, height, width)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)
        return _stride_slice(out, self.stride)


class MultiDirectionSurfaceConv3d(nn.Module):
    """XZ surface branch + YZ surface branch + normal Conv3d with residual fusion."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mode="accum",
        surface_kernel_size=5,
        offset_scale=1.0,
    ):
        super().__init__()
        self.xz_surface = SurfaceConv3d(
            in_channels,
            out_channels,
            kernel_size=surface_kernel_size,
            plane="xz",
            mode=mode,
            offset_scale=offset_scale,
            bias=False,
        )
        self.yz_surface = SurfaceConv3d(
            in_channels,
            out_channels,
            kernel_size=surface_kernel_size,
            plane="yz",
            mode=mode,
            offset_scale=offset_scale,
            bias=False,
        )
        self.normal_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        xz = self.xz_surface(x)
        yz = self.yz_surface(x)
        normal = self.normal_conv(x)
        fused = self.fuse(torch.cat([xz, yz, normal], dim=1))
        return self.act(fused + self.residual(x))


class DoubleSurfaceConv(nn.Module):
    """DoubleConv replacement using multi-direction dynamic surface convolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        mode="accum",
        surface_kernel_size=5,
        offset_scale=1.0,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            MultiDirectionSurfaceConv3d(
                in_channels,
                mid_channels,
                mode=mode,
                surface_kernel_size=surface_kernel_size,
                offset_scale=offset_scale,
            ),
            MultiDirectionSurfaceConv3d(
                mid_channels,
                out_channels,
                mode=mode,
                surface_kernel_size=surface_kernel_size,
                offset_scale=offset_scale,
            ),
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleAccumSurfaceConv(DoubleSurfaceConv):
    def __init__(self, in_channels, out_channels, mid_channels=None, surface_kernel_size=5, offset_scale=1.0):
        super().__init__(
            in_channels,
            out_channels,
            mid_channels=mid_channels,
            mode="accum",
            surface_kernel_size=surface_kernel_size,
            offset_scale=offset_scale,
        )


class DoubleEquationSurfaceConv(DoubleSurfaceConv):
    def __init__(self, in_channels, out_channels, mid_channels=None, surface_kernel_size=5, offset_scale=1.0):
        super().__init__(
            in_channels,
            out_channels,
            mid_channels=mid_channels,
            mode="equation",
            surface_kernel_size=surface_kernel_size,
            offset_scale=offset_scale,
        )
