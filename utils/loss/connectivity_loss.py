from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Offset3D = Tuple[int, int, int]
DEFAULT_OFFSETS_6N: Tuple[Offset3D, ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


class ConnectivityLoss(nn.Module):
    """
    nn.Module wrapper for connectivity_loss_v1.
    """

    def __init__(
        self,
        mask_kernel: int = 3,
        offsets: Sequence[Offset3D] = DEFAULT_OFFSETS_6N,
        loss_type: str = "smooth_l1",
        eps: float = 1e-6,
        foreground_channel: int = 1,
        input_is_logits: Optional[bool] = None,
        track_stats: bool = True,
    ) -> None:
        super().__init__()
        self.mask_kernel = mask_kernel
        self.offsets = tuple(offsets)
        self.loss_type = loss_type
        self.eps = eps
        self.foreground_channel = int(foreground_channel)
        self.input_is_logits = input_is_logits
        self.track_stats = bool(track_stats)
        self.latest_stats = None

    def forward(self, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = _extract_fault_probability(
            pred=pred_prob,
            foreground_channel=self.foreground_channel,
            input_is_logits=self.input_is_logits,
        )
        out = connectivity_loss_v1(
            pred_prob=pred_prob,
            target=target,
            mask_kernel=self.mask_kernel,
            offsets=self.offsets,
            loss_type=self.loss_type,
            eps=self.eps,
            return_stats=self.track_stats,
        )
        if self.track_stats:
            loss, stats = out
            self.latest_stats = stats
            return loss
        self.latest_stats = None
        return out


def _ensure_5d_single_channel(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Ensure x has shape [B, 1, D, H, W].
    Accepts [B, D, H, W] or [B, 1, D, H, W].
    """
    if x.dim() == 4:
        x = x.unsqueeze(1)
    if x.dim() != 5:
        raise ValueError(f"{name} must be 4D or 5D tensor, got shape={tuple(x.shape)}")
    if x.size(1) != 1:
        raise ValueError(f"{name} must have channel size 1, got shape={tuple(x.shape)}")
    return x


def _infer_is_logits(pred: torch.Tensor) -> bool:
    pred_detached = pred.detach()
    return bool(pred_detached.min().item() < 0.0 or pred_detached.max().item() > 1.0)


def _extract_fault_probability(
    pred: torch.Tensor,
    foreground_channel: int = 1,
    input_is_logits: Optional[bool] = None,
) -> torch.Tensor:
    """
    Convert model output to fault probability map [B,1,D,H,W].

    Supported inputs:
    - [B,1,D,H,W] or [B,D,H,W]
    - [B,C,D,H,W] (C>=2), using foreground_channel
    """
    if pred.dim() == 4:
        pred = pred.unsqueeze(1)
    if pred.dim() != 5:
        raise ValueError(f"pred must be 4D or 5D tensor, got shape={tuple(pred.shape)}")

    channels = pred.size(1)
    if channels == 1:
        is_logits = _infer_is_logits(pred) if input_is_logits is None else bool(input_is_logits)
        prob = torch.sigmoid(pred) if is_logits else pred
        return prob.clamp(0.0, 1.0)

    if foreground_channel < 0 or foreground_channel >= channels:
        raise ValueError(
            f"foreground_channel={foreground_channel} out of range for pred shape={tuple(pred.shape)}"
        )

    is_logits = _infer_is_logits(pred) if input_is_logits is None else bool(input_is_logits)
    if is_logits:
        prob = torch.softmax(pred, dim=1)[:, foreground_channel : foreground_channel + 1, ...]
    else:
        prob = pred[:, foreground_channel : foreground_channel + 1, ...]
    return prob.clamp(0.0, 1.0)


def _offset_display_name(offset: Offset3D) -> str:
    if offset == (1, 0, 0):
        return "z"
    if offset == (0, 1, 0):
        return "y"
    if offset == (0, 0, 1):
        return "x"
    dz, dy, dx = offset
    return f"({dz},{dy},{dx})"


def build_fault_neighborhood_mask(target: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """
    Build fault-neighborhood mask M from label Y.

    Args:
        target: [B,1,D,H,W] or [B,D,H,W], values in {0,1} (or probabilistic labels).
        kernel_size: neighborhood size for dilation-like max pooling. Must be positive odd.

    Returns:
        mask: [B,1,D,H,W], float tensor in [0,1]
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be positive odd integer, got {kernel_size}")

    target = _ensure_5d_single_channel(target, "target").float()
    target_bin = (target > 0.5).float()

    if kernel_size == 1:
        return target_bin

    pad = kernel_size // 2
    mask = F.max_pool3d(target_bin, kernel_size=kernel_size, stride=1, padding=pad)
    return torch.clamp(mask, 0.0, 1.0)


def _shift_pair(x: torch.Tensor, offset: Offset3D) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return aligned neighbor pairs x(p) and x(p+offset), both cropped to valid overlap.
    Supports positive and negative offsets.
    """
    if x.dim() != 5:
        raise ValueError(f"x must be 5D [B,1,D,H,W], got shape={tuple(x.shape)}")

    dz, dy, dx = offset
    _, _, d, h, w = x.shape

    if abs(dz) >= d or abs(dy) >= h or abs(dx) >= w:
        raise ValueError(
            f"Offset {offset} is too large for input spatial size {(d, h, w)}"
        )

    z1 = slice(max(0, -dz), d - max(0, dz))
    y1 = slice(max(0, -dy), h - max(0, dy))
    x1 = slice(max(0, -dx), w - max(0, dx))

    z2 = slice(max(0, dz), d - max(0, -dz))
    y2 = slice(max(0, dy), h - max(0, -dy))
    x2 = slice(max(0, dx), w - max(0, -dx))

    return x[:, :, z1, y1, x1], x[:, :, z2, y2, x2]


def _pointwise_connectivity_loss(
    pred_edge: torch.Tensor,
    gt_edge: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    loss_type = loss_type.lower()
    if loss_type == "l1":
        return F.l1_loss(pred_edge, gt_edge, reduction="none")
    if loss_type == "mse":
        return F.mse_loss(pred_edge, gt_edge, reduction="none")
    if loss_type == "bce":
        pred_edge = pred_edge.clamp(1e-6, 1.0 - 1e-6)
        return F.binary_cross_entropy(pred_edge, gt_edge, reduction="none")
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred_edge, gt_edge, reduction="none")
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def connectivity_loss_v1(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    mask_kernel: int = 7,
    offsets: Sequence[Offset3D] = DEFAULT_OFFSETS_6N,
    loss_type: str = "smooth_l1",
    eps: float = 1e-6,
    return_stats: bool = False,
) -> torch.Tensor:
    """
    Connectivity loss for 3D segmentation.

    Core definition:
        A_pred_delta(x) = P(x) * P(x + delta)
        A_gt_delta(x)   = Y(x) * Y(x + delta)
        L_conn = sum_delta  sum_x M_delta(x) * l(A_pred_delta, A_gt_delta)
                           / (sum_x M_delta(x) + eps)

    Args:
        pred_prob: [B,1,D,H,W] probability map in [0,1].
        target: [B,1,D,H,W] binary labels in {0,1}.
        mask_kernel: size of local fault-neighborhood mask.
        offsets: list/tuple of 3D offsets, e.g. 6-neighborhood positive half:
                 ((1,0,0),(0,1,0),(0,0,1)).
        loss_type: one of {"smooth_l1","bce","l1","mse"}.
        eps: numerical stability term for denominator.
    """
    pred_prob = _ensure_5d_single_channel(pred_prob, "pred_prob").float()
    target = _ensure_5d_single_channel(target, "target").float()

    if pred_prob.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred_prob={tuple(pred_prob.shape)}, target={tuple(target.shape)}"
        )
    if len(offsets) == 0:
        raise ValueError("offsets must contain at least one direction")

    pred_prob = pred_prob.clamp(0.0, 1.0)
    target_bin = (target > 0.5).float()
    mask = build_fault_neighborhood_mask(target_bin, kernel_size=mask_kernel)

    total_loss = pred_prob.new_tensor(0.0)
    valid_dirs = 0
    direction_losses = {}
    direction_edge_counts = {}

    for idx, offset in enumerate(offsets):
        p1, p2 = _shift_pair(pred_prob, offset)
        y1, y2 = _shift_pair(target_bin, offset)
        m1, m2 = _shift_pair(mask, offset)

        conn_pred = p1 * p2
        conn_gt = y1 * y2
        edge_mask = (m1 * m2).detach()

        diff = _pointwise_connectivity_loss(conn_pred, conn_gt, loss_type=loss_type)
        masked_sum = (diff * edge_mask).sum()
        normalizer = edge_mask.sum().clamp_min(eps)
        loss_dir = masked_sum / normalizer
        total_loss = total_loss + loss_dir
        valid_dirs += 1
        if return_stats:
            name = _offset_display_name(offset)
            if name in direction_losses:
                name = f"{name}_{idx}"
            direction_losses[name] = float(loss_dir.detach().item())
            direction_edge_counts[name] = float(edge_mask.sum().detach().item())

    loss = total_loss / float(valid_dirs)
    if not return_stats:
        return loss

    stats = {
        "loss": float(loss.detach().item()),
        "direction_losses": direction_losses,
        "direction_edge_counts": direction_edge_counts,
        "total_edge_count": float(sum(direction_edge_counts.values())),
    }
    return loss, stats




def connectivity_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    foreground_channel: int = 1,
    **kwargs,
) -> torch.Tensor:
    """
    Convenience function when model output is logits.

    Args:
        logits:
          - [B,2,D,H,W] (foreground channel selected by foreground_channel), or
          - [B,1,D,H,W].
        target: [B,1,D,H,W] or [B,D,H,W]
    """
    pred_prob = _extract_fault_probability(
        pred=logits,
        foreground_channel=foreground_channel,
        input_is_logits=True,
    )

    return connectivity_loss_v1(pred_prob=pred_prob, target=target, **kwargs)


__all__ = [
    "DEFAULT_OFFSETS_6N",
    "ConnectivityLoss",
    "build_fault_neighborhood_mask",
    "connectivity_loss_from_logits",
    "connectivity_loss_v1",
]
