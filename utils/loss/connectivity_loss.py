import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_5d_target(target):
    if target.dim() == 4:
        target = target.unsqueeze(1)
    if target.dim() != 5:
        raise ValueError("target must have shape [B, D, H, W] or [B, C, D, H, W].")
    if target.size(1) > 1:
        target = target[:, 1:2]
    return (target.float() > 0.5).float()


def _foreground_probability(pred, input_is_probability=None):
    if pred.dim() != 5:
        raise ValueError("pred must have shape [B, C, D, H, W].")

    if pred.size(1) == 2:
        return torch.softmax(pred, dim=1)[:, 1:2]

    if pred.size(1) != 1:
        raise ValueError("connectivity loss supports one-channel or two-channel predictions.")

    if input_is_probability is None:
        pred_detached = pred.detach()
        input_is_probability = pred_detached.min().item() >= 0.0 and pred_detached.max().item() <= 1.0

    return pred if input_is_probability else torch.sigmoid(pred)


def _shift_with_valid(x, dz, dy, dx):
    out = torch.zeros_like(x)
    valid = torch.zeros_like(x[:, :1])

    _, _, depth, height, width = x.shape

    src_z = slice(max(dz, 0), depth + min(dz, 0))
    src_y = slice(max(dy, 0), height + min(dy, 0))
    src_x = slice(max(dx, 0), width + min(dx, 0))

    dst_z = slice(max(-dz, 0), depth - max(dz, 0))
    dst_y = slice(max(-dy, 0), height - max(dy, 0))
    dst_x = slice(max(-dx, 0), width - max(dx, 0))

    out[:, :, dst_z, dst_y, dst_x] = x[:, :, src_z, src_y, src_x]
    valid[:, :, dst_z, dst_y, dst_x] = 1.0
    return out, valid


def _dilate(mask, kernel_size):
    if kernel_size <= 1:
        return mask.float()
    return F.max_pool3d(mask.float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def _pool_prob(pred, scale):
    if scale == 1:
        return pred
    return F.avg_pool3d(pred, kernel_size=scale, stride=scale)


def _pool_target(target, scale):
    if scale == 1:
        return target
    return F.max_pool3d(target.float(), kernel_size=scale, stride=scale)


def _make_offsets(neighborhood):
    if isinstance(neighborhood, str):
        neighborhood = neighborhood.lower()
    if neighborhood in (6, "6", "axis"):
        return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if neighborhood in (18, "18"):
        max_l1 = 2
    elif neighborhood in (26, "26"):
        max_l1 = 3
    else:
        raise ValueError("neighborhood must be one of 6, 18, or 26.")

    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                if abs(dz) + abs(dy) + abs(dx) > max_l1:
                    continue
                if dz > 0 or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0):
                    offsets.append((dz, dy, dx))
    return offsets


def _direction_weight(offset, axis_weights):
    dz, dy, dx = offset
    weights = torch.as_tensor(axis_weights, dtype=torch.float32)
    active = torch.as_tensor([abs(dz), abs(dy), abs(dx)], dtype=torch.float32)
    return float((active * weights).sum() / active.sum().clamp_min(1.0))


class ConnectivityLoss(nn.Module):
    """Break/merge separated multi-scale connectivity loss for 3D fault masks.

    The loss accepts the current training output directly:
    - two-channel logits/probabilities: [B, 2, D, H, W]
    - one-channel logits/probabilities: [B, 1, D, H, W]
    - labels: [B, D, H, W] or [B, 1, D, H, W]
    """

    def __init__(
        self,
        break_weight=1.0,
        merge_weight=0.2,
        loss_weight=1.0,
        scales=(1, 2, 4),
        scale_weights=(0.5, 1.0, 0.7),
        pred_threshold=0.3,
        mask_kernel_size=3,
        neighborhood=6,
        axis_weights=(0.7, 1.0, 1.0),
        break_axis_weights=None,
        merge_axis_weights=None,
        input_is_probability=None,
        detach_pred_mask=True,
        return_components=False,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__()
        loss_weight = kwargs.pop("weight", loss_weight)
        loss_weight = kwargs.pop("conn_weight", loss_weight)
        loss_weight = kwargs.pop("connectivity_weight", loss_weight)
        loss_weight = kwargs.pop("lambda_conn", loss_weight)
        pred_threshold = kwargs.pop("threshold", pred_threshold)
        pred_threshold = kwargs.pop("prob_threshold", pred_threshold)
        mask_kernel_size = kwargs.pop("kernel_size", mask_kernel_size)
        mask_kernel_size = kwargs.pop("mask_size", mask_kernel_size)
        mask_radius = kwargs.pop("radius", None)
        mask_radius = kwargs.pop("mask_radius", mask_radius)
        if mask_radius is not None:
            mask_kernel_size = int(mask_radius) * 2 + 1

        if isinstance(scales, int):
            scales = (scales,)
        if isinstance(scale_weights, (int, float)):
            scale_weights = (float(scale_weights),) * len(scales)
        if len(scales) != len(scale_weights):
            raise ValueError("scales and scale_weights must have the same length.")
        mask_kernel_size = int(mask_kernel_size)
        if mask_kernel_size % 2 == 0:
            mask_kernel_size += 1

        self.break_weight = break_weight
        self.merge_weight = merge_weight
        self.loss_weight = loss_weight
        self.scales = tuple(scales)
        self.scale_weights = tuple(scale_weights)
        self.pred_threshold = pred_threshold
        self.mask_kernel_size = mask_kernel_size
        self.offsets = tuple(_make_offsets(neighborhood))
        self.axis_weights = tuple(axis_weights)
        self.break_axis_weights = tuple(break_axis_weights) if break_axis_weights is not None else tuple(axis_weights)
        self.merge_axis_weights = tuple(merge_axis_weights) if merge_axis_weights is not None else tuple(axis_weights)
        self.input_is_probability = input_is_probability
        self.detach_pred_mask = detach_pred_mask
        self.return_components = return_components
        self.eps = eps

        self.last_break_loss = None
        self.last_merge_loss = None

    def forward(self, pred=None, target=None, *args, **kwargs):
        if pred is None:
            pred = kwargs.pop("pred_prob", None)
        if pred is None:
            pred = kwargs.pop("prob", None)
        if pred is None:
            pred = kwargs.pop("logits", None)
        if pred is None:
            pred = kwargs.pop("outputs", None)
        if target is None:
            target = kwargs.pop("label", None)
        if target is None:
            target = kwargs.pop("labels", None)
        if target is None:
            target = kwargs.pop("gt", None)
        if pred is None or target is None:
            raise TypeError("ConnectivityLoss.forward requires pred/pred_prob and target/labels.")

        pred = _foreground_probability(pred, self.input_is_probability)
        target = _as_5d_target(target).to(device=pred.device, dtype=pred.dtype)

        total = pred.new_tensor(0.0)
        break_total = pred.new_tensor(0.0)
        merge_total = pred.new_tensor(0.0)
        weight_total = pred.new_tensor(0.0)

        for scale, scale_weight in zip(self.scales, self.scale_weights):
            if min(pred.shape[-3:]) < scale:
                continue
            pred_s = _pool_prob(pred, scale)
            target_s = _pool_target(target, scale)

            break_s, merge_s = self._single_scale_loss(pred_s, target_s)
            scale_weight_t = pred.new_tensor(float(scale_weight))

            break_total = break_total + scale_weight_t * break_s
            merge_total = merge_total + scale_weight_t * merge_s
            total = total + scale_weight_t * (self.break_weight * break_s + self.merge_weight * merge_s)
            weight_total = weight_total + scale_weight_t

        if weight_total <= 0:
            return pred.sum() * 0.0

        total = self.loss_weight * total / weight_total.clamp_min(self.eps)
        break_total = break_total / weight_total.clamp_min(self.eps)
        merge_total = merge_total / weight_total.clamp_min(self.eps)

        self.last_break_loss = break_total.detach()
        self.last_merge_loss = merge_total.detach()

        if self.return_components:
            return total, {"break": break_total, "merge": merge_total}
        return total

    def _single_scale_loss(self, pred, target):
        pred = pred.clamp(self.eps, 1.0 - self.eps)

        pred_for_mask = pred.detach() if self.detach_pred_mask else pred
        target_region = _dilate(target, self.mask_kernel_size)
        pred_region = _dilate((pred_for_mask > self.pred_threshold).float(), self.mask_kernel_size)

        # Break loss must see GT-near regions even when prediction is currently low.
        break_mask = target_region

        # Merge loss focuses on GT-near and predicted-positive regions to avoid full-background domination.
        merge_mask = torch.clamp(target_region + pred_region, 0.0, 1.0)

        break_terms = []
        merge_terms = []

        for offset in self.offsets:
            dz, dy, dx = offset
            pred_n, valid = _shift_with_valid(pred, dz, dy, dx)
            target_n, _ = _shift_with_valid(target, dz, dy, dx)
            break_mask_n, _ = _shift_with_valid(break_mask, dz, dy, dx)
            merge_mask_n, _ = _shift_with_valid(merge_mask, dz, dy, dx)

            pred_aff = pred * pred_n
            target_aff = target * target_n

            break_pair_mask = valid * break_mask * break_mask_n
            merge_pair_mask = valid * merge_mask * merge_mask_n

            break_eligible = target_aff * break_pair_mask
            merge_eligible = (1.0 - target_aff) * merge_pair_mask

            break_loss = (break_eligible * (1.0 - pred_aff)).sum() / break_eligible.sum().clamp_min(self.eps)
            merge_loss = (merge_eligible * pred_aff).sum() / merge_eligible.sum().clamp_min(self.eps)

            break_dir_weight = pred.new_tensor(_direction_weight(offset, self.break_axis_weights))
            merge_dir_weight = pred.new_tensor(_direction_weight(offset, self.merge_axis_weights))

            break_terms.append(break_dir_weight * break_loss)
            merge_terms.append(merge_dir_weight * merge_loss)

        break_out = torch.stack(break_terms).mean() if break_terms else pred.new_tensor(0.0)
        merge_out = torch.stack(merge_terms).mean() if merge_terms else pred.new_tensor(0.0)
        return break_out, merge_out


class MultiScaleConnectivityLoss(ConnectivityLoss):
    pass


class DirectionalConnectivityLoss(ConnectivityLoss):
    pass


class BreakMergeConnectivityLoss(ConnectivityLoss):
    pass


class SoftConnectivityLoss(ConnectivityLoss):
    pass


class ConnectivityLoss3D(ConnectivityLoss):
    pass


def connectivity_loss(pred, target, **kwargs):
    return ConnectivityLoss(**kwargs)(pred, target)


def multiscale_connectivity_loss(pred, target, **kwargs):
    return ConnectivityLoss(**kwargs)(pred, target)


def directional_connectivity_loss(pred, target, **kwargs):
    return ConnectivityLoss(**kwargs)(pred, target)


def soft_connectivity_loss(pred, target, **kwargs):
    return ConnectivityLoss(**kwargs)(pred, target)


def __getattr__(name):
    if "connect" in name.lower() and "loss" in name.lower():
        return ConnectivityLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
