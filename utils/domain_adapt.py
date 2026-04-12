import torch
import torch.nn.functional as F
from collections import deque


def global_pool_3d(feat: torch.Tensor) -> torch.Tensor:
    """
    Input:  feat [B, C, D, H, W]
    Output: pooled [B, C]
    """
    if feat.dim() != 5:
        raise ValueError(f"global_pool_3d expects 5D input, got shape={tuple(feat.shape)}")
    return F.adaptive_avg_pool3d(feat, output_size=1).flatten(1)


def coral_loss(source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    """
    Deep CORAL loss between two 2D feature tensors.

    Args:
        source_feat: [N_s, C]
        target_feat: [N_t, C]
    """
    if source_feat.dim() != 2 or target_feat.dim() != 2:
        raise ValueError(
            f"coral_loss expects 2D tensors, got {tuple(source_feat.shape)} and {tuple(target_feat.shape)}"
        )
    if source_feat.size(1) != target_feat.size(1):
        raise ValueError(
            f"feature dim mismatch in coral_loss: {source_feat.size(1)} vs {target_feat.size(1)}"
        )

    source_centered = source_feat - source_feat.mean(dim=0, keepdim=True)
    target_centered = target_feat - target_feat.mean(dim=0, keepdim=True)

    ns = max(source_centered.size(0) - 1, 1)
    nt = max(target_centered.size(0) - 1, 1)

    cov_s = source_centered.t().mm(source_centered) / ns
    cov_t = target_centered.t().mm(target_centered) / nt

    diff = cov_s - cov_t
    d = source_feat.size(1)
    loss = (diff * diff).sum() / (4.0 * d * d)
    return loss


def compute_da_loss(
    feats_s: dict,
    feats_t: dict,
    layers=("x4",),
    layer_weights=None,
) -> torch.Tensor:
    """
    Compute multi-layer CORAL loss from feature dicts.

    Args:
        feats_s: source feature dict, keys include x3/x4 from encoder.
        feats_t: target feature dict, keys include x3/x4 from encoder.
        layers: aligned layers, default only x4.
        layer_weights: optional dict or list/tuple weights.
    """
    if isinstance(layers, str):
        layers = (layers,)

    if len(layers) == 0:
        ref = next(iter(feats_s.values()))
        return ref.new_zeros(())

    total = None

    for idx, layer_name in enumerate(layers):
        if layer_name not in feats_s or layer_name not in feats_t:
            raise KeyError(f"layer '{layer_name}' not found in feature dicts")

        src = feats_s[layer_name]
        tgt = feats_t[layer_name]

        if src.dim() == 5:
            src = global_pool_3d(src)
        if tgt.dim() == 5:
            tgt = global_pool_3d(tgt)

        if src.dim() != 2 or tgt.dim() != 2:
            raise ValueError(
                f"Layer '{layer_name}' must be 2D or 5D features, got {tuple(src.shape)} and {tuple(tgt.shape)}"
            )

        layer_loss = coral_loss(src, tgt)

        weight = 1.0
        if layer_weights is not None:
            if isinstance(layer_weights, dict):
                weight = float(layer_weights.get(layer_name, 1.0))
            elif isinstance(layer_weights, (list, tuple)):
                weight = float(layer_weights[idx]) if idx < len(layer_weights) else 1.0
            else:
                raise TypeError("layer_weights must be dict / list / tuple when provided")

        weighted = layer_loss * weight
        total = weighted if total is None else total + weighted

    return total


class CoralFeatureQueue:
    """
    Cross-batch feature queue for CORAL.

    It stores pooled source/target features from recent steps and computes CORAL
    on [current_batch + history_queue] to stabilize covariance estimation when
    per-step batch size is very small.
    """

    def __init__(
        self,
        layers=("x4",),
        queue_size=32,
        min_samples=2,
        layer_weights=None,
    ):
        if isinstance(layers, str):
            layers = (layers,)
        self.layers = tuple(layers)
        self.queue_size = int(queue_size)
        self.min_samples = int(min_samples)
        self.layer_weights = layer_weights

        self.src_queues = {layer: deque(maxlen=self.queue_size) for layer in self.layers}
        self.tgt_queues = {layer: deque(maxlen=self.queue_size) for layer in self.layers}

    @staticmethod
    def _to_2d(feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 5:
            return global_pool_3d(feat)
        if feat.dim() == 2:
            return feat
        raise ValueError(f"Feature must be 2D or 5D, got shape={tuple(feat.shape)}")

    def _weight_for_layer(self, layer_name, layer_idx):
        if self.layer_weights is None:
            return 1.0
        if isinstance(self.layer_weights, dict):
            return float(self.layer_weights.get(layer_name, 1.0))
        if isinstance(self.layer_weights, (list, tuple)):
            return float(self.layer_weights[layer_idx]) if layer_idx < len(self.layer_weights) else 1.0
        raise TypeError("layer_weights must be dict / list / tuple when provided")

    def _cat_history(self, history_deque, device, dtype):
        if len(history_deque) == 0:
            return None
        hist = torch.cat(list(history_deque), dim=0)
        return hist.to(device=device, dtype=dtype)

    def compute_loss(self, feats_s: dict, feats_t: dict) -> torch.Tensor:
        total = None

        for idx, layer_name in enumerate(self.layers):
            if layer_name not in feats_s or layer_name not in feats_t:
                raise KeyError(f"layer '{layer_name}' not found in feature dicts")

            src_cur = self._to_2d(feats_s[layer_name])
            tgt_cur = self._to_2d(feats_t[layer_name])

            src_hist = self._cat_history(
                self.src_queues[layer_name], device=src_cur.device, dtype=src_cur.dtype
            )
            tgt_hist = self._cat_history(
                self.tgt_queues[layer_name], device=tgt_cur.device, dtype=tgt_cur.dtype
            )

            src_bank = src_cur if src_hist is None else torch.cat([src_cur, src_hist], dim=0)
            tgt_bank = tgt_cur if tgt_hist is None else torch.cat([tgt_cur, tgt_hist], dim=0)

            if src_bank.size(0) < self.min_samples or tgt_bank.size(0) < self.min_samples:
                layer_loss = src_cur.new_zeros(())
            else:
                layer_loss = coral_loss(src_bank, tgt_bank)

            weight = self._weight_for_layer(layer_name, idx)
            weighted = layer_loss * weight
            total = weighted if total is None else total + weighted

        if total is None:
            ref = next(iter(feats_s.values()))
            total = ref.new_zeros(())
        return total

    def update(self, feats_s: dict, feats_t: dict):
        # Keep detached snapshots to avoid backprop through history.
        for layer_name in self.layers:
            src_cur = self._to_2d(feats_s[layer_name]).detach()
            tgt_cur = self._to_2d(feats_t[layer_name]).detach()
            self.src_queues[layer_name].append(src_cur)
            self.tgt_queues[layer_name].append(tgt_cur)

    def step(self, feats_s: dict, feats_t: dict) -> torch.Tensor:
        loss = self.compute_loss(feats_s, feats_t)
        self.update(feats_s, feats_t)
        return loss
