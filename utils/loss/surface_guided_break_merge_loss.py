"""GT surface geometry as fixed weights for native-resolution 3D affinities.

GT supplies geometry only at foreground voxels; prediction selects the region.
Normals are estimated from local foreground coordinate PCA, not mask gradients.
The scalar p_i*p_j has no direction: the spatial edge (j-i) is what is compared
with the normal. This is local geometry supervision, not a global topology test.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .connectivity_loss import _as_5d_target, _dilate, _make_offsets


def _edge_slices(shape, offset):
    """Views at i and i+offset, with no padding or wraparound."""
    if any(abs(step) >= size for size, step in zip(shape, offset)):
        return None
    first = (slice(None), slice(None)) + tuple(
        slice(max(-step, 0), size - max(step, 0))
        for size, step in zip(shape, offset)
    )
    second = (slice(None), slice(None)) + tuple(
        slice(max(step, 0), size + min(step, 0))
        for size, step in zip(shape, offset)
    )
    return first, second


class SurfaceGuidedBreakMergeLoss(nn.Module):
    """Surface-weighted break/merge loss for binary 3D fault segmentation.

    Inputs: prediction [B,1/2,D,H,W], GT [B,D,H,W] or [B,1/2,D,H,W].
    Positional ``pred`` defaults to logits. Set input_is_probability=True for
    probabilities (including this project's CEDNet output). The pred_prob=
    keyword explicitly selects probabilities; logits= explicitly selects logits.

    Q_i = confidence_i * n_i n_i^T on GT foreground, zero on all background.
    For unit edge direction u, q=(u^T Q_i u + u^T Q_j u)/2 and
    c=(trace(Q_i)+trace(Q_j))/2. Geometry weights are
        break: 1 + geometry_strength * (c-q)
        merge: 1 + geometry_strength * q.
    Positive/negative edges are averaged separately over UNWEIGHTED counts.
    Both endpoints must lie in the dilated prediction-positive region. GT
    classifies selected edges and supplies normals, but does not select regions.
    Geometry is GT-only, detached, and computed in float32 outside autocast.
    Combine this auxiliary term with a voxel segmentation loss.

    Voxel spacing order is (D,H,W), used for PCA coordinates and edge direction.
    Only the original resolution is used. Entirely missed GT outside the
    prediction region receives no auxiliary supervision; Dice/CE still supervises it.
    """

    def __init__(
        self,
        break_weight=1.0,
        merge_weight=0.2,
        loss_weight=1.0,
        geometry_strength=1.0,
        neighborhood=26,
        normal_kernel_size=5,
        min_points=6,
        voxel_spacing=(1.0, 1.0, 1.0),
        pred_threshold=0.3,
        mask_kernel_size=3,
        input_is_probability=False,
        return_components=False,
        eigh_chunk_size=8192,
        eps=1e-6,
    ):
        super().__init__()
        for name, value in (("break_weight", break_weight), ("merge_weight", merge_weight),
                            ("loss_weight", loss_weight), ("geometry_strength", geometry_strength)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative.")
        if normal_kernel_size < 3 or normal_kernel_size % 2 != 1:
            raise ValueError("normal_kernel_size must be an odd integer >= 3.")
        if mask_kernel_size < 1 or mask_kernel_size % 2 != 1:
            raise ValueError("mask_kernel_size must be a positive odd integer.")
        if min_points < 3 or int(min_points) != min_points:
            raise ValueError("min_points must be an integer >= 3.")
        if len(voxel_spacing) != 3 or any(s <= 0 or not math.isfinite(s) for s in voxel_spacing):
            raise ValueError("voxel_spacing must contain three finite positive values.")
        if not 0 <= pred_threshold <= 1 or not 0 < eps < 0.5:
            raise ValueError("Require pred_threshold in [0,1] and eps in (0,0.5).")
        if int(eigh_chunk_size) != eigh_chunk_size or eigh_chunk_size < 1:
            raise ValueError("eigh_chunk_size must be a positive integer.")
        if not isinstance(input_is_probability, bool):
            raise ValueError("Set input_is_probability explicitly to True or False.")

        self.break_weight = float(break_weight)
        self.merge_weight = float(merge_weight)
        self.loss_weight = float(loss_weight)
        self.geometry_strength = float(geometry_strength)
        self.offsets = tuple(_make_offsets(neighborhood))
        self.normal_kernel_size = int(normal_kernel_size)
        self.min_points = int(min_points)
        self.voxel_spacing = tuple(float(s) for s in voxel_spacing)
        self.pred_threshold = float(pred_threshold)
        self.mask_kernel_size = int(mask_kernel_size)
        self.input_is_probability = input_is_probability
        self.return_components = return_components
        self.eigh_chunk_size = int(eigh_chunk_size)
        self.eps = float(eps)
        self.last_break_loss = None
        self.last_merge_loss = None

        # Relative-coordinate moments avoid subtracting large global coordinates.
        radius = self.normal_kernel_size // 2
        coords = [torch.arange(-radius, radius + 1, dtype=torch.float32) * s
                  for s in self.voxel_spacing]
        z, y, x = torch.meshgrid(*coords, indexing="ij")
        kernels = torch.stack((torch.ones_like(z), z, y, x,
                               z*z, y*y, x*x, z*y, z*x, y*x)).unsqueeze(1)
        self.register_buffer("moment_kernels", kernels, persistent=False)

    @torch.no_grad()
    def estimate_geometry(self, target):
        """Return normals, symmetric normal tensors and confidence on GT only.

        normal: [B,3,D,H,W], GT only, arbitrary sign; zero if unsupported.
        normal_tensor: [B,6,D,H,W], packed (zz,yy,xx,zy,zx,yx), includes
        confidence. Using nn^T prevents sign cancellation.
        confidence: [B,1,D,H,W], trace(normal_tensor).
        Every returned field is zero at all GT-background voxels.
        """
        with torch.autocast(device_type=target.device.type, enabled=False):
            gt = _as_5d_target(target).detach().float()
            batch, _, depth, height, width = gt.shape
            normal = gt.new_zeros((batch, 3, depth, height, width))
            confidence = torch.zeros_like(gt)
            indices = (gt[:, 0] > 0.5).nonzero(as_tuple=True)
            if indices[0].numel():
                moments = F.conv3d(gt, self.moment_kernels.to(device=gt.device, dtype=torch.float32),
                                   padding=self.normal_kernel_size // 2)
                samples = moments.permute(0, 2, 3, 4, 1)[indices]
                del moments
                normals_flat = normal.permute(0, 2, 3, 4, 1)
                confidence_flat = confidence[:, 0]
                for start in range(0, samples.shape[0], self.eigh_chunk_size):
                    stop = start + self.eigh_chunk_size
                    local = samples[start:stop]
                    count = local[:, :1].clamp_min(1.0)
                    mean = local[:, 1:4] / count
                    second = local[:, 4:10] / count
                    covariance = local.new_zeros((local.shape[0], 3, 3))
                    covariance[:, 0, 0], covariance[:, 1, 1], covariance[:, 2, 2] = second[:, :3].unbind(1)
                    covariance[:, 0, 1] = covariance[:, 1, 0] = second[:, 3]
                    covariance[:, 0, 2] = covariance[:, 2, 0] = second[:, 4]
                    covariance[:, 1, 2] = covariance[:, 2, 1] = second[:, 5]
                    covariance -= mean.unsqueeze(2) * mean.unsqueeze(1)
                    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                    eigenvalues = eigenvalues.clamp_min(0.0)
                    reliability = ((eigenvalues[:, 1] - eigenvalues[:, 0]) /
                                   eigenvalues[:, 2].clamp_min(self.eps)).clamp(0.0, 1.0)
                    reliability *= (local[:, 0] >= self.min_points)
                    n = eigenvectors[:, :, 0] * (reliability > self.eps).unsqueeze(1)
                    index_chunk = tuple(index[start:stop] for index in indices)
                    normals_flat[index_chunk] = n
                    confidence_flat[index_chunk] = reliability

            nz, ny, nx = normal.split(1, dim=1)
            field = torch.cat((nz*nz, ny*ny, nx*nx, nz*ny, nz*nx, ny*nx), dim=1) * confidence
            return {"normal": normal, "normal_tensor": field,
                    "confidence": field[:, :3].sum(dim=1, keepdim=True)}

    def _project_direction(self, field, offset):
        direction = [step * s for step, s in zip(offset, self.voxel_spacing)]
        length = math.sqrt(sum(d*d for d in direction))
        z, y, x = (d / length for d in direction)
        return (z*z*field[:, 0:1] + y*y*field[:, 1:2] + x*x*field[:, 2:3]
                + 2*z*y*field[:, 3:4] + 2*z*x*field[:, 4:5] + 2*y*x*field[:, 5:6])

    def _geometry_pair_weights(self, geometry, offset):
        """Full-size weight maps at i for edges i -> i+offset (debug/test helper)."""
        field, confidence = geometry["normal_tensor"], geometry["confidence"]
        break_map, merge_map = torch.ones_like(confidence), torch.ones_like(confidence)
        slices = _edge_slices(field.shape[-3:], offset)
        if slices is not None:
            first, second = slices
            q = self._project_direction(field, offset)
            c_pair = 0.5 * (confidence[first] + confidence[second])
            q_pair = torch.minimum((0.5 * (q[first] + q[second])).clamp_min(0.0), c_pair)
            break_map[first] += self.geometry_strength * (c_pair - q_pair)
            merge_map[first] += self.geometry_strength * q_pair
        return break_map, merge_map

    def forward(self, pred=None, target=None, *, pred_prob=None, logits=None):
        if sum(value is not None for value in (pred, pred_prob, logits)) != 1 or target is None:
            raise TypeError("Provide exactly one of pred, pred_prob, logits, and provide target.")
        probability_input = self.input_is_probability
        if pred_prob is not None:
            pred, probability_input = pred_prob, True
        elif logits is not None:
            pred, probability_input = logits, False
        if pred.dim() != 5 or pred.size(1) not in (1, 2) or not pred.is_floating_point():
            raise ValueError("pred must be floating point [B,1/2,D,H,W].")
        if any(size == 0 for size in pred.shape):
            raise ValueError("pred dimensions must be nonempty.")

        with torch.autocast(device_type=pred.device.type, enabled=False):
            raw = pred if pred.dtype == torch.float64 else pred.float()
            if probability_input:
                probability = raw[:, -1:]
            else:
                probability = torch.softmax(raw, dim=1)[:, 1:2] if raw.size(1) == 2 else torch.sigmoid(raw)
            probability = probability.clamp(0.0, 1.0)
            gt = _as_5d_target(target).to(device=pred.device, dtype=probability.dtype).detach()
            if gt.shape != probability.shape:
                raise ValueError("pred and target batch/spatial shapes must match.")
            geometry = self.estimate_geometry(gt) if self.geometry_strength > 0 else None
            with torch.no_grad():
                # Region selection alone is hard-thresholded; affinities stay soft.
                # Keep eligibility identical when geometry_strength=0 so that
                # disabling geometry is a controlled weighting-only ablation.
                pred_region = _dilate((probability.detach() > self.pred_threshold).float(), self.mask_kernel_size)

            zero = probability.sum() * 0.0
            break_terms, merge_terms = [], []
            for offset in self.offsets:
                slices = _edge_slices(probability.shape[-3:], offset)
                if slices is None:
                    continue
                first, second = slices
                affinity = probability[first] * probability[second]
                pair_region = pred_region[first] * pred_region[second]
                target_affinity = gt[first] * gt[second]
                positive = target_affinity * pair_region
                negative = (1.0 - target_affinity) * pair_region
                if geometry is None:
                    weighted_positive, weighted_negative = positive, negative
                else:
                    break_map, merge_map = self._geometry_pair_weights(geometry, offset)
                    weighted_positive = positive * break_map[first]
                    weighted_negative = negative * merge_map[first]
                # Never divide by the geometry-weight sum: that cancels constant
                # tangential/normal weights on planar regions.
                break_terms.append((weighted_positive * (1.0 - affinity)).sum() /
                                   positive.sum().clamp_min(self.eps))
                merge_terms.append((weighted_negative * affinity).sum() /
                                   negative.sum().clamp_min(self.eps))
            break_loss = torch.stack(break_terms).mean() if break_terms else zero
            merge_loss = torch.stack(merge_terms).mean() if merge_terms else zero
            total = self.loss_weight * (self.break_weight * break_loss + self.merge_weight * merge_loss)
            self.last_break_loss = break_loss.detach()
            self.last_merge_loss = merge_loss.detach()
            if self.return_components:
                return total, {"break": break_loss, "merge": merge_loss}
            return total
