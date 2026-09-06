"""Visualize the exact GT geometry used by SurfaceGuidedBreakMergeLoss.

Example (from the repository root):
    python 可视化工具/visualize_surface_geometry.py data/data_3D_400/valid/y/0.npy
Add --show for local Matplotlib windows; the 3D view can be rotated with a mouse.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.loss.surface_guided_break_merge_loss import SurfaceGuidedBreakMergeLoss


def load_label(path):
    """Keep the training axis order D,H,W; accept a single volume only."""
    label = np.load(path, allow_pickle=False)
    if label.ndim == 5 and label.shape[0] == 1:
        label = label[0]
    if label.ndim == 4 and label.shape[0] in (1, 2):
        label = label[-1]
    if label.ndim != 3 or any(size == 0 for size in label.shape):
        raise ValueError("Expected [D,H,W], [1/2,D,H,W], or [1,1/2,D,H,W] for one label volume.")
    if not np.isfinite(label).all():
        raise ValueError("Label contains NaN or infinity.")
    return (label > 0.5).astype(np.uint8)


def display_normals(normal):
    """Choose a sign for display only. Raw saved PCA normals are untouched."""
    dominant = np.abs(normal).argmax(axis=0, keepdims=True)
    component = np.take_along_axis(normal, dominant, axis=0)[0]
    return normal * np.where(component < 0, -1.0, 1.0)[None]


def choose_slices(label):
    result = []
    for axis in range(3):
        counts = label.sum(axis=tuple(a for a in range(3) if a != axis))
        candidates = np.flatnonzero(counts == counts.max())
        result.append(int(candidates[np.argmin(np.abs(candidates - (label.shape[axis] - 1) / 2))]))
    return tuple(result)


def slice_view(array, axis, index):
    # For a D slice: vertical H, horizontal W; H slice: D,W; W slice: D,H.
    return np.take(array, index, axis=axis)


def limit_indices(coords, maximum):
    if len(coords) <= maximum:
        return coords
    return coords[np.random.default_rng(42).choice(len(coords), maximum, replace=False)]


def slice_layout(ax, shape, axis, spacing):
    remaining = [a for a in range(3) if a != axis]
    vertical, horizontal = remaining
    names = ("D", "H", "W")
    ax.set_xlabel(names[horizontal] + " coordinate")
    ax.set_ylabel(names[vertical] + " coordinate")
    ax.set_facecolor("#f1f4f8")
    extent = (-0.5 * spacing[horizontal], (shape[horizontal] - 0.5) * spacing[horizontal],
              (shape[vertical] - 0.5) * spacing[vertical], -0.5 * spacing[vertical])
    return vertical, horizontal, extent


def plot_3d(plt, label, normal, confidence, args):
    fig = plt.figure(figsize=(10, 8), facecolor="white", layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    points = limit_indices(np.argwhere(label > 0), args.max_points)
    if len(points):
        coords = points * np.asarray(args.spacing)
        ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0],
                   c=confidence[tuple(points.T)], cmap="viridis", vmin=0, vmax=1,
                   s=3, alpha=0.40, linewidths=0, depthshade=False)
    supported = (label > 0) & (confidence > 1e-6) & (confidence >= args.min_confidence)
    lattice = np.zeros_like(label, dtype=bool)
    lattice[::args.arrow_step, ::args.arrow_step, ::args.arrow_step] = True
    # A thin sheet can miss the lattice entirely, so fall back to foreground sampling.
    arrows = np.argwhere(supported & lattice)
    if not len(arrows):
        arrows = np.argwhere(supported)
    arrows = limit_indices(arrows, args.max_arrows)
    if len(arrows):
        coords = arrows * np.asarray(args.spacing)
        vectors = normal[(slice(None),) + tuple(arrows.T)].T
        ax.quiver(coords[:, 2], coords[:, 1], coords[:, 0],
                  vectors[:, 2], vectors[:, 1], vectors[:, 0],
                  length=args.arrow_length, normalize=False, pivot="middle",
                  color="#dd522c", linewidth=0.85, arrow_length_ratio=0.25)
    bounds = (np.asarray(label.shape) - 1) * np.asarray(args.spacing)
    ax.set_xlim(-0.5, max(bounds[2], 0.5))
    ax.set_ylim(-0.5, max(bounds[1], 0.5))
    ax.set_zlim(max(bounds[0], 0.5), -0.5)
    ax.set_box_aspect(np.maximum(bounds[[2, 1, 0]], 1))
    ax.set_xlabel("W coordinate")
    ax.set_ylabel("H coordinate")
    ax.set_zlabel("D coordinate")
    ax.view_init(elev=22, azim=-55)
    ax.set_title("GT fault surface and local normals", fontsize=16, pad=18)
    scalar = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis")
    fig.colorbar(scalar, ax=ax, shrink=0.65, pad=0.07, label="Local plane confidence")
    fig.suptitle(f"{args.label.stem}  |  {len(points):,} displayed GT points, {len(arrows)} normal arrows\n"
                 "Orange arrows: normal axis (sign is arbitrary); background has no normals",
                 fontsize=10, color="#475569")
    return fig, len(points), len(arrows)


def plot_normal_slices(plt, label, normal, confidence, slices, args):
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), layout="constrained")
    for axis, (index, ax) in enumerate(zip(slices, axes)):
        vertical, horizontal, extent = slice_layout(ax, label.shape, axis, args.spacing)
        gt_slice = slice_view(label, axis, index)
        conf_slice = slice_view(confidence, axis, index)
        ax.imshow(gt_slice, cmap=ListedColormap(["#f1f4f8", "#92b4d2"]),
                  vmin=0, vmax=1, interpolation="nearest", extent=extent, origin="upper")
        selected = (gt_slice > 0) & (conf_slice > 1e-6) & (conf_slice >= args.min_confidence)
        lattice = np.zeros_like(gt_slice, dtype=bool)
        lattice[::args.arrow_step, ::args.arrow_step] = True
        positions = np.argwhere(selected & lattice)
        if not len(positions):
            positions = np.argwhere(selected)
        positions = limit_indices(positions, args.max_arrows)
        if len(positions):
            row, col = positions.T
            h = slice_view(normal[horizontal], axis, index)[row, col]
            v = slice_view(normal[vertical], axis, index)[row, col]
            horizontal_position, vertical_position = col * args.spacing[horizontal], row * args.spacing[vertical]
            visible = np.hypot(h, v) >= 0.1
            ax.quiver(horizontal_position[visible], vertical_position[visible],
                      h[visible] * args.arrow_length, v[visible] * args.arrow_length,
                      color="#bd3d22", angles="xy", scale_units="xy", scale=1,
                      pivot="middle", width=0.004, headwidth=3)
            ax.scatter(horizontal_position[~visible], vertical_position[~visible],
                       facecolors="none", edgecolors="#bd3d22", s=16, linewidths=0.8)
        ax.set_title(f"{'DHW'[axis]} = {index}  |  projected normal", fontsize=12)
    fig.suptitle("Local normal directions on three orthogonal slices\n"
                 "Red arrow: in-plane projection; open circle: normal nearly perpendicular to the slice",
                 fontsize=13)
    return fig


def plot_confidence(plt, label, confidence, slices, args):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f1f4f8")
    for axis, (index, ax) in enumerate(zip(slices, axes)):
        _, _, extent = slice_layout(ax, label.shape, axis, args.spacing)
        gt_slice = slice_view(label, axis, index)
        confidence_slice = slice_view(confidence, axis, index)
        masked = np.ma.masked_where(gt_slice == 0, confidence_slice)
        display = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1,
                            interpolation="nearest", extent=extent, origin="upper")
        mean = confidence_slice[gt_slice > 0].mean() if gt_slice.any() else 0
        ax.set_title(f"{'DHW'[axis]} = {index}  |  GT mean confidence {mean:.3f}", fontsize=12)
    fig.colorbar(display, ax=list(axes), shrink=0.75, pad=0.02, label="Local plane confidence [0, 1]")
    fig.suptitle("GT local-plane confidence weights\n"
                 "Light gray: background (stored as zero); purple: unreliable GT geometry",
                 fontsize=13)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label", type=Path, help="Synthetic binary GT .npy file; original D,H,W axis order")
    parser.add_argument("--output", type=Path, help="Output folder (default: visualization_outputs/<label name>)")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda:0")
    parser.add_argument("--normal-kernel-size", type=int, default=5)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1., 1., 1.), metavar=("D", "H", "W"))
    parser.add_argument("--slices", type=int, nargs=3, metavar=("D", "H", "W"), help="Indices; default selects foreground-rich slices")
    parser.add_argument("--min-confidence", type=float, default=0.2, help="Only hide unreliable arrows; exported arrays retain everything")
    parser.add_argument("--arrow-step", type=int, default=8, help="Sparse arrow grid spacing in voxels")
    parser.add_argument("--arrow-length", type=float, default=5., help="3D arrow length in coordinate units")
    parser.add_argument("--max-arrows", type=int, default=250)
    parser.add_argument("--max-points", type=int, default=25000)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--show", action="store_true", help="Open Matplotlib figures after export; rotate 3D with mouse")
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be in [0,1]")
    if min(args.arrow_step, args.max_arrows, args.max_points, args.dpi) < 1 or not np.isfinite(args.arrow_length) or args.arrow_length <= 0:
        parser.error("Arrow spacing/count/length, point count, and dpi must be positive")
    label = load_label(args.label)
    slices = tuple(args.slices) if args.slices is not None else choose_slices(label)
    if any(index < 0 or index >= size for index, size in zip(slices, label.shape)):
        parser.error("--slices indices must be within the D,H,W label dimensions")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    criterion = SurfaceGuidedBreakMergeLoss(normal_kernel_size=args.normal_kernel_size,
                                          min_points=args.min_points,
                                          voxel_spacing=args.spacing).to(args.device)
    geometry = criterion.estimate_geometry(torch.from_numpy(label[None, None]).to(args.device))
    normal = geometry["normal"][0].cpu().numpy()
    confidence = geometry["confidence"][0, 0].cpu().numpy()
    output = args.output or ROOT / "visualization_outputs" / args.label.stem
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "normals.npy", normal)
    np.save(output / "confidence.npy", confidence)
    n_display = display_normals(normal)
    surface, point_count, arrow_count = plot_3d(plt, label, n_display, confidence, args)
    figures = (("surface_normals_3d.png", surface),
               ("normal_slices.png", plot_normal_slices(plt, label, n_display, confidence, slices, args)),
               ("confidence_slices.png", plot_confidence(plt, label, confidence, slices, args)))
    for filename, fig in figures:
        fig.savefig(output / filename, dpi=args.dpi, facecolor="white")
    gt_conf = confidence[label > 0]
    metadata = {"label": str(args.label.resolve()), "shape_DHW": list(label.shape),
                "normal_channels": ["n_D", "n_H", "n_W"], "spacing_DHW": list(args.spacing),
                "slices_DHW": list(slices), "normal_kernel_size": args.normal_kernel_size,
                "min_points": args.min_points, "gt_voxels": int(label.sum()),
                "mean_GT_confidence": float(gt_conf.mean()) if gt_conf.size else None,
                "displayed_points": point_count, "displayed_arrows": arrow_count,
                "arrow_min_confidence": args.min_confidence,
                "note": "PCA normal signs are arbitrary; display signs canonicalized only. Background geometry is zero. Confidence is not a directional pair-loss weight."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "shape_DHW": list(label.shape),
                      "slices_DHW": slices, "gt_voxels": int(label.sum()), "displayed_arrows": arrow_count}, ensure_ascii=False))
    if args.show:
        plt.show()
    else:
        for _, fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
