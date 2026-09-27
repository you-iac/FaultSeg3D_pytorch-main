"""Visualize the validation slices where U-Net and ground truth disagree most.

For each 3D sample, inference is performed first.  The script then compares the
binary prediction with the label on every D/H/W slice and independently selects
the slice with the largest disagreement score in each direction.  The default
score is XOR voxel count (false positives + false negatives).

Each output PNG contains four matched 3D cut-away views:
seismic volume, ground truth, U-Net prediction, and FP/FN disagreement.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SUF_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SUF_ROOT))
import visualize_unet_valid_3d as base


DEFAULT_DATA = ROOT / "data" / "data_3D_400" / "valid"
DEFAULT_MODEL = SUF_ROOT / "00_Baseline_Unet_8"
DEFAULT_OUTPUT = SUF_ROOT / "Unet_valid_3D_difference_results"


def difference_mask(
    prediction: np.ndarray,
    target: np.ndarray,
    mode: str,
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if mode == "xor":
        return np.logical_xor(prediction, target)
    if mode == "false-positive":
        return prediction & ~target
    if mode == "false-negative":
        return target & ~prediction
    raise ValueError(f"Unsupported difference mode: {mode}")


def centered_argmax(scores: np.ndarray, edge_margin: int = 0) -> int:
    """Use the closest-to-centre slice as deterministic tie breaker."""
    scores = np.asarray(scores)
    if edge_margin < 0 or edge_margin * 2 >= len(scores):
        raise ValueError(
            f"edge_margin={edge_margin} is invalid for axis length {len(scores)}"
        )
    valid_scores = scores[edge_margin : len(scores) - edge_margin]
    candidates = np.flatnonzero(valid_scores == valid_scores.max()) + edge_margin
    center = (len(scores) - 1) / 2.0
    return int(min(candidates, key=lambda index: (abs(index - center), index)))


def keep_interior(mask: np.ndarray, edge_margin: int) -> np.ndarray:
    """Remove the boundary shell from a 3D boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    if edge_margin < 0 or any(edge_margin * 2 >= size for size in mask.shape):
        raise ValueError(
            f"edge_margin={edge_margin} is invalid for volume shape {mask.shape}"
        )
    if edge_margin == 0:
        return mask.copy()
    interior = np.zeros_like(mask, dtype=bool)
    interior[
        edge_margin:-edge_margin,
        edge_margin:-edge_margin,
        edge_margin:-edge_margin,
    ] = mask[
        edge_margin:-edge_margin,
        edge_margin:-edge_margin,
        edge_margin:-edge_margin,
    ]
    return interior


def choose_max_difference_slices(
    prediction: np.ndarray,
    target: np.ndarray,
    mode: str = "xor",
    edge_margin: int = 8,
) -> tuple[tuple[int, int, int], tuple[int, int, int], np.ndarray]:
    """Select D/H/W slices with maximum prediction-label disagreement."""
    mask = keep_interior(difference_mask(prediction, target, mode), edge_margin)
    scores_d = mask.sum(axis=(1, 2), dtype=np.int64)
    scores_h = mask.sum(axis=(0, 2), dtype=np.int64)
    scores_w = mask.sum(axis=(0, 1), dtype=np.int64)
    d0 = centered_argmax(scores_d, edge_margin)
    h0 = centered_argmax(scores_h, edge_margin)
    w0 = centered_argmax(scores_w, edge_margin)
    return (
        (d0, h0, w0),
        (int(scores_d[d0]), int(scores_h[h0]), int(scores_w[w0])),
        mask,
    )


def difference_facecolors(
    gray: np.ndarray,
    false_positive: np.ndarray,
    false_negative: np.ndarray,
) -> np.ndarray:
    """Paint FP red and FN blue directly into an opaque seismic texture."""
    rgb = colormaps["gray"](gray)[..., :3]
    red = np.array([1.0, 0.05, 0.01], dtype=np.float64)
    blue = np.array([0.02, 0.35, 1.0], dtype=np.float64)
    alpha = 0.90
    fp = np.asarray(false_positive, dtype=bool)
    fn = np.asarray(false_negative, dtype=bool)
    rgb[fp] = (1.0 - alpha) * rgb[fp] + alpha * red
    rgb[fn] = (1.0 - alpha) * rgb[fn] + alpha * blue
    return rgb


def overlay_difference_planes(
    ax,
    seismic: np.ndarray,
    norm,
    false_positive: np.ndarray,
    false_negative: np.ndarray,
    positions: tuple[int, int, int],
    view_sides: tuple[int, int, int],
    render_stride: int,
) -> None:
    """Render FP/FN as face colours to avoid 3D scatter depth-sorting loss."""
    d0, h0, w0 = positions
    d, h, w = seismic.shape
    side_d, side_h, side_w = view_sides
    ds = base.indices_to_boundary(d0, d, side_d, render_stride)
    hs = base.indices_to_boundary(h0, h, side_h, render_stride)
    ws = base.indices_to_boundary(w0, w, side_w, render_stride)

    # D plane.
    xx, yy = np.meshgrid(ws, hs)
    zz = np.full_like(xx, d0, dtype=float) + side_d * 0.30
    gray = norm(seismic[d0][np.ix_(hs, ws)])
    fp = false_positive[d0][np.ix_(hs, ws)]
    fn = false_negative[d0][np.ix_(hs, ws)]
    ax.plot_surface(
        xx, yy, zz, facecolors=difference_facecolors(gray, fp, fn),
        rstride=1, cstride=1, shade=False, antialiased=False,
    )

    # H plane.
    xx, zz = np.meshgrid(ws, ds)
    yy = np.full_like(xx, h0, dtype=float) + side_h * 0.30
    gray = norm(seismic[:, h0, :][np.ix_(ds, ws)])
    fp = false_positive[:, h0, :][np.ix_(ds, ws)]
    fn = false_negative[:, h0, :][np.ix_(ds, ws)]
    ax.plot_surface(
        xx, yy, zz, facecolors=difference_facecolors(gray, fp, fn),
        rstride=1, cstride=1, shade=False, antialiased=False,
    )

    # W plane.
    yy, zz = np.meshgrid(hs, ds)
    xx = np.full_like(yy, w0, dtype=float) + side_w * 0.30
    gray = norm(seismic[:, :, w0][np.ix_(ds, hs)])
    fp = false_positive[:, :, w0][np.ix_(ds, hs)]
    fn = false_negative[:, :, w0][np.ix_(ds, hs)]
    ax.plot_surface(
        xx, yy, zz, facecolors=difference_facecolors(gray, fp, fn),
        rstride=1, cstride=1, shade=False, antialiased=False,
    )


def render_difference_result(
    seismic: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    sample_name: str,
    output_path: Path,
    selection_mode: str,
    edge_margin: int,
    render_stride: int,
) -> dict[str, object]:
    positions, slice_scores, selected_mask = choose_max_difference_slices(
        prediction, target, selection_mode, edge_margin
    )
    view_angle, view_sides, visible_lengths = base.choose_view_configuration(
        positions, seismic.shape
    )
    norm = base.robust_normalizer(seismic)
    iou, dice = base.binary_metrics(prediction, target)
    full_false_positive = prediction & ~target
    full_false_negative = target & ~prediction
    false_positive = keep_interior(full_false_positive, edge_margin)
    false_negative = keep_interior(full_false_negative, edge_margin)

    fig = plt.figure(figsize=(22.5, 6.2), dpi=160, facecolor="white")
    axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]
    base.draw_volume_corner(
        axes[0], seismic, None, positions, norm, "Seismic volume",
        render_stride, view_angle, view_sides,
    )
    base.draw_volume_corner(
        axes[1], seismic, target, positions, norm, "Ground truth",
        render_stride, view_angle, view_sides,
    )
    base.draw_volume_corner(
        axes[2], seismic, prediction, positions, norm, "U-Net prediction",
        render_stride, view_angle, view_sides,
    )
    base.draw_volume_corner(
        axes[3], seismic, None, positions, norm, "Difference: FP / FN",
        render_stride, view_angle, view_sides,
    )
    overlay_difference_planes(
        axes[3],
        seismic,
        norm,
        false_positive,
        false_negative,
        positions,
        view_sides,
        render_stride,
    )

    fig.suptitle(
        f"Validation sample {sample_name}   |   fault IoU {iou:.4f}   "
        f"fault Dice {dice:.4f}   |   interior FP {int(false_positive.sum()):,}   "
        f"interior FN {int(false_negative.sum()):,}",
        fontsize=14,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        f"Maximum {selection_mode} slices: D={positions[0]} ({slice_scores[0]:,}), "
        f"H={positions[1]} ({slice_scores[1]:,}), W={positions[2]} ({slice_scores[2]:,})   "
        f"|   excluded boundary={edge_margin} voxels/side   "
        f"|   view: elevation={view_angle[0]:.1f}°, azimuth={view_angle[1]:.1f}°   "
        "|   orange=label/prediction, red=FP, blue=FN",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.subplots_adjust(left=0.008, right=0.992, bottom=0.055, top=0.91, wspace=0.015)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    return {
        "sample": sample_name,
        "selection_mode": selection_mode,
        "edge_margin": edge_margin,
        "fault_iou": f"{iou:.6f}",
        "fault_dice": f"{dice:.6f}",
        "full_false_positive": int(full_false_positive.sum()),
        "full_false_negative": int(full_false_negative.sum()),
        "interior_false_positive": int(false_positive.sum()),
        "interior_false_negative": int(false_negative.sum()),
        "interior_selected_difference": int(selected_mask.sum()),
        "slice_d": positions[0],
        "slice_d_difference": slice_scores[0],
        "slice_h": positions[1],
        "slice_h_difference": slice_scores[1],
        "slice_w": positions[2],
        "slice_w_difference": slice_scores[2],
        "view_elevation": f"{view_angle[0]:.1f}",
        "view_azimuth": f"{view_angle[1]:.1f}",
        "cutaway_d": "+" if view_sides[0] > 0 else "-",
        "cutaway_h": "+" if view_sides[1] > 0 else "-",
        "cutaway_w": "+" if view_sides[2] > 0 else "-",
        "visible_length_d": visible_lengths[0],
        "visible_length_h": visible_lengths[1],
        "visible_length_w": visible_lengths[2],
        "mean_predicted_fault_probability": (
            f"{float(probability[prediction].mean()):.6f}" if prediction.any() else "0.000000"
        ),
        "image": output_path.name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument(
        "--edge-margin",
        type=int,
        default=8,
        help="Ignore this many voxels at both ends of every D/H/W axis",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("xor", "false-positive", "false-negative"),
        default="xor",
        help="Per-slice score used to select D/H/W positions",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda:0"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    x_dir = args.data / "x"
    y_dir = args.data / "y"
    x_files = sorted(x_dir.glob("*.npy"), key=base.numeric_key)[: args.limit]
    if not x_files:
        raise RuntimeError(f"No validation inputs found in {x_dir}")
    pairs = [(x_path, y_dir / x_path.name) for x_path in x_files]
    missing = [str(y_path) for _, y_path in pairs if not y_path.is_file()]
    if missing:
        raise FileNotFoundError("Missing labels: " + ", ".join(missing))

    args.output.mkdir(parents=True, exist_ok=True)
    model = base.load_model(args.model_dir, device)
    records: list[dict[str, object]] = []

    with torch.inference_mode():
        for index, (x_path, y_path) in enumerate(pairs, start=1):
            seismic = np.load(x_path).astype(np.float32, copy=False)
            target = np.load(y_path) > 0.5
            if seismic.ndim != 3 or target.shape != seismic.shape:
                raise ValueError(f"Invalid pair {x_path.name}: x={seismic.shape}, y={target.shape}")

            tensor = torch.from_numpy(np.ascontiguousarray(seismic))[None, None].to(device)
            output = model(tensor)
            probability = output[0, 1].detach().cpu().numpy()
            prediction = probability >= args.threshold
            output_path = args.output / f"{x_path.stem}_unet_difference_3d.png"
            record = render_difference_result(
                seismic,
                target,
                prediction,
                probability,
                x_path.stem,
                output_path,
                args.selection_mode,
                args.edge_margin,
                max(1, args.render_stride),
            )
            records.append(record)
            print(
                f"[{index:02d}/{len(pairs):02d}] {x_path.name}: "
                f"D/H/W={record['slice_d']}/{record['slice_h']}/{record['slice_w']}, "
                f"interior FP={record['interior_false_positive']}, "
                f"FN={record['interior_false_negative']} "
                f"-> {output_path.name}",
                flush=True,
            )
            del tensor, output

    csv_path = args.output / "difference_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f"Completed {len(records)} samples on {device} using {args.selection_mode} selection.")
    print(f"Images and metrics: {args.output}")


if __name__ == "__main__":
    main()
