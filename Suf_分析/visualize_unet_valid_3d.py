"""Run baseline 3D U-Net on validation data and render real 3D result figures.

Each PNG contains three views built from the same three orthogonal seismic
planes: input volume, ground-truth faults, and U-Net prediction.  Faults are
drawn from the actual 3D label/prediction arrays, not from a 2D projection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SUF_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "data_3D_400" / "valid"
DEFAULT_MODEL = SUF_ROOT / "00_Baseline_Unet_8"
DEFAULT_OUTPUT = SUF_ROOT / "Unet_valid_3D_results"


def numeric_key(path: Path) -> tuple[int, int | str]:
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.name


def import_model(model_py: Path):
    module_name = "unet_valid_vis_" + hashlib.md5(str(model_py).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, model_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model module: {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(model_dir: Path, device: torch.device) -> torch.nn.Module:
    model_py = model_dir / "models.py"
    checkpoint = model_dir / "models" / "FaultSeg3D_BEST.pth"
    if not model_py.is_file():
        raise FileNotFoundError(model_py)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    module = import_model(model_py)
    model = module.FaultSeg3D(1, 2)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def choose_fault_rich_slices(label: np.ndarray) -> tuple[int, int, int]:
    """Choose one interior, fault-rich plane for each D/H/W axis."""
    if label.ndim != 3:
        raise ValueError(f"Expected a 3D label, got {label.shape}")

    positions: list[int] = []
    for axis, size in enumerate(label.shape):
        counts = label.sum(axis=tuple(i for i in range(3) if i != axis))
        margin = max(1, size // 10)
        interior = counts[margin : size - margin]
        if interior.size == 0 or float(interior.max()) <= 0:
            positions.append(size // 2)
        else:
            positions.append(int(np.argmax(interior)) + margin)
    return positions[0], positions[1], positions[2]


def choose_view_configuration(
    positions: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> tuple[tuple[float, float], tuple[int, int, int], tuple[int, int, int]]:
    """Choose the largest cut-away octant and maximize its projected face area.

    Each selected plane splits the volume into two sides.  The side with the
    longer span is chosen independently for D/H/W, yielding the largest of the
    eight cuboids anchored at the plane intersection.  If its face areas are
    Ax, Ay and Az, the viewing vector proportional to (Ax, Ay, Az) maximizes
    Ax*|vx| + Ay*|vy| + Az*|vz| under a unit-vector constraint.  This gives a
    real per-sample camera angle instead of merely rotating symmetric planes.
    """
    d0, h0, w0 = positions
    d, h, w = shape

    def longer_side(position: int, size: int) -> tuple[int, int]:
        low_span = position + 1
        high_span = size - position
        return (-1, low_span) if low_span >= high_span else (1, high_span)

    side_d, length_d = longer_side(d0, d)
    side_h, length_h = longer_side(h0, h)
    side_w, length_w = longer_side(w0, w)

    # Face areas whose normals point along W(X), H(Y), and D(Z).
    area_x = float(length_h * length_d)
    area_y = float(length_w * length_d)
    area_z = float(length_w * length_h)
    view_x = side_w * area_x
    view_y = side_h * area_y
    # D=0 is displayed at the top, hence the sign reversal for elevation.
    view_z = -side_d * area_z
    azimuth = math.degrees(math.atan2(view_y, view_x))
    elevation = math.degrees(math.atan2(view_z, math.hypot(view_x, view_y)))
    return (
        (elevation, azimuth),
        (side_d, side_h, side_w),
        (length_d, length_h, length_w),
    )


def indices_to_boundary(position: int, size: int, side: int, stride: int) -> np.ndarray:
    """Return render indices from a selected plane to one volume boundary."""
    if side < 0:
        values = np.arange(0, position + 1, stride, dtype=int)
        if values[-1] != position:
            values = np.append(values, position)
    else:
        values = np.arange(position, size, stride, dtype=int)
        if values[-1] != size - 1:
            values = np.append(values, size - 1)
    return values


def robust_normalizer(volume: np.ndarray) -> Normalize:
    low, high = np.percentile(volume, (2.0, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(volume)), float(np.nanmax(volume))
    if high <= low:
        high = low + 1.0
    return Normalize(vmin=float(low), vmax=float(high), clip=True)


def colored_plane(gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    rgb = colormaps["gray"](gray)[..., :3]
    if mask is None:
        return rgb
    fault = np.asarray(mask, dtype=bool)
    # Orange-red preserves the seismic texture while keeping faults prominent.
    color = np.array([1.0, 0.16, 0.01], dtype=np.float64)
    alpha = 0.78
    rgb[fault] = (1.0 - alpha) * rgb[fault] + alpha * color
    return rgb


def boundary_2d(mask: np.ndarray) -> np.ndarray:
    """Return a dependency-free four-neighbour boundary mask."""
    mask = np.asarray(mask, dtype=bool)
    core = mask.copy()
    core[1:, :] &= mask[:-1, :]
    core[:-1, :] &= mask[1:, :]
    core[:, 1:] &= mask[:, :-1]
    core[:, :-1] &= mask[:, 1:]
    return mask & ~core


def plot_fault_boundaries(
    ax,
    mask: np.ndarray,
    positions: tuple[int, int, int],
    index_ranges: tuple[np.ndarray, np.ndarray, np.ndarray],
    step: int,
) -> None:
    d0, h0, w0 = positions
    ds, hs, ws = index_ranges

    def keep(coords: np.ndarray, row_indices: np.ndarray, col_indices: np.ndarray) -> np.ndarray:
        if not coords.size:
            return coords
        selected = (
            (coords[:, 0] >= row_indices[0])
            & (coords[:, 0] <= row_indices[-1])
            & (coords[:, 1] >= col_indices[0])
            & (coords[:, 1] <= col_indices[-1])
        )
        return coords[selected][::step]

    # Constant-depth plane: mask[d0, H, W] -> X=W, Y=H, Z=D.
    coords = keep(np.argwhere(boundary_2d(mask[d0, :, :])), hs, ws)
    if coords.size:
        ax.scatter(coords[:, 1], coords[:, 0], d0, s=2.1, c="#ff2600", depthshade=False)

    # Constant-H plane: mask[D, h0, W] -> X=W, Y=H, Z=D.
    coords = keep(np.argwhere(boundary_2d(mask[:, h0, :])), ds, ws)
    if coords.size:
        ax.scatter(coords[:, 1], h0, coords[:, 0], s=2.1, c="#ff2600", depthshade=False)

    # Constant-W plane: mask[D, H, w0] -> X=W, Y=H, Z=D.
    coords = keep(np.argwhere(boundary_2d(mask[:, :, w0])), ds, hs)
    if coords.size:
        ax.scatter(w0, coords[:, 1], coords[:, 0], s=2.1, c="#ff2600", depthshade=False)


def draw_volume_corner(
    ax,
    seismic: np.ndarray,
    mask: np.ndarray | None,
    positions: tuple[int, int, int],
    norm: Normalize,
    title: str,
    stride: int,
    view_angle: tuple[float, float],
    view_sides: tuple[int, int, int],
) -> None:
    d, h, w = seismic.shape
    d0, h0, w0 = positions

    # Render only the largest cut-away cuboid. Inference and metrics still use
    # the complete volume at full resolution.
    side_d, side_h, side_w = view_sides
    ds = indices_to_boundary(d0, d, side_d, stride)
    hs = indices_to_boundary(h0, h, side_h, stride)
    ws = indices_to_boundary(w0, w, side_w, stride)

    # Horizontal plane (D=d0).
    xx, yy = np.meshgrid(ws, hs)
    zz = np.full_like(xx, d0)
    plane = norm(seismic[d0][np.ix_(hs, ws)])
    plane_mask = None if mask is None else mask[d0][np.ix_(hs, ws)]
    ax.plot_surface(
        xx, yy, zz, facecolors=colored_plane(plane, plane_mask),
        rstride=1, cstride=1, shade=False, antialiased=False,
    )

    # Vertical plane (H=h0).
    xx, zz = np.meshgrid(ws, ds)
    yy = np.full_like(xx, h0)
    plane = norm(seismic[:, h0, :][np.ix_(ds, ws)])
    plane_mask = None if mask is None else mask[:, h0, :][np.ix_(ds, ws)]
    ax.plot_surface(
        xx, yy, zz, facecolors=colored_plane(plane, plane_mask),
        rstride=1, cstride=1, shade=False, antialiased=False,
    )

    # Vertical plane (W=w0).
    yy, zz = np.meshgrid(hs, ds)
    xx = np.full_like(yy, w0)
    plane = norm(seismic[:, :, w0][np.ix_(ds, hs)])
    plane_mask = None if mask is None else mask[:, :, w0][np.ix_(ds, hs)]
    ax.plot_surface(
        xx, yy, zz, facecolors=colored_plane(plane, plane_mask),
        rstride=1, cstride=1, shade=False, antialiased=False,
    )

    if mask is not None:
        plot_fault_boundaries(
            ax,
            mask,
            positions,
            (ds, hs, ws),
            step=max(1, stride // 2),
        )

    # Edges of the selected cut-away cuboid make the chosen observation region
    # explicit and remove the misleading symmetry of three full planes.
    edge = "#4d4d4d"
    x_min, x_max = int(ws[0]), int(ws[-1])
    y_min, y_max = int(hs[0]), int(hs[-1])
    z_min, z_max = int(ds[0]), int(ds[-1])
    for x_values, y_values, z_values in (
        ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
        ([x_min, x_max], [y_max, y_max], [z_min, z_min]),
        ([x_min, x_max], [y_min, y_min], [z_max, z_max]),
        ([x_min, x_max], [y_max, y_max], [z_max, z_max]),
        ([x_min, x_min], [y_min, y_max], [z_min, z_min]),
        ([x_max, x_max], [y_min, y_max], [z_min, z_min]),
        ([x_min, x_min], [y_min, y_max], [z_max, z_max]),
        ([x_max, x_max], [y_min, y_max], [z_max, z_max]),
        ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
        ([x_min, x_min], [y_max, y_max], [z_min, z_max]),
        ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
    ):
        ax.plot(x_values, y_values, z_values, color=edge, linewidth=0.5, alpha=0.55)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_max, z_min)
    ax.set_box_aspect((x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1))
    elevation, azimuth = view_angle
    ax.view_init(elev=elevation, azim=azimuth)
    # The slice coordinates and camera angles are reported below the figure;
    # removing dense 3D tick labels keeps every camera octant equally legible.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title(title, fontsize=12, pad=4)
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    pred_count = int(prediction.sum())
    target_count = int(target.sum())
    iou = intersection / union if union else 1.0
    denominator = pred_count + target_count
    dice = 2.0 * intersection / denominator if denominator else 1.0
    return iou, dice


def render_result(
    seismic: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    sample_name: str,
    output_path: Path,
    stride: int,
) -> tuple[
    float,
    float,
    tuple[int, int, int],
    tuple[float, float],
    tuple[int, int, int],
    tuple[int, int, int],
]:
    positions = choose_fault_rich_slices(target)
    view_angle, view_sides, visible_lengths = choose_view_configuration(positions, seismic.shape)
    norm = robust_normalizer(seismic)
    iou, dice = binary_metrics(prediction, target)

    fig = plt.figure(figsize=(18, 6.2), dpi=160, facecolor="white")
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    draw_volume_corner(
        axes[0], seismic, None, positions, norm, "Seismic volume", stride, view_angle, view_sides
    )
    draw_volume_corner(
        axes[1], seismic, target, positions, norm, "Ground truth", stride, view_angle, view_sides
    )
    draw_volume_corner(
        axes[2], seismic, prediction, positions, norm, "U-Net prediction", stride, view_angle, view_sides
    )

    mean_fg_probability = float(probability[prediction].mean()) if prediction.any() else 0.0
    fig.suptitle(
        f"Validation sample {sample_name}   |   IoU {iou:.4f}   Dice {dice:.4f}   "
        f"mean predicted-fault probability {mean_fg_probability:.3f}",
        fontsize=14,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        f"Orthogonal slices: D={positions[0]}, H={positions[1]}, W={positions[2]}   "
        f"|   view: elevation={view_angle[0]:.1f}°, azimuth={view_angle[1]:.0f}°   "
        f"|   cut-away: D{'+' if view_sides[0] > 0 else '-'} "
        f"H{'+' if view_sides[1] > 0 else '-'} W{'+' if view_sides[2] > 0 else '-'}   "
        "|   red/orange = fault voxel",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.055, top=0.91, wspace=0.02)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return iou, dice, positions, view_angle, view_sides, visible_lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda:0" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    x_dir = args.data / "x"
    y_dir = args.data / "y"
    x_files = sorted(x_dir.glob("*.npy"), key=numeric_key)[: args.limit]
    if not x_files:
        raise RuntimeError(f"No validation inputs found in {x_dir}")
    pairs = [(x_path, y_dir / x_path.name) for x_path in x_files]
    missing = [str(y_path) for _, y_path in pairs if not y_path.is_file()]
    if missing:
        raise FileNotFoundError("Missing labels: " + ", ".join(missing))

    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_dir, device)
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
            output_path = args.output / f"{x_path.stem}_unet_3d.png"
            iou, dice, positions, view_angle, view_sides, visible_lengths = render_result(
                seismic,
                target,
                prediction,
                probability,
                x_path.stem,
                output_path,
                max(1, args.render_stride),
            )
            records.append(
                {
                    "sample": x_path.stem,
                    "iou": f"{iou:.6f}",
                    "dice": f"{dice:.6f}",
                    "predicted_fault_voxels": int(prediction.sum()),
                    "target_fault_voxels": int(target.sum()),
                    "slice_d": positions[0],
                    "slice_h": positions[1],
                    "slice_w": positions[2],
                    "view_elevation": f"{view_angle[0]:.1f}",
                    "view_azimuth": f"{view_angle[1]:.0f}",
                    "cutaway_d": "+" if view_sides[0] > 0 else "-",
                    "cutaway_h": "+" if view_sides[1] > 0 else "-",
                    "cutaway_w": "+" if view_sides[2] > 0 else "-",
                    "visible_length_d": visible_lengths[0],
                    "visible_length_h": visible_lengths[1],
                    "visible_length_w": visible_lengths[2],
                    "image": output_path.name,
                }
            )
            print(
                f"[{index:02d}/{len(pairs):02d}] {x_path.name}: "
                f"IoU={iou:.4f}, Dice={dice:.4f} -> {output_path.name}",
                flush=True,
            )
            del tensor, output

    csv_path = args.output / "metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    mean_iou = float(np.mean([float(row["iou"]) for row in records]))
    mean_dice = float(np.mean([float(row["dice"]) for row in records]))
    print(f"Completed {len(records)} samples on {device}. Mean IoU={mean_iou:.4f}, Dice={mean_dice:.4f}")
    print(f"Images and metrics: {args.output}")


if __name__ == "__main__":
    main()
