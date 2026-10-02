import datetime
import tkinter as tk
from tkinter import filedialog

import cigvis
from cigvis import colormap
from cigvis.vispynodes.vis_canvas import VisCanvas
import numpy as np
import vispy
from vispy.color import Colormap
from vispy.gloo.util import _screenshot


from typing import Optional


# 与“地震+预测切片显示.py”保持一致：把地震数据限制在中灰区域，
# 保留纹理但降低其视觉权重，从而突出断层。
SEISMIC_GRAY_MIN = 0.32
SEISMIC_GRAY_MAX = 0.68
SEISMIC_PERCENTILE_LOW = 1.0
SEISMIC_PERCENTILE_HIGH = 99.0
FAULT_COLOR = "#8B0000"
FIXED_SLICE_POS = (40, 40, 400)  # Inline, Xline, Time 的数组下标

def select_file(title: str) -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("NumPy文件", "*.npy"), ("所有文件", "*.*")],
    )
    root.destroy()
    return file_path or None


def load_volume(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"文件不是3D数据: {path}, shape={arr.shape}")
    return np.asarray(arr).transpose((1, 2, 0)).astype(np.float32, copy=False)


def fixed_slice_pos(shape: tuple[int, int, int]) -> list[list[int]]:
    """返回固定的 Inline、Xline、Time 切片，并检查是否越界。"""
    for axis, value, size in zip(
        ("Inline", "Xline", "Time"),
        FIXED_SLICE_POS,
        shape,
    ):
        if not 0 <= value < size:
            raise ValueError(
                f"{axis} 切片像素号 {value} 越界，有效范围为 0～{size - 1}。"
            )
    return [[value] for value in FIXED_SLICE_POS]


def create_seismic_cmap() -> Colormap:
    """创建完全不透明、低对比度的中灰色地震色表。"""
    gray_min = float(np.clip(SEISMIC_GRAY_MIN, 0.0, 1.0))
    gray_max = float(np.clip(SEISMIC_GRAY_MAX, gray_min, 1.0))
    return Colormap(
        [
            (gray_min, gray_min, gray_min, 1.0),
            (gray_max, gray_max, gray_max, 1.0),
        ]
    )


def create_fault_cmap() -> Colormap:
    """零值透明、断层像素为完全不透明暗红色。"""
    solid_red = Colormap([FAULT_COLOR, FAULT_COLOR])
    return colormap.set_alpha_except_min(solid_red, alpha=1.0)


def seismic_clim(volume: np.ndarray) -> list[float]:
    """按参考脚本的 1%～99% 分位数计算地震显示范围。"""
    low, high = np.nanpercentile(
        volume,
        [SEISMIC_PERCENTILE_LOW, SEISMIC_PERCENTILE_HIGH],
    )
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low, high = np.nanmin(volume), np.nanmax(volume)
    return [float(low), float(high)]


def get_camera_params(canvas: VisCanvas) -> dict[str, object]:
    """读取当前主视图的全部可复用相机参数。"""
    camera = canvas.view[0].camera
    center = tuple(float(v) for v in np.asarray(camera.center).ravel()[:3])
    return {
        "azimuth": float(camera.azimuth),
        "elevation": float(camera.elevation),
        "fov": float(camera.fov),
        "scale_factor": float(camera.scale_factor),
        "center": center,
    }


def bind_camera_title(
    canvas: VisCanvas,
    base_title: str = "Seismic3D",
) -> vispy.app.Timer:
    """在窗口标题中实时显示视角；按 P 输出可复制的相机参数。"""
    last_title = ""

    def _update_title(_event=None) -> None:
        nonlocal last_title
        params = get_camera_params(canvas)
        cx, cy, cz = params["center"]
        title = (
            f"{base_title} | az={params['azimuth']:.1f}° "
            f"el={params['elevation']:.1f}° fov={params['fov']:.1f}° "
            f"scale={params['scale_factor']:.2f} "
            f"center=({cx:.1f}, {cy:.1f}, {cz:.1f})"
        )
        if title != last_title:
            canvas.title = title
            last_title = title

    def _on_key_press(event) -> None:
        key_name = getattr(event.key, "name", str(event.key)).upper()
        if key_name != "P":
            return
        params = get_camera_params(canvas)
        print("\n当前视角参数（复制到其他三个 VisCanvas 中即可）:")
        print(
            f"azimuth={params['azimuth']:.6f}, "
            f"elevation={params['elevation']:.6f}, "
            f"fov={params['fov']:.6f}, "
            f"scale_factor={params['scale_factor']:.6f}, "
            f"center={params['center']},"
        )

    canvas.events.key_press.connect(_on_key_press)
    timer = vispy.app.Timer(interval=0.1, connect=_update_title, start=True)
    _update_title()
    return timer


def main() -> None:
    file_path_x = select_file("选择地震数据文件")
    file_path_y = select_file("选择断层数据文件")
    if not file_path_x or not file_path_y:
        print("已取消文件选择，程序结束。")
        return

    x = load_volume(file_path_x)
    y = load_volume(file_path_y)
    if x.shape != y.shape:
        raise ValueError(f"地震与断层形状不一致: x={x.shape}, y={y.shape}")

    pos = fixed_slice_pos(x.shape)
    bg_cmap = create_seismic_cmap()
    fg_cmap = create_fault_cmap()
    bg_clim = seismic_clim(x)

    def create_slice_nodes(slice_pos: list[list[int]]) -> list[object]:
        created = cigvis.create_slices(
            x,
            pos=slice_pos,
            cmap=bg_cmap,
            clim=bg_clim,
            interpolation="nearest",
        )
        return cigvis.add_mask(
            created,
            y,
            cmaps=fg_cmap,
            interpolation="nearest",
        )

    nodes = create_slice_nodes(pos)
    nodes += cigvis.create_colorbar_from_nodes(nodes, "Amplitude", select="slices")
    nodes += cigvis.create_axis(
        x.shape,
        mode="axis",
        axis_pos=[3, 3, 1],
        tick_nums=5,
        starts=[0, 0, 0],
        axis_labels=["Inline", "Xline", "Time"],
    )

    canvas = VisCanvas(
        visual_nodes=nodes,
        size=(700, 600),
        title="Seismic3D",
        bgcolor="white",
        azimuth=30,
        elevation=40,
    )
    camera_title_timer = bind_camera_title(canvas, base_title="Seismic3D")
    canvas.show()

    out_name = "seis_fault_3d_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    screenshot = _screenshot()
    vispy.io.write_png(out_name, screenshot)
    print(f"可视化已保存: {out_name}")
    print(
        f"显示数据尺寸: Inline={x.shape[0]}, "
        f"Xline={x.shape[1]}, Time={x.shape[2]}"
    )
    print(
        f"固定切片像素号: Inline={pos[0][0]}, "
        f"Xline={pos[1][0]}, Time={pos[2][0]}"
    )
    print(
        "提示: 切片输入功能已移除；按 P 可打印完整视角参数。"
    )

    vispy.app.run()
    camera_title_timer.stop()


if __name__ == "__main__":
    main()
