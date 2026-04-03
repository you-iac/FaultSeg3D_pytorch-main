import datetime
import tkinter as tk
from tkinter import filedialog

import cigvis
from cigvis import colormap
from cigvis.vispynodes import XYZAxis
from cigvis.vispynodes.vis_canvas import VisCanvas
import numpy as np
import vispy
from vispy.gloo.util import _screenshot


def select_file(title: str) -> str | None:
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


def safe_slice_pos(shape: tuple[int, int, int]) -> list[list[int]]:
    return [[shape[0] // 2], [shape[1] // 2], [shape[2] // 2]]


def bind_mouse_xyz(canvas: VisCanvas, base_title: str = "Seismic3D") -> None:
    def _on_mouse_move(event):
        hover_on = canvas.visual_at(event.pos)
        if not hasattr(hover_on, "get_click_pos3d"):
            canvas.title = base_title
            return
        try:
            pos3d = hover_on.get_click_pos3d(event)
            if pos3d is None or len(pos3d) < 3:
                canvas.title = base_title
                return
            x, y, z = [int(round(float(v))) for v in pos3d[:3]]
            canvas.title = f"{base_title} | X={x} Y={y} Z={z}"
        except Exception:
            canvas.title = base_title
            return

    canvas.events.mouse_move.connect(_on_mouse_move)


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

    fg_cmap = colormap.set_alpha_except_min("jet", alpha=1)
    pos = safe_slice_pos(x.shape)
    nodes = cigvis.create_slices(x, pos=pos, cmap="gray")
    nodes = cigvis.add_mask(nodes, y, cmaps=fg_cmap, interpolation="nearest")
    nodes += cigvis.create_colorbar_from_nodes(nodes, "Amplitude", select="slices")
    nodes.append(XYZAxis())

    canvas = VisCanvas(
        visual_nodes=nodes,
        size=(700, 600),
        title="Seismic3D",
        azimuth=30,
        elevation=40,
    )
    bind_mouse_xyz(canvas, base_title="Seismic3D")
    canvas.show()

    out_name = "seis_fault_3d_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    screenshot = _screenshot()
    vispy.io.write_png(out_name, screenshot)
    print(f"可视化已保存: {out_name}")
    print("提示: 鼠标移动到切片上，窗口标题栏会实时显示 XYZ 坐标。")

    vispy.app.run()


if __name__ == "__main__":
    main()
