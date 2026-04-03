import cigvis
from cigvis import colormap
import numpy as np
import tkinter as tk
from tkinter import filedialog

from matplotlib import pyplot as plt
from matplotlib import style as mplstyle


def select_file(title="选择文件"):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("NumPy文件", "*.npy"), ("所有文件", "*.*")],
    )
    root.destroy()
    return file_path if file_path else None


if __name__ == "__main__":
    file_path_x = select_file("选择地震数据文件")
    x = np.load(file_path_x).transpose((1, 2, 0)).astype(np.float32)

    file_path_y = select_file("选择断层数据文件")
    y = np.load(file_path_y).transpose((1, 2, 0)).astype(np.float32)

    nodes = cigvis.create_slices(
        x,
        pos=[[0], [0], [0]],
        cmap="gray",
    )

    body_nodes = cigvis.create_bodys(
        y,
        level=0.5,
        cmap="jet",
    )

    if isinstance(body_nodes, list):
        nodes += body_nodes
    else:
        nodes.append(body_nodes)

    cigvis.plot3D(
        nodes,
        size=(700, 600),
        savename="example.png",
        xyz_axis=True,
        azimuth=90,
        elevation=70,
        distance=500,
    )
