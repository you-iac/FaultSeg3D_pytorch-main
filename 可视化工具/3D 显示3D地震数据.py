import cigvis
from cigvis import colormap
import numpy as np
import tkinter as tk
from tkinter import filedialog


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
    x = np.load(file_path_x).transpose((1, 2, 0))

    import matplotlib as mpl

    mpl.rcParams["axes.labelsize"] = 20
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10

    nodes = cigvis.create_slices(
        x,
        pos=[[55], [85], [340]],
        cmap="Petrel",
    )
    nodes += cigvis.create_colorbar_from_nodes(
        nodes,
        "Amplitude",
        select="slices",
    )

    cigvis.plot3D(
        nodes,
        size=(700, 600),
        savename="example.png",
        xyz_axis=True,
        azimuth=80,
        elevation=80,
        distance=500,
    )
