
import segyio
import os
import sys
import time
from obspy.io.segy.segy import _read_segy, SEGYBinaryFileHeader
from obspy import read
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
def select_file():
    # 创建 Tkinter 根窗口并隐藏
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 弹出文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择文件",
        filetypes=[("所有文件", "*.*"), ("文本文件", "*.txt"), ("Python文件", "*.py")]
    )

    # 销毁根窗口
    root.destroy()

    return file_path if file_path else None

if __name__ == "__main__":

    filename = "D:/Ccc/PCB10011008700.npy"
    data = np.load(filename)
    data.shape

    # 预计算固定尺寸（假设所有切片尺寸相同）
    fig_width = data.shape[1] / 20 + 2      # +2 的目的是显示times刻度
    fig_height = data.shape[0] / 20

    for i in range(data.shape[1]):
        # 创建新Figure并显式关闭
        if i % 50 == 0 :
            fig = plt.figure(figsize=(fig_width, fig_height))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.imshow(data[:, :, i], vmin=-0.01, vmax=0.01, aspect=1)
            plt.savefig(f'D:/Ccc/T-{i}.png')
            plt.close(fig)  # 明确关闭Figure释放内存

    fig_width = data.shape[1] / 20 + 2      # +2 的目的是显示times刻度
    fig_height = data.shape[0] / 20
    ###########   Y
    for i in range(0, data.shape[2]):
        if i % 10 == 0:
            fig = plt.figure(figsize=(fig_width, fig_height))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.imshow(data[i,:,:], vmin=-0.01, vmax=0.01, aspect=1)
            plt.savefig(f'D:/data/{filename}/Y-{i}.png')
            plt.close(fig)  # 明确关闭Figure释放内存
