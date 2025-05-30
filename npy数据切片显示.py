import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def select_file():
    # 创建Tkinter根窗口并隐藏
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 弹出文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择文件",  # 对话框标题
        filetypes=[("所有文件", "*.*")]  # 可选文件类型过滤
    )

    # 关闭Tkinter窗口
    root.destroy()

    return file_path if file_path else None  # 返回路径，取消选择则返回None
if __name__ == '__main__':
    #读取文件
    filename = os.path.splitext(select_file())[0]

    data = np.load(filename+".npy")
    print(data.shape) #[Time, x, y ]

    #读取Times
    # 预计算固定尺寸（假设所有切片尺寸相同）
    fig_width = data.shape[2] / 20 + 2      # +2 的目的是显示times刻度
    fig_height = data.shape[1] / 20

    if os.path.exists(filename+'/') :
        print("dir exists")
    else:
        os.makedirs(filename + '/')

    __step = 50
    for i in range(0, data.shape[0], __step):
        # 创建新Figure并显式关闭
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.imshow(data[i, :, :], vmin=-0.01, vmax=0.01, aspect=1)

        plt.savefig(f'{filename}/T-{i}.png')
        plt.close(fig)  # 明确关闭Figure释放内存
    print('#')
    fig_width = data.shape[2] / 20 + 2      # +2 的目的是显示times刻度
    fig_height = data.shape[0] / 20
    ###########   Y
    for i in range(0, data.shape[1], __step):
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.imshow(data[:,i,:], vmin=-0.01, vmax=0.01, aspect=1)
        plt.savefig(f'{filename}/X-{i}.png')
        plt.close(fig)  # 明确关闭Figure释放内存
    print('#')
    fig_width = data.shape[1] / 20 + 2      # +2 的目的是显示times刻度
    fig_height = data.shape[0] / 20 + 2
    ###########   Y
    for i in range(0, data.shape[2], __step):
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.imshow(data[:,:,i], vmin=-0.01, vmax=0.01, aspect=1)
        plt.savefig(f'{filename}/Y-{i}.png')
        plt.close(fig)  # 明确关闭Figure释放内存
    print('#')


