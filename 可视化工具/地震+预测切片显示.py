import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import time
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb


# ========== 显示与导出参数 ==========
# gamma 越小越亮、越大越暗；论文风格建议范围 0.85~1.05
SEISMIC_GAMMA = 1.0

# 地震灰度的输出范围：范围越窄，背景越平淡、越不抢眼
# 当前限制在中灰区域，以突出暗红色断层；需要更淡可继续增大 MIN 或减小 MAX
SEISMIC_GRAY_MIN = 0.32
SEISMIC_GRAY_MAX = 0.68

# 振幅显示分位范围。范围越宽，被压成最亮/最暗颜色的数据越少
SEISMIC_PERCENTILE_LOW = 1.0
SEISMIC_PERCENTILE_HIGH = 99.0

# 地震背景淡出后混合到的底色；数值越小越暗（白色为 '#FFFFFF'）
SEISMIC_BACKGROUND_COLOR = '#8C8C8C'

# 断层预测的颜色和最大不透明度，可填写任意十六进制颜色
FAULT_COLOR = '#8B0000'  # 暗红色；例如更深可用 '#650000'
FAULT_OPACITY = 1.0

# 每个原始采样点在导出图片中占用的像素数；越大图片像素越高、文件也越大
PIXELS_PER_SAMPLE = 4

# 导出 DPI。它与下面的画布尺寸配合，保证图片具有较高的像素分辨率
OUTPUT_DPI = 300

# Lanczos 在放大时比 bicubic 更锐利；预测层保持 nearest，避免红色区域被模糊扩张
SEISMIC_INTERPOLATION = 'nearest'
PREDICTION_INTERPOLATION = 'nearest'


def select_file(title="选择文件"):
    """弹出窗口返回选择的文件绝对路径"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 弹出文件选择对话框
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("NumPy文件", "*.npy"), ("所有文件", "*.*")]
    )
    
    # 关闭Tkinter窗口
    root.destroy()
    
    return file_path if file_path else None


def get_step_value():
    """弹出对话框获取切片间隔"""
    root = tk.Tk()
    root.withdraw()
    
    step = simpledialog.askinteger(
        "设置切片间隔",
        "请输入切片间隔（像素）：",
        initialvalue=16,
        minvalue=1,
        maxvalue=100
    )
    
    root.destroy()
    return step if step else 10


def create_overlay_colormap():
    """按照全局参数创建预测数据的透明颜色映射。"""
    fault_rgb = to_rgb(FAULT_COLOR)
    colors = [(*fault_rgb, 0.0), (*fault_rgb, FAULT_OPACITY)]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('fault_overlay', colors, N=n_bins)
    return cmap


def create_seismic_colormap():
    """创建限制最高亮度的论文风格灰度颜色映射。"""
    gray_min = np.clip(SEISMIC_GRAY_MIN, 0.0, 1.0)
    gray_max = np.clip(SEISMIC_GRAY_MAX, gray_min, 1.0)
    colors = [(gray_min, gray_min, gray_min),
              (gray_max, gray_max, gray_max)]
    cmap = LinearSegmentedColormap.from_list('paper_gray', colors, N=256)
    cmap.set_bad(SEISMIC_BACKGROUND_COLOR)
    return cmap


def plot_overlay_slice(seismic_slice, prediction_slice, vmin_seismic, vmax_seismic):
    """
    绘制叠加的切片图像
    
    参数:
        seismic_slice: 地震数据切片
        prediction_slice: 预测数据切片
        vmin_seismic, vmax_seismic: 地震数据的显示范围
    """
    # PowerNorm 提亮暗部，同时保留地震同相轴的黑白对比
    seismic_norm = PowerNorm(
        gamma=SEISMIC_GAMMA,
        vmin=vmin_seismic,
        vmax=vmax_seismic,
        clip=True
    )
    ax = plt.gca()
    ax.set_facecolor(SEISMIC_BACKGROUND_COLOR)
    plt.imshow(
        seismic_slice,
        cmap=create_seismic_colormap(),
        norm=seismic_norm,
        aspect=1,
        interpolation=SEISMIC_INTERPOLATION
    )
    
    # 叠加红色半透明的预测数据
    red_cmap = create_overlay_colormap()
    plt.imshow(
        prediction_slice,
        cmap=red_cmap,
        vmin=0.0,
        vmax=1.0,
        aspect=1,
        interpolation=PREDICTION_INTERPOLATION
    )


def get_figure_size(slice_shape):
    """根据切片形状计算高分辨率画布尺寸（单位：英寸）。"""
    rows, cols = slice_shape
    title_margin_inches = 0.45
    return (
        cols * PIXELS_PER_SAMPLE / OUTPUT_DPI,
        rows * PIXELS_PER_SAMPLE / OUTPUT_DPI + title_margin_inches
    )


if __name__ == '__main__':
    print("=" * 60)
    print("地震数据 + 预测结果 切片显示工具")
    print("=" * 60)
    
    # ========== 1. 选择地震数据文件 ==========
    print("\n[1/4] 请选择地震数据文件（.npy）...")
    seismic_file = select_file("选择地震数据文件")
    if not seismic_file:
        print("未选择地震数据文件，程序退出。")
        exit()
    print(f"✓ 已选择: {seismic_file}")
    
    # ========== 2. 选择预测结果文件 ==========
    print("\n[2/4] 请选择预测结果文件（.npy）...")
    prediction_file = select_file("选择预测结果文件")
    if not prediction_file:
        print("未选择预测结果文件，程序退出。")
        exit()
    print(f"✓ 已选择: {prediction_file}")
    
    # ========== 3. 设置切片间隔 ==========
    print("\n[3/4] 请设置切片间隔...")
    step = get_step_value()
    print(f"✓ 切片间隔设置为: {step} 像素")
    
    # ========== 4. 加载数据 ==========
    print("\n[4/4] 加载数据中...")
    seismic_data = np.load(seismic_file)
    prediction_data = np.load(prediction_file)
    
    print(f"✓ 地震数据形状: {seismic_data.shape}")
    print(f"✓ 预测数据形状: {prediction_data.shape}")
    print(f"✓ 地震数据范围: [{seismic_data.min():.4f}, {seismic_data.max():.4f}]")
    print(f"✓ 预测数据范围: [{prediction_data.min():.4f}, {prediction_data.max():.4f}]")
    
    # 检查形状是否匹配
    if seismic_data.shape != prediction_data.shape:
        print(f"\n❌ 错误: 两个数据的形状不匹配！")
        print(f"   地震数据: {seismic_data.shape}")
        print(f"   预测数据: {prediction_data.shape}")
        exit()
    
    # ========== 5. 计算地震数据显示范围（使用百分位数增强对比度）==========
    percentile_low = np.percentile(seismic_data, SEISMIC_PERCENTILE_LOW)
    percentile_high = np.percentile(seismic_data, SEISMIC_PERCENTILE_HIGH)
    print(
        f"✓ 地震数据显示范围（{SEISMIC_PERCENTILE_LOW:g}%-"
        f"{SEISMIC_PERCENTILE_HIGH:g}%分位）: "
        f"[{percentile_low:.4f}, {percentile_high:.4f}]"
    )
    
    # ========== 6. 创建输出文件夹 ==========
    # 使用地震文件名作为基础
    base_name = os.path.splitext(os.path.basename(seismic_file))[0]
    output_dir = os.path.join(os.path.dirname(seismic_file), f"{base_name}_overlay_slices")
    
    if os.path.exists(output_dir):
        print(f"✓ 输出目录已存在: {output_dir}")
    else:
        os.makedirs(output_dir)
        print(f"✓ 创建输出目录: {output_dir}")
    
    print("\n" + "=" * 60)
    print("开始生成切片...")
    print("=" * 60)
    print(f"显示亮度 gamma: {SEISMIC_GAMMA}（数值越小越亮）")
    print(f"地震灰度范围: {SEISMIC_GRAY_MIN}~{SEISMIC_GRAY_MAX}（范围越窄，背景越平淡）")
    print(f"断层颜色: {FAULT_COLOR}，不透明度: {FAULT_OPACITY}")
    print(f"输出分辨率: 每个采样点 {PIXELS_PER_SAMPLE}×{PIXELS_PER_SAMPLE} 像素，{OUTPUT_DPI} DPI")
    
    # ========== 7. 第一维度切片 (T/X方向) ==========
    print(f"\n正在处理第一维度切片 (共 {seismic_data.shape[0]} 个)...")
    with tqdm(total=seismic_data.shape[0], desc='维度-0 (T/X)') as pbar:
        for i in range(0, seismic_data.shape[0], step):
            slice_shape = seismic_data[i, :, :].shape
            fig = plt.figure(
                figsize=get_figure_size(slice_shape),
                dpi=OUTPUT_DPI,
                facecolor=SEISMIC_BACKGROUND_COLOR
            )
            plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)
            
            # 绘制叠加图像
            plot_overlay_slice(
                seismic_data[i, :, :],
                prediction_data[i, :, :],
                percentile_low,
                percentile_high
            )
            
            plt.title(f'Dimension-0, Index={i}', fontsize=10, pad=5)
            plt.axis('off')
            
            plt.savefig(f'{output_dir}/Dim0_T{i:04d}.png', dpi=OUTPUT_DPI,
                        bbox_inches='tight', facecolor=SEISMIC_BACKGROUND_COLOR)
            plt.close(fig)
            pbar.update(step)
    
    # ========== 8. 第二维度切片 (X/Y方向) ==========
    print(f"\n正在处理第二维度切片 (共 {seismic_data.shape[1]} 个)...")
    with tqdm(total=seismic_data.shape[1], desc='维度-1 (X/Y)') as pbar:
        for i in range(0, seismic_data.shape[1], step):
            slice_shape = seismic_data[:, i, :].shape
            fig = plt.figure(
                figsize=get_figure_size(slice_shape),
                dpi=OUTPUT_DPI,
                facecolor=SEISMIC_BACKGROUND_COLOR
            )
            plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)
            
            # 绘制叠加图像
            plot_overlay_slice(
                seismic_data[:, i, :],
                prediction_data[:, i, :],
                percentile_low,
                percentile_high
            )
            
            plt.title(f'Dimension-1, Index={i}', fontsize=10, pad=5)
            plt.axis('off')
            
            plt.savefig(f'{output_dir}/Dim1_X{i:04d}.png', dpi=OUTPUT_DPI,
                        bbox_inches='tight', facecolor=SEISMIC_BACKGROUND_COLOR)
            plt.close(fig)
            pbar.update(step)
    
    # ========== 9. 第三维度切片 (Y/Z方向) ==========
    print(f"\n正在处理第三维度切片 (共 {seismic_data.shape[2]} 个)...")
    with tqdm(total=seismic_data.shape[2], desc='维度-2 (Y/Z)') as pbar:
        for i in range(0, seismic_data.shape[2], step):
            slice_shape = seismic_data[:, :, i].shape
            fig = plt.figure(
                figsize=get_figure_size(slice_shape),
                dpi=OUTPUT_DPI,
                facecolor=SEISMIC_BACKGROUND_COLOR
            )
            plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)
            
            # 绘制叠加图像
            plot_overlay_slice(
                seismic_data[:, :, i],
                prediction_data[:, :, i],
                percentile_low,
                percentile_high
            )
            
            plt.title(f'Dimension-2, Index={i}', fontsize=10, pad=5)
            plt.axis('off')
            
            plt.savefig(f'{output_dir}/Dim2_Y{i:04d}.png', dpi=OUTPUT_DPI,
                        bbox_inches='tight', facecolor=SEISMIC_BACKGROUND_COLOR)
            plt.close(fig)
            pbar.update(step)
    
    # ========== 10. 生成图例说明 ==========
    print("\n生成颜色图例...")
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4),
        facecolor=SEISMIC_BACKGROUND_COLOR
    )
    
    # 地震数据示例
    seismic_norm = PowerNorm(
        gamma=SEISMIC_GAMMA,
        vmin=percentile_low,
        vmax=percentile_high,
        clip=True
    )
    ax1.set_facecolor(SEISMIC_BACKGROUND_COLOR)
    ax1.imshow(seismic_data[seismic_data.shape[0]//2, :, :],
               cmap=create_seismic_colormap(), norm=seismic_norm,
               interpolation=SEISMIC_INTERPOLATION)
    ax1.set_title('地震数据（灰度）', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 叠加效果示例
    red_cmap = create_overlay_colormap()
    ax2.set_facecolor(SEISMIC_BACKGROUND_COLOR)
    ax2.imshow(seismic_data[seismic_data.shape[0]//2, :, :],
               cmap=create_seismic_colormap(), norm=seismic_norm,
               interpolation=SEISMIC_INTERPOLATION)
    ax2.imshow(prediction_data[prediction_data.shape[0]//2, :, :],
               cmap=red_cmap, vmin=0.0, vmax=1.0,
               interpolation=PREDICTION_INTERPOLATION)
    ax2.set_title('地震数据 + 断层预测（红色叠加）', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/_说明_颜色图例.png', dpi=OUTPUT_DPI,
                bbox_inches='tight', facecolor=SEISMIC_BACKGROUND_COLOR)
    plt.close(fig)
    
    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("✓ 所有切片生成完成！")
    print("=" * 60)
    total_images = (seismic_data.shape[0] // step + 1) + \
                   (seismic_data.shape[1] // step + 1) + \
                   (seismic_data.shape[2] // step + 1)
    print(f"\n统计信息:")
    print(f"  - 总共生成图片: ~{total_images} 张")
    print(f"  - 切片间隔: {step} 像素")
    print(f"  - 输出目录: {output_dir}")
    print(f"\n说明:")
    print(f"  - 灰色: 地震数据")
    print(f"  - 红色: 断层预测（越红表示断层概率越高）")
    print(f"  - 查看 '_说明_颜色图例.png' 了解颜色含义")
    print("\n" + "=" * 60)

