import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import time
from tqdm import tqdm


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
    return step if step else 16


def create_overlay_colormap():
    """创建预测数据的透明红色颜色映射"""
    from matplotlib.colors import LinearSegmentedColormap
    
    # 从透明到红色的渐变
    colors = [(1, 0, 0, 0),      # 完全透明
              (1, 0, 0, 0.8)]    # 红色，80%不透明
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('transparent_red', colors, N=n_bins)
    return cmap


def plot_overlay_slice(seismic_slice, prediction_slice, vmin_seismic, vmax_seismic):
    """
    绘制叠加的切片图像
    
    参数:
        seismic_slice: 地震数据切片
        prediction_slice: 预测数据切片
        vmin_seismic, vmax_seismic: 地震数据的显示范围
    """
    # 使用灰度显示地震数据
    plt.imshow(seismic_slice, cmap='gray', vmin=vmin_seismic, vmax=vmax_seismic, aspect=1)
    
    # 叠加红色半透明的预测数据
    red_cmap = create_overlay_colormap()
    plt.imshow(prediction_slice, cmap=red_cmap, vmin=0.0, vmax=1.0, aspect=1, alpha=0.9)


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
    percentile_low = np.percentile(seismic_data, 2)
    percentile_high = np.percentile(seismic_data, 98)
    print(f"✓ 地震数据显示范围（2%-98%分位）: [{percentile_low:.4f}, {percentile_high:.4f}]")
    
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
    
    # ========== 7. 第一维度切片 (T/X方向) ==========
    print(f"\n正在处理第一维度切片 (共 {seismic_data.shape[0]} 个)...")
    fig_width = seismic_data.shape[2] / 20 + 2
    fig_height = seismic_data.shape[1] / 20
    
    with tqdm(total=seismic_data.shape[0], desc='维度-0 (T/X)') as pbar:
        for i in range(0, seismic_data.shape[0], step):
            fig = plt.figure(figsize=(fig_width, fig_height))
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            
            # 绘制叠加图像
            plot_overlay_slice(
                seismic_data[i, :, :],
                prediction_data[i, :, :],
                percentile_low,
                percentile_high
            )
            
            plt.title(f'Dimension-0, Index={i}', fontsize=10, pad=5)
            plt.axis('off')
            
            plt.savefig(f'{output_dir}/Dim0_T{i:04d}.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            pbar.update(step)
    
    # ========== 8. 第二维度切片 (X/Y方向) ==========
    print(f"\n正在处理第二维度切片 (共 {seismic_data.shape[1]} 个)...")
    fig_width = seismic_data.shape[2] / 20 + 2
    fig_height = seismic_data.shape[0] / 20
    
    with tqdm(total=seismic_data.shape[1], desc='维度-1 (X/Y)') as pbar:
        for i in range(0, seismic_data.shape[1], step):
            fig = plt.figure(figsize=(fig_width, fig_height))
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            
            # 绘制叠加图像
            plot_overlay_slice(
                seismic_data[:, i, :],
                prediction_data[:, i, :],
                percentile_low,
                percentile_high
            )
            
            plt.title(f'Dimension-1, Index={i}', fontsize=10, pad=5)
            plt.axis('off')
            
            plt.savefig(f'{output_dir}/Dim1_X{i:04d}.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            pbar.update(step)
    
    # ========== 9. 第三维度切片 (Y/Z方向) ==========
    print(f"\n正在处理第三维度切片 (共 {seismic_data.shape[2]} 个)...")
    fig_width = seismic_data.shape[1] / 20 + 2
    fig_height = seismic_data.shape[0] / 20 + 2
    
    with tqdm(total=seismic_data.shape[2], desc='维度-2 (Y/Z)') as pbar:
        for i in range(0, seismic_data.shape[2], step):
            fig = plt.figure(figsize=(fig_width, fig_height))
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            
            # 绘制叠加图像
            plot_overlay_slice(
                seismic_data[:, :, i],
                prediction_data[:, :, i],
                percentile_low,
                percentile_high
            )
            
            plt.title(f'Dimension-2, Index={i}', fontsize=10, pad=5)
            plt.axis('off')
            
            plt.savefig(f'{output_dir}/Dim2_Y{i:04d}.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            pbar.update(step)
    
    # ========== 10. 生成图例说明 ==========
    print("\n生成颜色图例...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 地震数据示例
    ax1.imshow(seismic_data[seismic_data.shape[0]//2, :, :], 
               cmap='gray', vmin=percentile_low, vmax=percentile_high)
    ax1.set_title('地震数据（灰度）', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 叠加效果示例
    red_cmap = create_overlay_colormap()
    ax2.imshow(seismic_data[seismic_data.shape[0]//2, :, :], 
               cmap='gray', vmin=percentile_low, vmax=percentile_high)
    ax2.imshow(prediction_data[prediction_data.shape[0]//2, :, :], 
               cmap=red_cmap, vmin=0.0, vmax=1.0, alpha=0.9)
    ax2.set_title('地震数据 + 断层预测（红色叠加）', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/_说明_颜色图例.png', dpi=150, bbox_inches='tight')
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

