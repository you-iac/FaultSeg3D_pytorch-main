# 地震数据 + 断层预测 3D可视化工具
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 在导入seismic_canvas之前修复matplotlib样式问题
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

# 保存原始的use函数
_original_use = mplstyle.use

# 创建一个包装函数来处理不存在的样式
def _safe_style_use(style):
    if isinstance(style, str) and 'seaborn' in style:
        # 尝试使用默认样式而不是seaborn
        try:
            _original_use('default')
        except:
            pass
    else:
        try:
            _original_use(style)
        except OSError:
            _original_use('default')

# 临时替换style.use函数
mplstyle.use = _safe_style_use
plt.style.use = _safe_style_use

# 现在可以安全地导入seismic_canvas
from seismic_canvas import SeismicCanvas, volume_slices, XYZAxis, Colorbar
from vispy.color import Colormap, get_colormap

# 恢复原始函数
mplstyle.use = _original_use
plt.style.use = _original_use


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


# ============ 1. 选择并加载数据 ============
print("=" * 70)
print("地震数据 + 断层预测 3D可视化工具")
print("=" * 70)

print("\n[步骤 1/2] 请选择地震数据文件（.npy）...")
seismic_file = select_file("选择地震数据文件")
if not seismic_file:
    print("❌ 未选择地震数据文件，程序退出。")
    exit()
print(f"✓ 已选择: {seismic_file}")

print("\n[步骤 2/2] 请选择预测结果文件（.npy）...")
prediction_file = select_file("选择预测结果文件")
if not prediction_file:
    print("❌ 未选择预测结果文件，程序退出。")
    exit()
print(f"✓ 已选择: {prediction_file}")

print("\n正在加载数据...")
seismic_data = np.load(seismic_file)
prediction = np.load(prediction_file)

print(f"✓ 地震数据形状: {seismic_data.shape}")
print(f"✓ 预测数据形状: {prediction.shape}")
print(f"✓ 地震数据范围: [{seismic_data.min():.3f}, {seismic_data.max():.3f}]")
print(f"✓ 预测数据范围: [{prediction.min():.3f}, {prediction.max():.3f}]")

# 检查形状是否匹配
if seismic_data.shape != prediction.shape:
    print(f"\n❌ 错误: 两个数据的形状不匹配！")
    print(f"   地震数据: {seismic_data.shape}")
    print(f"   预测数据: {prediction.shape}")
    exit()

# ============ 2. 数据预处理（可选）============
# 如果预测值范围不合适，可以归一化
# prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())

# 如果需要阈值过滤（只显示高置信度的预测）
# prediction_filtered = prediction.copy()
# prediction_filtered[prediction < 0.5] = 0  # 低于0.5的设为0（透明）

# ============ 3. 设置颜色映射 ============
# 地震数据：使用对比度强的颜色映射
# 可选方案：
#   'seismic' - 蓝白红，对比度强
#   'gray' - 黑白灰度，经典地震数据显示
#   'Greys' - 白到黑的灰度
#   'bone' - 带蓝色调的灰度，对比度好
seismic_cmap = 'gray'  # 推荐使用gray或seismic

# 手动设置范围以增强对比度（根据数据的百分位数，过滤极端值）
print("\n正在计算最佳显示范围...")
percentile_low = np.percentile(seismic_data, 2)  # 第2百分位
percentile_high = np.percentile(seismic_data, 98)  # 第98百分位
seismic_range = (percentile_low, percentile_high)
print(f"✓ 地震数据显示范围: [{percentile_low:.3f}, {percentile_high:.3f}]")
print(f"  (原始范围: [{seismic_data.min():.3f}, {seismic_data.max():.3f}])")

# 预测数据：从透明到红色的渐变（更直观显示断层）
n_colors = 256
rgba = np.zeros((n_colors, 4))
# 创建从透明到纯红色的渐变
# 如果想要更亮的红色，可以把0.5改成0.8或1.0
# 如果想要橙红色，可以给G通道加一点值，比如 np.linspace(0.0, 0.3, n_colors)
# 如果想要黄色，可以设置 R=1.0, G=1.0, B=0.0
rgba[:, 0] = np.linspace(0.5, 1.0, n_colors)  # R通道：深红到亮红
rgba[:, 1] = np.linspace(0.0, 0.0, n_colors)  # G通道（纯红色）
rgba[:, 2] = np.linspace(0.0, 0.0, n_colors)  # B通道（纯红色）
rgba[:, 3] = np.linspace(0.0, 0.95, n_colors) ** 0.5  # Alpha通道：透明到不透明（使用0.5次方使低值更透明）
prediction_cmap = Colormap(rgba)
prediction_range = (0.0, 1.0)  # 根据你的预测值范围调整

# ============ 4. 设置切片位置 ============
# 选择要显示的切片位置
x_slices = [seismic_data.shape[0] // 2]  # x方向中间位置
y_slices = [seismic_data.shape[1] // 2]  # y方向中间位置
z_slices = [seismic_data.shape[2] - 20]  # z方向靠近底部

# 或者显示多个切片
# x_slices = [100, 200, 300]
# y_slices = [150, 250]
# z_slices = [30, 50, 70]

# ============ 5. 创建可视化节点 ============
print("\n" + "=" * 70)
print("正在创建3D可视化...")
print("=" * 70)
visual_nodes = volume_slices(
    volumes=[seismic_data, prediction],  # 两个体数据
    cmaps=[seismic_cmap, prediction_cmap],  # 对应的colormap
    clims=[seismic_range, prediction_range],  # 对应的值范围
    x_pos=x_slices,
    y_pos=y_slices,
    z_pos=z_slices,
    interpolation='bilinear',  # 双线性插值，更平滑
    seismic_coord_system=True  # 使用地震坐标系（z轴向下）
)

# ============ 6. 创建坐标轴和颜色条 ============
xyz_axis = XYZAxis(seismic_coord_system=True)
colorbar = Colorbar(
    cmap=prediction_cmap,
    clim=prediction_range,
    label_str='断层预测概率 (红色=高概率)',
    size=(800, 20)
)

# ============ 7. 创建画布并显示 ============
canvas = SeismicCanvas(
    title='地震数据 + 断层预测',
    visual_nodes=visual_nodes,
    xyz_axis=xyz_axis,
    colorbar=colorbar,
    size=(1400, 1000),
    bgcolor='white',
    axis_scales=(1, 1, 1.5),  # z轴拉伸1.5倍
    fov=30,
    elevation=30,
    azimuth=45,
    zoom_factor=1.5
)

print("\n" + "=" * 70)
print("✓ 3D可视化窗口已打开！")
print("=" * 70)
print("\n交互操作提示:")
print("  🖱️  鼠标左键拖动    : 旋转视角")
print("  🖱️  鼠标右键/滚轮   : 缩放")
print("  ⌨️  Shift + 左键    : 平移场景")
print("  ⌨️  Ctrl + 左键     : 拖动切片")
print("  ⌨️  空格键          : 重置视角")
print("  ⌨️  S键             : 保存截图")
print("  ⌨️  D键             : 切换拖动模式")
print("  ⌨️  Esc键           : 退出程序")
print("\n说明:")
print("  📊 灰色背景: 地震数据")
print("  🔴 红色叠加: 断层预测（越红表示断层概率越高）")
print("=" * 70 + "\n")

from vispy import app
app.run()