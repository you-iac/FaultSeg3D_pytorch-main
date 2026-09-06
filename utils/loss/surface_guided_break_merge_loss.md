# SurfaceGuidedBreakMergeLoss 使用说明

新损失位于 `utils/loss/surface_guided_break_merge_loss.py`。
原有 `utils/loss/connectivity_loss.py` 与 `ConnectivityLoss` 分支保持原样。
不需要更改模型、数据标签或推理代码。

## 最简单的调用

在原训练命令中替换损失选项即可：

```powershell
python main_.py --mode train --exp surface_guided_test --loss_func SurfaceGuidedBreakMergeLoss --epochs 50
```

`main.py` 也已注册这个选项。训练集路径等其他参数沿用原命令。
当前 CEDNet 返回两通道 Softmax 概率，因此训练接入默认
`--surface_input_type probabilities`。如果更换成输出原始 logits 的模型，
新损失应使用 `--surface_input_type logits`；这个选项只控制新辅助损失，
现有 Dice/CE 实现没有改动，其输入约定仍需与所用骨干匹配。

直接在其他训练代码调用：

```python
from utils.loss.surface_guided_break_merge_loss import SurfaceGuidedBreakMergeLoss

criterion = SurfaceGuidedBreakMergeLoss(input_is_probability=True).to(device)
loss_surface = criterion(outputs, labels)
loss = loss_dice + loss_ce + 0.1 * loss_surface
loss.backward()
```

类的默认 `input_is_probability=False` 用于原始 logits；不要凭数值范围自动猜测。
`criterion(pred_prob=outputs, target=labels)` 显式指定概率，
`criterion(logits=outputs, target=labels)` 显式指定 logits。
支持单/双通道二分类预测，标签支持 `[B,D,H,W]`、`[B,1,D,H,W]`
及双通道 one-hot；空间尺寸和 batch 必须匹配。

## 计算过程

1. 对每个 GT 前景体素，取 `5×5×5` 窗口内的 GT 前景坐标，计算局部
   协方差。最小特征值对应的特征向量是局部平面法向 `n`。
   这是标准的局部 PCA 法向估计思路，参见
   [PCL 法向估计文档](https://pointclouds.org/documentation/tutorials/normal_estimation.html)。
2. 设三个升序特征值为 `λ0 ≤ λ1 ≤ λ2`，可靠性
   `c = (λ1 - λ0) / max(λ2, eps)`，限制在 `[0,1]`。
   不足 6 个 GT 支撑点时可靠性为 0。孤立点、线状或近似各向同性区域
   不应强行指定曲面方向。交叉面不一定都能被该可靠性判据识别。
3. 仅在 GT 前景体素存储 `Q = c * n nᵀ`，仅需 6 个对称矩阵元素。
   这样 `n` 与 `-n` 给出同一结果。背景的法向、可靠性和 `Q` 全部为 0，
   不传播法向，也不计算距离衰减。GT 本身的法向可靠性不足时保持基础监督。
4. 仅由预测决定检查区域：

   ```text
   region = dilate(p.detach() > pred_threshold, mask_kernel_size)
   pair_mask(i,j) = region_i * region_j
   ```

   默认 `pred_threshold=0.3`，膨胀核为 `3×3×3`。只有两个端点都在
   预测膨胀区域内、且未越过体积边界的邻接对参与损失。
   正邻接对和负邻接对使用同一个区域，不添加 GT 区域或其他 GT 选区兜底。
   GT 只提供曲面方向，并以 `y_i*y_j` 区分断裂项和误连接项。
5. 默认使用 26 邻域，每条无向边只计算一次（13 个偏移）。对于相邻
   位置 `i,j`，取预测乘积 `a=p_i*p_j`，标签乘积 `t=y_i*y_j`。
   用物理空间边方向的单位向量 `u` 计算：

   ```text
   q = (uᵀ Q_i u + uᵀ Q_j u) / 2
   c_pair = (trace(Q_i) + trace(Q_j)) / 2
   w_break = 1 + geometry_strength * (c_pair - q)
   w_merge = 1 + geometry_strength * q
   ```

   真实前景邻接对使用 `w_break * (1-a)`；其他选中的邻接对使用
   `w_merge * a`。切向真实连接得到更强补断监督，法向错误连接得到
   更强抑制。两者均保留基础权重 1，不会因法向不可靠而失去监督。
   两端都是 GT 前景时，取两端几何信息的平均；只有一端为 GT 前景时，
   只有该端贡献几何信息，仍按两端平均，因此贡献为该端的一半。
   两端都是背景时 `Q_i=Q_j=0`，两个权重均为 1，使用普通邻接惩罚。

6. 正邻接对与负邻接对分别除以其**未加几何权重的数量**，再按方向取平均。
   不用各方向的几何权重和作分母，以免平面上恒定方向权重被抵消。
   最终 `L_surface = L_break + 0.2 * L_merge`。

`estimate_geometry(target)` 返回 `normal`、`normal_tensor` 和 `confidence`，
不返回距离。三者在 GT 背景体素处全部为 0。
关闭几何权重时仍使用相同的预测选边区域。

## 反向传播与训练权重

法向、可靠性和区域掩膜都不参与反向传播，仅作为固定监督权重。
预测概率始终保持连续：

```text
正边：d[w_break*(1-p_i*p_j)]/dp_i = -w_break*p_j
负边：d[w_merge*p_i*p_j]/dp_i =  w_merge*p_j
```

梯度经过 Sigmoid/Softmax（若输入 logits）、网络输出回到模型参数。
GT 不作为网络输入；推理时不需要 GT，也不运行此损失。
几何与损失在 autocast 外计算，低精度输入转 FP32，梯度仍传回原输入。

阈值只负责选区，邻接乘积使用原始连续概率。预测区域为空时，损失返回
保留计算图的零值，可以正常反向传播但梯度为零。若真实断层完全漏检，且位于
预测膨胀区域外，此辅助损失不会对该处提供梯度；该处的学习依赖 Dice/CE。

项目接入使用：`Dice + WeightedCE + alpha * L_surface`。
默认前 10 轮 `alpha=0`，第 11–20 轮线性升至 0.1，之后为 0.1。
预热阶段跳过新损失的几何计算；没有 epoch 上下文时使用完整权重。

可调整的命令行参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--surface_weight` | 0.1 | 预热后的辅助损失权重 |
| `--surface_geometry_strength` | 1.0 | 几何加权强度；0 用于单独消融几何指导 |
| `--surface_input_type` | probabilities | 新损失接收 probabilities 或 logits |

PCA 窗口、邻域、`pred_threshold`、`mask_kernel_size` 和
`voxel_spacing=(D,H,W)` 等可在类构造时设置。spacing 用于局部几何估计与
邻接方向，默认单位为体素；预测膨胀核的大小按体素指定。
没有法向传播半径或距离衰减参数。
默认只使用原分辨率，避免预测平均池化/标签最大池化引入额外增厚目标。

## 验证与边界

```powershell
python -m unittest discover -s test -p test_surface_guided_break_merge_loss.py -v
```

测试涵盖平面/倾斜面与不同 spacing、GT 背景零几何场、仅由预测选择邻接对、退化标签、
预测完美时零损失、补断/假桥梯度方向、数值梯度校验、输入形式、
体积边界、预热及组合损失，以及 CUDA 可用时的 FP16/autocast。

这是预测区域内、由 GT 曲面几何指导的局部邻接约束，没有计算全局连通域或断层实例归属，
不能保证所有断裂被修复或所有错误合并被排除。仍需用真实数据评估收益。
`geometry_strength=0` 回退到这个新类的普通邻接项，不等同于旧损失的
默认设置（旧类还包含多尺度和不同方向权重）。

`main_.py` 在本仓库被 `.gitignore` 忽略；工作区中的训练选项已更新，
迁移代码时请同时携带该文件的修改，或使用已注册选项的 `main.py`。
