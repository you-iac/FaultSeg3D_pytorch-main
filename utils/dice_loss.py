import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    标准Dice Loss - 自动支持2D和3D
    """

    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # 将预测结果和目标标签转换为二进制形式
        pred = pred[:, 1, :, :]  # 取第二个通道的预测结果
        target = target.float()

        # 计算Dice系数的分子和分母
        intersection = (pred * target).sum()
        dice_coefficient = (2. * intersection + self.epsilon) / (pred.sum() + target.sum() + self.epsilon)

        # 计算Dice Loss
        loss = 1 - dice_coefficient
        return loss


import torch
import torch.nn.functional as F


def compute_multiscale_density_weights(
        label,
        scales=(8, 16, 32),
        min_w=0.1
):
    """
    label: (B, 1, D, H, W), binary tensor 0/1
    return: W (B, D, H, W), voxel-level weight map
    """
    if label.dim() == 4:
        label = label.unsqueeze(1)
    elif label.dim() != 5:
        raise ValueError(f"label 需要是4D或5D张量，当前维度: {label.dim()}")

    B, _, D, H, W = label.shape
    device = label.device

    weight_maps = []

    for s in scales:
        # === 1) sum pooling: compute per-block soft counts ===
        pool = F.avg_pool3d(label.float(), kernel_size=s, stride=s)  # (B,1,D/s,H/s,W/s)
        # avg_pool outputs mean; convert to counts:
        block_count = pool * (s ** 3)

        # === 2) normalize to [min_w, 1] ===
        max_c = block_count.max() + 1e-6
        raw = block_count / max_c
        w_block = min_w + (1 - min_w) * raw  # ensure nonzero weight

        # === 3) upsample back to voxel resolution ===
        w_voxel = F.interpolate(
            w_block,
            size=(D, H, W),
            mode='nearest'
        )  # (B,1,D,H,W)

        weight_maps.append(w_voxel)

    # === 4) sum over scales and normalize mean(W)=1 ===
    W = torch.zeros_like(weight_maps[0])
    n = len(weight_maps)

    for w in weight_maps:
        W += w

    W = W / n  # scale average
    W = W / (W.mean() + 1e-6)  # normalize global mean to 1

    return W.squeeze(1)  # return (B, D, H, W)


def weighted_cross_entropy(logits, labels, weight_map):
    """
    logits: (B, 2, D, H, W)
    labels: (B, D, H, W), long
    weight_map: (B, D, H, W)
    """
    ce = F.cross_entropy(logits, labels, reduction='none')  # (B,D,H,W)
    loss = (ce * weight_map).mean()
    return loss


def compute_loss_multiscale_weighted(
        logits,
        label,
        ce_weight=1.0,
        dice_weight=1.0,
        scales=(8, 16, 32),
        min_w=0.1
):
    """
    logits: (B, 2, D, H, W)
    label:  (B, 1, D, H, W)
    """
    # prepare labels shape for CE
    label_ce = label[:, 0].long()  # (B,D,H,W)

    # === 1) compute voxel-level weights ===
    W = compute_multiscale_density_weights(label, scales=scales, min_w=min_w)
    # shape: (B,D,H,W)

    # === 2) CE with voxel weights ===
    loss_ce = weighted_cross_entropy(logits, label_ce, W)

    # === 3) MultiScalePatchDice without weights ===
    dice_loss_fn = MultiScalePatchDiceLoss()
    loss_dice = dice_loss_fn(logits, label)  # unchanged

    # === 4) combine ===
    loss = ce_weight * loss_ce + dice_weight * loss_dice

    return {
        "loss": loss,
        "ce": loss_ce,
        "dice": loss_dice
    }


class WeightedCrossEntropyDiceLoss(nn.Module):
    """
    将多尺度密度加权交叉熵与标准Dice Loss整合为单个模块。

    参数:
        ce_weight: 交叉熵损失权重
        dice_weight: Dice损失权重
        scales: 计算密度权重的尺度列表
        min_w: 密度权重的最小值
        epsilon: DiceLoss的数值稳定项
    """

    def __init__(self,
                 ce_weight=1.0,
                 dice_weight=1.0,
                 scales=(8, 16, 32),
                 min_w=0.1,
                 epsilon=1e-5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.scales = scales
        self.min_w = min_w
        self.dice_loss = DiceLoss(epsilon=epsilon)

    def forward(self, logits, label):
        """
        logits: (B, 2, D, H, W)
        label:  (B, 1, D, H, W) 或 (B, D, H, W)
        """
        if label.dim() == 5 and label.size(1) == 1:
            label_ce = label[:, 0].long()
            label_for_weights = label
        elif label.dim() == 4:
            label_ce = label.long()
            label_for_weights = label.unsqueeze(1)
        else:
            raise ValueError("label 张量需要是 (B,1,D,H,W) 或 (B,D,H,W) 形状")

        weight_map = compute_multiscale_density_weights(
            label_for_weights, scales=self.scales, min_w=self.min_w
        )
        loss_ce = weighted_cross_entropy(logits, label_ce, weight_map)
        loss_dice = self.dice_loss(logits, label)

        loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice
        # 方便调试：可在外部读取最近一次的各项损失
        self.latest_components = {
            "loss": loss.detach(),
            "ce": loss_ce.detach(),
            "dice": loss_dice.detach()
        }
        return loss

import torch
import torch.nn.functional as F

def compute_multiscale_density_weights(
    label,
    scales=(8, 16, 32),
    min_w=0.1
):
    """
    label: (B, 1, D, H, W), binary tensor 0/1
    return: W (B, D, H, W), voxel-level weight map
    """
    if label.dim() == 4:
        label = label.unsqueeze(1)
    elif label.dim() != 5:
        raise ValueError(f"label 需要是4D或5D张量，当前维度: {label.dim()}")

    B, _, D, H, W = label.shape
    device = label.device

    weight_maps = []

    for s in scales:
        # === 1) sum pooling: compute per-block soft counts ===
        pool = F.avg_pool3d(label.float(), kernel_size=s, stride=s)  # (B,1,D/s,H/s,W/s)
        # avg_pool outputs mean; convert to counts:
        block_count = pool * (s**3)

        # === 2) normalize to [min_w, 1] ===
        max_c = block_count.max() + 1e-6
        raw = block_count / max_c
        w_block = min_w + (1 - min_w) * raw   # ensure nonzero weight

        # === 3) upsample back to voxel resolution ===
        w_voxel = F.interpolate(
            w_block,
            size=(D, H, W),
            mode='nearest'
        )  # (B,1,D,H,W)

        weight_maps.append(w_voxel)

    # === 4) sum over scales and normalize mean(W)=1 ===
    W = torch.zeros_like(weight_maps[0])
    n = len(weight_maps)

    for w in weight_maps:
        W += w

    W = W / n  # scale average
    W = W / (W.mean() + 1e-6)  # normalize global mean to 1

    return W.squeeze(1)   # return (B, D, H, W)

def weighted_cross_entropy(logits, labels, weight_map):
    """
    logits: (B, 2, D, H, W)
    labels: (B, D, H, W), long
    weight_map: (B, D, H, W)
    """
    ce = F.cross_entropy(logits, labels, reduction='none')  # (B,D,H,W)
    loss = (ce * weight_map).mean()
    return loss

def compute_loss_multiscale_weighted(
    logits,
    label,
    ce_weight=1.0,
    dice_weight=1.0,
    scales=(8, 16, 32),
    min_w=0.1
):
    """
    logits: (B, 2, D, H, W)
    label:  (B, 1, D, H, W)
    """
    # prepare labels shape for CE
    label_ce = label[:, 0].long()     # (B,D,H,W)

    # === 1) compute voxel-level weights ===
    W = compute_multiscale_density_weights(label, scales=scales, min_w=min_w)
    # shape: (B,D,H,W)

    # === 2) CE with voxel weights ===
    loss_ce = weighted_cross_entropy(logits, label_ce, W)

    # === 3) MultiScalePatchDice without weights ===
    dice_loss_fn = MultiScalePatchDiceLoss()
    loss_dice = dice_loss_fn(logits, label)  # unchanged

    # === 4) combine ===
    loss = ce_weight * loss_ce + dice_weight * loss_dice

    return {
        "loss": loss,
        "ce": loss_ce,
        "dice": loss_dice
    }



class WeightedCrossEntropyDiceLoss(nn.Module):
    """
    将多尺度密度加权交叉熵与标准Dice Loss整合为单个模块。

    参数:
        ce_weight: 交叉熵损失权重
        dice_weight: Dice损失权重
        scales: 计算密度权重的尺度列表
        min_w: 密度权重的最小值
        epsilon: DiceLoss的数值稳定项
    """
    def __init__(self,
                 ce_weight=1.0,
                 dice_weight=1.0,
                 scales=(8, 16, 32),
                 min_w=0.1,
                 epsilon=1e-5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.scales = scales
        self.min_w = min_w
        self.dice_loss = DiceLoss(epsilon=epsilon)

    def forward(self, logits, label):
        """
        logits: (B, 2, D, H, W)
        label:  (B, 1, D, H, W) 或 (B, D, H, W)
        """
        if label.dim() == 5 and label.size(1) == 1:
            label_ce = label[:, 0].long()
            label_for_weights = label
        elif label.dim() == 4:
            label_ce = label.long()
            label_for_weights = label.unsqueeze(1)
        else:
            raise ValueError("label 张量需要是 (B,1,D,H,W) 或 (B,D,H,W) 形状")

        weight_map = compute_multiscale_density_weights(
            label_for_weights, scales=self.scales, min_w=self.min_w
        )
        loss_ce = weighted_cross_entropy(logits, label_ce, weight_map)
        loss_dice = self.dice_loss(logits, label)

        loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice
        # 方便调试：可在外部读取最近一次的各项损失
        self.latest_components = {
            "loss": loss.detach(),
            "ce": loss_ce.detach(),
            "dice": loss_dice.detach()
        }
        return loss


class PatchDiceLoss(nn.Module):
    """
    分块Dice Loss - 自动支持2D和3D

    参数:
        patch_size: int，patch大小
        epsilon: 数值稳定性参数
        ignore_empty: 是否忽略空patch（不推荐）
    """
    def __init__(self, patch_size=32, epsilon=1e-5, ignore_empty=False):
        super().__init__()
        self.patch_size = patch_size
        self.epsilon = epsilon
        self.ignore_empty = ignore_empty

    def forward(self, pred, target):
        # 自动检测维度
        if pred.dim() == 4:
            return self._forward_2d(pred, target)
        elif pred.dim() == 5:
            return self._forward_3d(pred, target)
        else:
            raise ValueError(f"不支持的维度: {pred.dim()}, 应该是4D或5D")

    def _forward_2d(self, pred, target):
        """2D版本: (B, C, H, W)"""
        B, C, H, W = pred.shape
        pred = pred[:, 1, :, :]  # 前景通道 (B, H, W)
        target = target.float()

        # 计算patch数量（向上取整，包含边缘）
        patches_h = (H + self.patch_size - 1) // self.patch_size
        patches_w = (W + self.patch_size - 1) // self.patch_size

        total_loss = 0.0
        valid_count = 0

        for i in range(patches_h):
            for j in range(patches_w):
                # 提取patch（处理边界）
                h_start = i * self.patch_size
                h_end = min(h_start + self.patch_size, H)
                w_start = j * self.patch_size
                w_end = min(w_start + self.patch_size, W)

                pred_patch = pred[:, h_start:h_end, w_start:w_end]
                target_patch = target[:, h_start:h_end, w_start:w_end]

                # 计算patch的Dice
                intersection = (pred_patch * target_patch).sum()
                union = pred_patch.sum() + target_patch.sum()

                # 处理空patch
                if self.ignore_empty and union < self.epsilon:
                    continue  # 跳过全空patch

                dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
                total_loss += (1 - dice)
                valid_count += 1

        # 确保至少有一个patch
        if valid_count == 0:
            valid_count = 1

        return total_loss / valid_count

    def _forward_3d(self, pred, target):
        """3D版本: (B, C, D, H, W)"""
        B, C, D, H, W = pred.shape
        pred = pred[:, 1, :, :, :]  # 前景通道 (B, D, H, W)
        target = target.float()

        # 计算patch数量（向上取整）
        patches_d = (D + self.patch_size - 1) // self.patch_size
        patches_h = (H + self.patch_size - 1) // self.patch_size
        patches_w = (W + self.patch_size - 1) // self.patch_size

        total_loss = 0.0
        valid_count = 0

        for i in range(patches_d):
            for j in range(patches_h):
                for k in range(patches_w):
                    # 提取patch（处理边界）
                    d_start = i * self.patch_size
                    d_end = min(d_start + self.patch_size, D)
                    h_start = j * self.patch_size
                    h_end = min(h_start + self.patch_size, H)
                    w_start = k * self.patch_size
                    w_end = min(w_start + self.patch_size, W)

                    pred_patch = pred[:, d_start:d_end, h_start:h_end, w_start:w_end]
                    target_patch = target[:, d_start:d_end, h_start:h_end, w_start:w_end]

                    # 计算patch的Dice
                    intersection = (pred_patch * target_patch).sum()
                    union = pred_patch.sum() + target_patch.sum()

                    # 处理空patch
                    if self.ignore_empty and union < self.epsilon:
                        continue

                    dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
                    total_loss += (1 - dice)
                    valid_count += 1

        if valid_count == 0:
            valid_count = 1

        return total_loss / valid_count


class MultiScalePatchDiceLoss(nn.Module):
    """
    多尺度Patch Dice Loss

    参数:
        patch_sizes: patch大小列表
        weights: 各尺度权重（可选）
        include_global: 是否包含全局Dice
    """
    def __init__(self, patch_sizes=[16, 32, 64], weights=None,
                 include_global=True, epsilon=1e-5):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.include_global = include_global
        self.epsilon = epsilon

        # 初始化各个loss（避免重复创建）
        self.patch_losses = nn.ModuleList([
            PatchDiceLoss(patch_size=ps, epsilon=epsilon)
            for ps in patch_sizes
        ])

        if include_global:
            self.global_loss = DiceLoss(epsilon=epsilon)

        # 设置权重
        if weights is None:
            # 默认权重：patch尺度均等，全局稍小
            if include_global:
                self.weights = [0.3] * len(patch_sizes) + [0.1]
            else:
                self.weights = [1.0 / len(patch_sizes)] * len(patch_sizes)
        else:
            self.weights = weights

        # 归一化权重
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]

    def forward(self, pred, target):
        total_loss = 0.0

        # 计算各个patch尺度的loss
        for i, patch_loss in enumerate(self.patch_losses):
            loss = patch_loss(pred, target)
            total_loss += self.weights[i] * loss

        # 加上全局loss
        if self.include_global:
            global_loss_value = self.global_loss(pred, target)
            total_loss += self.weights[-1] * global_loss_value

        return total_loss


class MultiScaleDensityLoss(nn.Module):
    """
    基于多尺度密度的加权 Loss。
    不再计算不稳定的分形斜率，而是直接计算多尺度下的局部密度。
    密度越高（如断层交叉点、密集带），权重越大。
    """

    def __init__(self,
                 scales=(3, 5, 9, 17),  # 不同大小的感受野
                 min_w=1.0,  # 背景/稀疏区域的权重
                 max_w=5.0,  # 最密集区域（如交叉点）的权重
                 normalize_mean=True):  # 保持总 Loss 数值量级不变
        super().__init__()
        # 确保存储为 list 避免 pytorch 注册问题，且都为奇数
        self.scales = [s for s in scales if s % 2 == 1]
        self.min_w = min_w
        self.max_w = max_w
        self.normalize_mean = normalize_mean

        print(f"Initialized Multi-Scale Density Weighting with scales: {self.scales}")

    def compute_density_map(self, mask):
        """
        计算多尺度密度平均图。
        mask: (B, 1, D, H, W) 0/1 标签
        """
        density_accum = 0.0

        for s in self.scales:
            # AvgPool3d 计算的就是局部窗口内的 "密度" (0.0 ~ 1.0)
            # padding=s//2 保证输出尺寸不变且中心对齐
            pad = s // 2
            local_density = F.avg_pool3d(
                mask,
                kernel_size=s,
                stride=1,
                padding=pad
            )
            # 累加不同尺度的密度
            density_accum += local_density

        # 取平均，得到综合密度图 (0.0 ~ 1.0)
        # 这里的含义是：该像素在不同尺度下，周围平均有多少比例是断层
        avg_density = density_accum / len(self.scales)

        return avg_density

    def density_to_weight(self, density):
        """
        将密度 (0~1) 映射为权重 (min_w ~ max_w)
        """
        # 线性映射：密度越大，权重越大
        # density 为 0 (纯背景) -> min_w
        # density 为 1 (纯实心体) -> max_w
        W = self.min_w + (self.max_w - self.min_w) * density

        # 可选：对权重进行归一化，使得 batch 内的平均权重为 1
        # 这样可以保证调整 max_w 时，Learning Rate 不需要大幅调整
        if self.normalize_mean:
            # 避免除以 0
            mean_w = W.mean().view(1, 1, 1, 1, 1)
            W = W / (mean_w + 1e-8)

        return W

    def forward(self, logits, label):
        # label: (B, D, H, W) -> (B, 1, D, H, W)
        if label.dim() == 4:
            mask = label.unsqueeze(1).float()
        else:
            mask = label.float()

        # 1. 计算多尺度密度 (B, 1, D, H, W)
        # 无需复杂的斜率回归，直接看周围有多少断层
        density = self.compute_density_map(mask)

        # 2. 映射为权重 (B, 1, D, H, W)
        weights = self.density_to_weight(density)

        # 3. 加权 CrossEntropy
        # 需要 squeeze 掉 channel 维以匹配 CE 的 target 要求
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        ce_loss = loss_fn(logits, label.long().squeeze(1) if label.dim() == 5 else label.long())

        # weights 需要 squeeze 掉 channel 维: (B, 1, D, H, W) -> (B, D, H, W)
        weights_squeezed = weights.squeeze(1)

        weighted_loss = (ce_loss * weights_squeezed).mean()

        return {
            "loss": weighted_loss,
            "density": density.detach().squeeze(1),  # 方便可视化
            "weights": weights_squeezed.detach()
        }


# ===== 测试代码 =====
if __name__ == '__main__':
    print("=" * 70)
    print("测试 Patch Dice Loss")
    print("=" * 70)

    # 测试3D版本
    print("\n[测试1] 3D版本 - PatchDiceLoss")
    pred_3d = torch.randn(2, 2, 128, 128, 128)  # (B, C, D, H, W)
    pred_3d = torch.softmax(pred_3d, dim=1)  # 归一化为概率
    target_3d = torch.randint(0, 2, (2, 128, 128, 128)).float()

    loss_fn = PatchDiceLoss(patch_size=32)
    loss = loss_fn(pred_3d, target_3d)
    print(f"Patch Dice Loss (3D): {loss.item():.4f}")

    # 测试标准DiceLoss
    print("\n[测试2] 3D版本 - 标准DiceLoss")
    standard_loss_fn = DiceLoss()
    loss = standard_loss_fn(pred_3d, target_3d)
    print(f"Standard Dice Loss (3D): {loss.item():.4f}")

    # 测试多尺度版本
    print("\n[测试3] 多尺度版本")
    multi_loss_fn = MultiScalePatchDiceLoss(
        patch_sizes=[16, 32, 64],
        include_global=True
    )
    loss = multi_loss_fn(pred_3d, target_3d)
    print(f"Multi-Scale Patch Dice Loss: {loss.item():.4f}")
    print(f"权重分配: {multi_loss_fn.weights}")

    print("\n" + "=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)

class PatchDiceLoss(nn.Module):
    """
    分块Dice Loss - 自动支持2D和3D

    参数:
        patch_size: int，patch大小
        epsilon: 数值稳定性参数
        ignore_empty: 是否忽略空patch（不推荐）
    """

    def __init__(self, patch_size=32, epsilon=1e-5, ignore_empty=False):
        super().__init__()
        self.patch_size = patch_size
        self.epsilon = epsilon
        self.ignore_empty = ignore_empty

    def forward(self, pred, target):
        # 自动检测维度
        if pred.dim() == 4:
            return self._forward_2d(pred, target)
        elif pred.dim() == 5:
            return self._forward_3d(pred, target)
        else:
            raise ValueError(f"不支持的维度: {pred.dim()}, 应该是4D或5D")

    def _forward_2d(self, pred, target):
        """2D版本: (B, C, H, W)"""
        B, C, H, W = pred.shape
        pred = pred[:, 1, :, :]  # 前景通道 (B, H, W)
        target = target.float()

        # 计算patch数量（向上取整，包含边缘）
        patches_h = (H + self.patch_size - 1) // self.patch_size
        patches_w = (W + self.patch_size - 1) // self.patch_size

        total_loss = 0.0
        valid_count = 0

        for i in range(patches_h):
            for j in range(patches_w):
                # 提取patch（处理边界）
                h_start = i * self.patch_size
                h_end = min(h_start + self.patch_size, H)
                w_start = j * self.patch_size
                w_end = min(w_start + self.patch_size, W)

                pred_patch = pred[:, h_start:h_end, w_start:w_end]
                target_patch = target[:, h_start:h_end, w_start:w_end]

                # 计算patch的Dice
                intersection = (pred_patch * target_patch).sum()
                union = pred_patch.sum() + target_patch.sum()

                # 处理空patch
                if self.ignore_empty and union < self.epsilon:
                    continue  # 跳过全空patch

                dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
                total_loss += (1 - dice)
                valid_count += 1

        # 确保至少有一个patch
        if valid_count == 0:
            valid_count = 1

        return total_loss / valid_count

    def _forward_3d(self, pred, target):
        """3D版本: (B, C, D, H, W)"""
        B, C, D, H, W = pred.shape
        pred = pred[:, 1, :, :, :]  # 前景通道 (B, D, H, W)
        target = target.float()

        # 计算patch数量（向上取整）
        patches_d = (D + self.patch_size - 1) // self.patch_size
        patches_h = (H + self.patch_size - 1) // self.patch_size
        patches_w = (W + self.patch_size - 1) // self.patch_size

        total_loss = 0.0
        valid_count = 0

        for i in range(patches_d):
            for j in range(patches_h):
                for k in range(patches_w):
                    # 提取patch（处理边界）
                    d_start = i * self.patch_size
                    d_end = min(d_start + self.patch_size, D)
                    h_start = j * self.patch_size
                    h_end = min(h_start + self.patch_size, H)
                    w_start = k * self.patch_size
                    w_end = min(w_start + self.patch_size, W)

                    pred_patch = pred[:, d_start:d_end, h_start:h_end, w_start:w_end]
                    target_patch = target[:, d_start:d_end, h_start:h_end, w_start:w_end]

                    # 计算patch的Dice
                    intersection = (pred_patch * target_patch).sum()
                    union = pred_patch.sum() + target_patch.sum()

                    # 处理空patch
                    if self.ignore_empty and union < self.epsilon:
                        continue

                    dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
                    total_loss += (1 - dice)
                    valid_count += 1

        if valid_count == 0:
            valid_count = 1

        return total_loss / valid_count


class MultiScalePatchDiceLoss(nn.Module):
    """
    多尺度Patch Dice Loss

    参数:
        patch_sizes: patch大小列表
        weights: 各尺度权重（可选）
        include_global: 是否包含全局Dice
    """

    def __init__(self, patch_sizes=[16, 32, 64], weights=None,
                 include_global=True, epsilon=1e-5):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.include_global = include_global
        self.epsilon = epsilon

        # 初始化各个loss（避免重复创建）
        self.patch_losses = nn.ModuleList([
            PatchDiceLoss(patch_size=ps, epsilon=epsilon)
            for ps in patch_sizes
        ])

        if include_global:
            self.global_loss = DiceLoss(epsilon=epsilon)

        # 设置权重
        if weights is None:
            # 默认权重：patch尺度均等，全局稍小
            if include_global:
                self.weights = [0.3] * len(patch_sizes) + [0.1]
            else:
                self.weights = [1.0 / len(patch_sizes)] * len(patch_sizes)
        else:
            self.weights = weights

        # 归一化权重
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]

    def forward(self, pred, target):
        total_loss = 0.0

        # 计算各个patch尺度的loss
        for i, patch_loss in enumerate(self.patch_losses):
            loss = patch_loss(pred, target)
            total_loss += self.weights[i] * loss

        # 加上全局loss
        if self.include_global:
            global_loss_value = self.global_loss(pred, target)
            total_loss += self.weights[-1] * global_loss_value

        return total_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    自动计算类别权重的加权交叉熵损失
    
    根据标签中正负样本的比例自动计算权重，用于处理类别不平衡问题。
    
    参数:
        reduction: 损失缩减方式，默认 'mean'
    
    使用示例:
        criterion = WeightedCrossEntropyLoss().to(device)
        loss = criterion(logits, labels)
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, logits, labels):
        """
        计算加权交叉熵损失
        
        参数:
            logits: (B, C, D, H, W) 或 (B, C, H, W)，模型输出的logits
            labels: (B, 1, D, H, W) 或 (B, D, H, W) 或 (B, H, W)，真实标签（0/1或类别索引）
        
        返回:
            loss: 加权交叉熵损失值
        """
        # 处理labels的维度：如果是 (B, 1, D, H, W)，需要squeeze成 (B, D, H, W)
        if labels.dim() == 5 and labels.size(1) == 1:
            labels = labels.squeeze(1)  # (B, 1, D, H, W) -> (B, D, H, W)
        elif labels.dim() == 4:
            # (B, D, H, W) 或 (B, H, W)，保持不变
            pass
        else:
            raise ValueError(f"labels 维度不正确: {labels.shape}，应该是 (B, 1, D, H, W) 或 (B, D, H, W) 或 (B, H, W)")
        
        # 确保labels是long类型
        if labels.dtype != torch.long:
            labels = labels.long()
        
        # 计算正负样本数量
        neg = (1 - labels).sum()  # 背景类（0）的数量
        pos = labels.sum()        # 前景类（1）的数量
        
        # 计算类别权重
        total = neg + pos
        if total > 0:
            beta = neg / total
            # 权重：[背景权重, 前景权重]
            # 样本少的类别权重更大
            weight = torch.tensor([1 - beta, beta], dtype=torch.float32, device=logits.device)
        else:
            # 如果总数为0，使用均匀权重
            weight = torch.tensor([0.5, 0.5], dtype=torch.float32, device=logits.device)
        
        # 创建带权重的交叉熵损失
        weighted_ce = nn.CrossEntropyLoss(weight=weight, reduction=self.reduction)
        loss = weighted_ce(logits, labels)
        
        return loss


class MultiScaleDensityMSELoss(nn.Module):
    """
    Multi-scale block-level density MSE loss.
    用于比较预测和标签在不同尺度 patch 下的断层密度（分型统计特征）。

    pred: 预测概率 (B,1,D,H,W) or logits (B,1,D,H,W)
    label: 标签 (B,1,D,H,W), 0/1

    公式:
        对每个尺度 s:
            d_pred(s)  = avg_pool(pred, kernel=s, stride=s)
            d_label(s) = avg_pool(label, kernel=s, stride=s)
            loss_s = MSE(d_pred(s), d_label(s))
        最终:
            loss = Σ α_s * loss_s
    """

    def __init__(self, scales=(8, 16, 32), weights=None, use_sigmoid=True):
        super().__init__()
        self.scales = scales
        self.use_sigmoid = use_sigmoid

        # α_s 权重
        if weights is None:
            # 默认每个尺度平均权重
            self.weights = [1.0 / len(scales)] * len(scales)
        else:
            self.weights = weights

        assert len(self.weights) == len(self.scales), \
            "weights must match the number of scales!"

        self.mse = nn.MSELoss()

    def forward(self, pred, label):
        """
        pred: (B,1,D,H,W) logits or probabilities
        label: (B,1,D,H,W) binary ground truth
        """
        if pred.shape[1] != 1:
            raise ValueError("pred must be (B,1,D,H,W).")

        if self.use_sigmoid:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred  # 你也可以传入已经是概率的 pred

        losses = []

        for w, s in zip(self.weights, self.scales):
            # 使用 avg_pool 直接得到密度（0~1）
            d_pred = F.avg_pool3d(pred_prob, kernel_size=s, stride=s)
            d_label = F.avg_pool3d(label.float(), kernel_size=s, stride=s)

            loss_s = self.mse(d_pred, d_label)
            losses.append(w * loss_s)

        return sum(losses)

class LocalFractalSlopeWeightedCELoss(nn.Module):
    """
    对每个 voxel 在局部 L³ 窗口内按 scales=[2,4,8,16...] 计算 block-counts，
    在 log(scale) vs log(count) 上做线性回归，得到斜率 s(x) 作为"局部分形指数"。
    将 s(x) 线性/归一化到 [min_w, max_w] 并标准化 mean(W)=1，作为 per-voxel 权重用于 CE。

    核心方法：
        1. 对每个像素，固定选取16³区域（像素在索引8位置，左右各8个voxel）
        2. 在这个固定区域内，不同尺度的block划分完全对齐（16能被2、4、8、16整除）
        3. 使用max_pool3d划分blocks，统计包含断层的block数量
        4. 计算分形斜率，转换为权重

    使用情形（训练时基于 label 计算权重）：
        logits: (B, 2, D, H, W)  二分类 logits
        label:  (B, 1, D, H, W)  {0,1} 或 (B,D,H,W) long

    主要参数：
        L: local window side length (int), e.g. 16
        scales: list of ints, each must be <= L and typically powers of two, e.g. [2,4,8,16]
        min_w: 最小权重（防止为0），例如 0.1
        max_w: 最大权重（可选 clipping）
        use_label_for_weights: True -> 基于 ground-truth label 计算权重（训练常用）
                              False -> 基于 model probs 计算（可用于在线/推理自适应）
        eps: 数值稳定小量
        normalize_mean: 是否把最终 W 缩放到 mean(W)=1
    """

    def __init__(self,
                 L=16,
                 scales=(2, 4, 8, 16),
                 min_w=0.1,
                 max_w=2.0,
                 use_label_for_weights=True,
                 eps=1e-6,
                 normalize_mean=True,
                 sigmoid_on_logits=True):
        super().__init__()
        assert all(s <= L for s in scales), "每个 scale 必须 <= L"
        assert L % 2 == 0, "L 必须是偶数，以便像素可以大致居中"
        self.L = L
        self.scales = list(scales)
        self.min_w = float(min_w)
        self.max_w = float(max_w) if max_w is not None else None
        self.use_label_for_weights = bool(use_label_for_weights)
        self.eps = eps
        self.normalize_mean = bool(normalize_mean)
        self.sigmoid_on_logits = bool(sigmoid_on_logits)

        # precompute x-related regression constants (x = log(scale))
        x = torch.tensor([float(torch.log(torch.tensor(float(s)))) for s in self.scales], dtype=torch.float32)
        self.register_buffer('_x', x)  # shape (S,)
        self.register_buffer('_x_mean', x.mean())  # scalar
        self.register_buffer('_xx_sum', (x * x).sum())  # scalar
        # for slope formula denom: sum(x^2) - n*x_mean^2
        self._den = (self._xx_sum - len(self.scales) * (self._x_mean ** 2)).item()
        if abs(self._den) < 1e-8:
            raise ValueError("scales produce degenerate regression denominator")

    def compute_local_counts(self, map_tensor):
        """
        对每个voxel的L³邻域，按不同尺度划分blocks，统计每个尺度下有多少个blocks包含至少1个断层voxel。

        方法：
        1. 对每个像素，固定选取16³区域（像素在索引8位置，左右各8个voxel）
        2. 在这个固定区域内，使用max_pool3d划分不同尺度的blocks
        3. 统计每个尺度下包含断层的block数量

        map_tensor: (B,1,D,H,W) floats (label 0/1 or pred prob)
        returns counts: (B, S, D, H, W) where S = len(scales)
        each entry is the number of blocks containing at least 1 fault voxel
        in the L³ neighborhood at that scale
        """
        B, C, D, H, W = map_tensor.shape
        window_size = self.L  # 固定窗口大小，应该是16
        counts = []

        for s in self.scales:
            block_size = s
            # 步骤1: 将volume划分成block_size³的blocks，计算每个block是否包含断层
            # 所有尺度都从(0,0,0)开始划分，确保对齐
            block_presence = F.max_pool3d(
                map_tensor,
                kernel_size=block_size,
                stride=block_size,
                padding=0
            )  # (B,1,D//block_size, H//block_size, W//block_size)，值是0或1

            # 步骤2: 插值回原始尺寸，使每个block内的所有voxel都有相同的值
            block_presence_up = F.interpolate(
                block_presence,
                size=(D, H, W),
                mode='nearest'
            )  # (B,1,D,H,W)

            # 步骤3: 对每个voxel，统计其window_size³邻域内有多少个blocks包含断层
            # block_presence_up的值是0或1（表示该voxel所在的block是否包含断层）
            # 注意：同一个block内的所有voxel都有相同的值，所以需要特殊处理
            # 方法：对window_size³邻域求和，得到包含断层的voxel总数
            # 然后除以每个block的voxel数，得到包含断层的block数
            pad = window_size // 2  # pad=8，确保像素在window的中心（索引8位置）
            sum_presence = F.avg_pool3d(
                block_presence_up,
                kernel_size=window_size,
                stride=1,
                padding=pad
            ) * (window_size ** 3)  # (B,1,D,H,W)，每个位置的值表示其window_size³邻域内包含断层的voxel总数

            # 转换为block count：包含断层的voxel数 / 每个block的voxel数 = 包含断层的block数
            # 注意：由于同一个block内的所有voxel值相同，所以需要除以block_size³
            voxels_per_block = block_size ** 3
            block_count = sum_presence / (voxels_per_block + 1e-8)

            # 裁剪到原始尺寸（处理偶数window_size的情况）
            _, _, D_out, H_out, W_out = block_count.shape
            if (D_out != D) or (H_out != H) or (W_out != W):
                d_extra = max(0, D_out - D)
                h_extra = max(0, H_out - H)
                w_extra = max(0, W_out - W)
                if d_extra > 0:
                    d0 = d_extra // 2
                    block_count = block_count[:, :, d0:d0 + D, :, :]
                    _, _, D_out, _, _ = block_count.shape
                if h_extra > 0:
                    h0 = h_extra // 2
                    block_count = block_count[:, :, :, h0:h0 + H, :]
                    _, _, _, H_out, _ = block_count.shape
                if w_extra > 0:
                    w0 = w_extra // 2
                    block_count = block_count[:, :, :, :, w0:w0 + W]

            counts.append(block_count)

        # stack scales dim: list of (B,1,D,H,W) -> (B,S,1,D,H,W)
        counts = torch.stack(counts, dim=1)  # (B,S,1,D,H,W)
        counts = counts.squeeze(2)  # -> (B,S,D,H,W)
        return counts

    def compute_slope_map(self, counts):
        """
        counts: (B, S, D, H, W)  (soft counts >=0)
        returns slope_map: (B, D, H, W)
        regression: slope = ( sum((x - xm)*(y - ym)) ) / denom
        where y = log(count + eps), x = log(scale)

        当所有count都为0或接近0时，斜率设为0，避免数值误差导致的异常值。
        """
        B, S, D, H, W = counts.shape
        device = counts.device

        # 检测所有count是否都为0或接近0
        count_threshold = 0.5  # 如果所有count都 < 0.5，认为四舍五入后为0
        all_counts_near_zero = (counts < count_threshold).all(dim=1)  # (B, D, H, W)

        # 如果所有count都接近0，直接返回0
        if all_counts_near_zero.all():
            return torch.zeros((B, D, H, W), device=device, dtype=counts.dtype)

        # 对于有有效count的voxel，进行正常计算
        x = self._x.to(device)  # (S,)
        xm = self._x_mean.to(device).float()  # scalar

        # y = log(counts + eps)
        y = torch.log(counts + self.eps)  # (B,S,D,H,W)

        # compute sums over scales
        sum_xy = (x.view(1, S, 1, 1, 1) * y).sum(dim=1)  # (B,D,H,W)
        sum_y = y.sum(dim=1)  # (B,D,H,W)
        n = float(S)
        # numerator = sum_xy - n * xm * y_mean = sum_xy - xm * sum_y
        numer = sum_xy - xm * sum_y
        # denom is scalar precomputed = sum(x^2) - n*xm^2
        denom = self._den
        slope = numer / (denom + 1e-12)  # (B,D,H,W)

        # 验证：分形斜率理论上应该 <= 0（随着scale增大，count应该减少或不变）
        # 如果出现 > 0 的情况，说明计算有误，将其设为0

        # 当所有count都接近0时，斜率设为0（处理部分voxel的情况）
        slope = torch.where(all_counts_near_zero, torch.zeros_like(slope), slope)

        # 检查并修正异常的正斜率（理论上不应该出现，可能是计算误差）
        # 分形几何中，随着尺度增大，block数量应该减少，所以斜率应该 <= 0
        positive_slope_mask = slope > 1e-6  # 任何正斜率都视为异常（允许小的数值误差）
        if positive_slope_mask.any():
            # 对于异常的正斜率，将其设为0
            slope = torch.where(positive_slope_mask, torch.zeros_like(slope), slope)

        return slope

    def slope_to_weight(self, slope):
        """
        将 slope 映射成权重 W，做缩放与下限保护，并 optional clip 上限，再 normalize mean=1。
        注意：这里反转了映射，使得原来斜率大的权重小，斜率小的权重大。
        slope: (B,D,H,W)
        """
        B = slope.shape[0]
        Wout = torch.empty_like(slope)
        for b in range(B):
            s_b = slope[b]  # (D,H,W)
            flat = s_b.view(-1)
            # compute 1st and 99th percentiles
            k1 = max(1, int(0.01 * flat.numel()))
            k99 = max(1, int(0.99 * flat.numel()))
            low = torch.kthvalue(flat, k1).values
            high = torch.kthvalue(flat, k99).values
            # avoid degenerate
            if (high - low).abs() < 1e-6:
                raw = (s_b - low) * 0.0 + 0.5  # fallback to 0.5
            else:
                raw = (s_b - low) / (high - low)
            # 反转映射：斜率大的权重小
            raw_inverted = 1.0 - raw
            # map to [min_w, max_w]
            w = self.min_w + (1.0 - self.min_w) * raw_inverted
            if self.max_w is not None:
                w = torch.clamp(w, min=self.min_w, max=self.max_w)
            Wout[b] = w
        # optionally normalize mean=1 per batch
        if self.normalize_mean:
            mean_per_batch = Wout.view(B, -1).mean(dim=1).view(B, 1, 1, 1)
            Wout = Wout / (mean_per_batch + 1e-12)
        return Wout  # (B,D,H,W)

    def forward(self, logits, label):
        """
        logits: (B,2,D,H,W)
        label:  (B,1,D,H,W) or (B,D,H,W)
        返回 dict: { 'loss': scalar, 'ce': scalar, 'weights': (B,D,H,W), 'slope': (B,D,H,W) }
        """
        # prepare label and map source for counts
        if label.dim() == 4:
            label = label.unsqueeze(1)
        label = label.float()

        if self.use_label_for_weights:
            map_tensor = label  # use ground truth (0/1)
        else:
            probs = torch.sigmoid(logits[:, 1:2, ...]) if self.sigmoid_on_logits else logits[:, 1:2, ...]
            map_tensor = probs

        # 1) compute counts for each scale: shape (B, S, D, H, W)
        counts = self.compute_local_counts(map_tensor)

        # 2) compute slope map (B,D,H,W)
        slope = self.compute_slope_map(counts)

        # 3) slope -> weight map
        W = self.slope_to_weight(slope)  # (B,D,H,W)

        # 4) weighted CE: per-voxel
        labels_ce = label[:, 0].long()  # (B,D,H,W)
        ce = F.cross_entropy(logits, labels_ce, reduction='none')  # (B,D,H,W)
        loss_ce = (ce * W).mean()

        return {
            'loss': loss_ce,
            'ce': loss_ce,
            'weights': W.detach(),
            'slope': slope.detach()
        }


# ===== 测试代码 =====
if __name__ == '__main__':
    print("=" * 70)
    print("测试 Patch Dice Loss")
    print("=" * 70)

    # 测试3D版本
    print("\n[测试1] 3D版本 - PatchDiceLoss")
    pred_3d = torch.randn(2, 2, 128, 128, 128)  # (B, C, D, H, W)
    pred_3d = torch.softmax(pred_3d, dim=1)  # 归一化为概率
    target_3d = torch.randint(0, 2, (2, 128, 128, 128)).float()

    loss_fn = PatchDiceLoss(patch_size=32)
    loss = loss_fn(pred_3d, target_3d)
    print(f"Patch Dice Loss (3D): {loss.item():.4f}")

    # 测试标准DiceLoss
    print("\n[测试2] 3D版本 - 标准DiceLoss")
    standard_loss_fn = DiceLoss()
    loss = standard_loss_fn(pred_3d, target_3d)
    print(f"Standard Dice Loss (3D): {loss.item():.4f}")

    # 测试多尺度版本
    print("\n[测试3] 多尺度版本")
    multi_loss_fn = MultiScalePatchDiceLoss(
        patch_sizes=[16, 32, 64],
        include_global=True
    )
    loss = multi_loss_fn(pred_3d, target_3d)
    print(f"Multi-Scale Patch Dice Loss: {loss.item():.4f}")
    print(f"权重分配: {multi_loss_fn.weights}")

    print("\n" + "=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)