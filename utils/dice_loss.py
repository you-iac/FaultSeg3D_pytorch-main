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