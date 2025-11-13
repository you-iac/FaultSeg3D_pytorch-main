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
        # 自动处理维度
        if pred.dim() == 4:  # 2D: (B, C, H, W)
            pred = pred[:, 1, :, :]
        elif pred.dim() == 5:  # 3D: (B, C, D, H, W)
            pred = pred[:, 1, :, :, :]
        else:
            raise ValueError(f"不支持的维度: {pred.dim()}")

        target = target.float()

        # 计算Dice系数的分子和分母
        intersection = (pred * target).sum()
        dice_coefficient = (2. * intersection + self.epsilon) / (pred.sum() + target.sum() + self.epsilon)

        # 计算Dice Loss
        loss = 1 - dice_coefficient
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