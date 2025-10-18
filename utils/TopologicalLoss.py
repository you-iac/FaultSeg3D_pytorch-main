import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_topological.nn import CubicalComplex
from torch_topological.nn import WassersteinDistance


class TopologicalLoss(nn.Module):
    """
    基于持久同调（Persistent Homology）的通用拓扑损失函数。
    它约束了预测结果与真实标签在连通性(H0)和孔洞(H1)上的拓扑差异。
    """

    def __init__(self, lambda_topo=0.1, dim_H0=True, dim_H1=True):
        super().__init__()
        self.lambda_topo = lambda_topo  # 拓扑损失的权重

        # 1. 配置同调维度
        dims = []
        if dim_H0:
            dims.append(0)
        if dim_H1:
            dims.append(1)

        # CubicalPersistence 适用于体素数据 (3D 或 2D)
        if not dims:
            raise ValueError("至少需要选择 H0 或 H1 中的一个维度进行约束。")

        self.ph_computer = CubicalComplex(dim=dims)
        self.dimensions = dims  # 记录配置的维度

        # 2. 距离计算器
        # 使用 p=2 的 Wasserstein 距离 (W2)
        self.wasserstein = WassersteinDistance(p=2)

        print(f"TopologicalLoss initialized. Dims: {dims}, lambda_topo={lambda_topo}")

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算拓扑损失。
        Args:
            outputs: 模型的 logit 或概率输出 (B, C, D, H, W)。
            targets: 真实标签 (B, 1, D, H, W) 或 (B, D, H, W)。
        """

        # --- 0. 数据预处理 ---
        # 假设是二分类分割，C=2 (背景+前景)，提取前景概率
        P_prob = F.softmax(outputs, dim=1)[:, 1, ...].contiguous()

        # G (Ground Truth) 必须是浮点数张量，且维度匹配 P_prob
        # 假设 targets 已经处理到 (B, D, H, W) 或 (B, H, W) 形状
        G = targets.squeeze(1).float().contiguous()

        # 确保输入张量形状是 PH 计算机可接受的 (例如 BATCH, D, H, W)
        if P_prob.dim() != 4:
            # 检查是否为 3D 数据 (B, D, H, W)
            raise ValueError(f"Input P_prob must be 4D (B, D, H, W) for 3D processing, got {P_prob.dim()}D")

        # --- 1. 计算持久图 (Persistence Diagrams) ---
        pd_predicted = self.ph_computer(P_prob)
        pd_ground_truth = self.ph_computer(G)

        # --- 2. 计算 Wasserstein 距离作为损失 ---
        total_topo_loss = 0.0

        # 逐个维度计算距离并求和
        for i, dim in enumerate(self.dimensions):
            # self.wasserstein 返回 (distance_tensor, matching_index)
            distance, _ = self.wasserstein(pd_predicted[i], pd_ground_truth[i])

            # 权重可以根据维度调整，这里使用统一的 lambda_topo
            total_topo_loss += distance.mean()

        L_topo = self.lambda_topo * total_topo_loss

        return L_topo