import os
import torch
from dataloader.dataloader import FaultDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.dice_loss import DiceLoss, PatchDiceLoss, MultiScalePatchDiceLoss,WeightedCrossEntropyDiceLoss

from .cldice import soft_cldice, soft_dice
from .TopologicalLoss import TopologicalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F


def save_args_info(args):
    # save args to config.txt
    argsDict = args.__dict__
    result_path = './EXP/' + '/' + args.exp + '/'

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if args.mode == 'train':
        with open(result_path + 'config.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
    elif args.mode == 'valid_only':
        with open(result_path + 'config_valid_only.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
    elif args.mode == 'pred':
        with open(result_path + 'config_pred.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')


def load_data(args):
    # args.mode=['train', 'valid_only', 'pred']
    if args.mode == 'train':
        # 训练时的训练集
        train_dataset = FaultDataset(args.train_path, args.mode, transform=None)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)

        valid_dataset = FaultDataset(args.valid_path, args.mode, transform=None)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size_not_train, shuffle=True,
                                      num_workers=args.workers, drop_last=True)

        print("--- create train dataloader ---")
        print(len(train_dataset), ", train dataset created")
        print(len(train_dataloader), ", train dataloader created")

        print("--- create valid dataloader ---")
        print(len(valid_dataset), ", valid dataset created")
        print(len(valid_dataloader), ", valid dataloaders created")

        return train_dataloader, valid_dataloader

    elif args.mode == 'valid_only':
        dataset = FaultDataset(args.valid_path, args.mode, transform=None)
        dataloader = DataLoader(dataset, batch_size=args.batch_size_not_train, shuffle=True, num_workers=args.workers,
                                drop_last=True)

        print("--- create valid dataloader ---")
        print(len(dataset), ", valid dataset created")
        print(len(dataloader), ", valid dataloaders created")

        return dataloader

    else:  # args.mode=='pred'
        dataset = FaultDataset(args.pred_path, args.mode, transform=None)
        dataloader = DataLoader(dataset, batch_size=args.batch_size_not_train, shuffle=False, num_workers=args.workers,
                                drop_last=True)
        print("--- create prediction dataloader ---")
        print(len(dataset), ", prediction dataset created")
        print(len(dataloader), ", prediction dataloaders created")
        return dataloader


def compute_loss(outputs, labels, args):
    if args.loss_func == 'dice':
        criterion = DiceLoss().to(args.device)
        loss = criterion(outputs, labels)
        return loss
    elif args.loss_func == 'cross_with_weight':
        neg = (1 - labels).sum()  # 算有多少个0
        pos = labels.sum()  # 算有多少个1
        beta = neg / (neg + pos)
        weight = torch.tensor([1 - beta, beta]).to(args.device)
        loss = nn.CrossEntropyLoss(weight=weight, reduction='mean')(outputs, labels.long())
        return loss
    elif args.loss_func == 'dice_plus_ce':
        # print("dice_plus_ce");
        # 计算Dice Loss
        Patch_dice = DiceLoss().to(args.device)
        loss_dice = Patch_dice(outputs, labels)

        # 计算加权交叉熵损失
        neg = (1 - labels).sum()
        pos = labels.sum()
        beta = neg / (neg + pos)
        weight = torch.tensor([1 - beta, beta]).to(args.device)
        loss_ce = nn.CrossEntropyLoss(weight=weight, reduction='mean')(outputs, labels.long())

        # 组合损失（可调整权重系数）
        combined_loss = loss_dice + loss_ce  # 简单相加
        # 或按比例相加：combined_loss = alpha * loss_dice + (1 - alpha) * loss_ce
        return combined_loss
    elif args.loss_func == '_D+MSDW_C':
        "dice_plus_MultiscaleDensityWeights"
        loss_dice = WeightedCrossEntropyDiceLoss().to(args.device)
        loss = loss_dice(outputs, labels)
        return loss
    elif args.loss_func == 'dice_plus_PatchDice':
        # 计算Patch Dice Loss
        patch_size = getattr(args, 'patch_size', 32)  # 默认patch_size=32
        Patch_dice = PatchDiceLoss(patch_size=patch_size).to(args.device)
        loss_dice = Patch_dice(outputs, labels)

        # 计算加权交叉熵损失
        neg = (1 - labels).sum()
        pos = labels.sum()
        beta = neg / (neg + pos)
        weight = torch.tensor([1 - beta, beta]).to(args.device)
        loss_ce = nn.CrossEntropyLoss(weight=weight, reduction='mean')(outputs, labels.long())

        # 组合损失（可调整权重系数）
        combined_loss = loss_dice + loss_ce  # 简单相加
        return combined_loss
    elif args.loss_func == 'multi_scale_patch_dice':
        # 多尺度Patch Dice Loss
        patch_sizes = getattr(args, 'patch_sizes', [16, 32, 64])
        multi_dice = MultiScalePatchDiceLoss(
            patch_sizes=patch_sizes,
            include_global=True
        ).to(args.device)
        loss = multi_dice(outputs, labels)
        return loss
    elif args.loss_func == 'multi_scale_patch_dice_plus_ce':
        # 多尺度Patch Dice + CE
        patch_sizes = getattr(args, 'patch_sizes', [16, 32, 64])
        multi_dice = MultiScalePatchDiceLoss(
            patch_sizes=patch_sizes,
            include_global=True
        ).to(args.device)
        loss_dice = multi_dice(outputs, labels)

        # 计算加权交叉熵损失
        neg = (1 - labels).sum()
        pos = labels.sum()
        beta = neg / (neg + pos)
        weight = torch.tensor([1 - beta, beta]).to(args.device)
        loss_ce = nn.CrossEntropyLoss(weight=weight, reduction='mean')(outputs, labels.long())

        combined_loss = loss_dice + loss_ce
        return combined_loss
    elif args.loss_func == 'dice_ce_plus_smooth':
        # 1. 计算基础分割损失 (Dice + 加权CE)
        # 假设 DiceLoss 和 CrossEntropyLoss 能够正确处理 (B, C, D, H, W) 的 3D 输入
        Patch_dice = DiceLoss().to(args.device)
        loss_dice = Patch_dice(outputs, labels)

        # 计算权重
        neg = (1 - labels).sum()
        pos = labels.sum()
        beta = neg / (neg + pos)
        # 假设是二分类，权重为 [背景权重, 前景权重]
        weight = torch.tensor([1 - beta, beta]).to(args.device)

        # CrossEntropyLoss 的输入是 logits (outputs)
        loss_ce = nn.CrossEntropyLoss(weight=weight, reduction='mean')(outputs, labels.long())

        loss_seg = loss_dice + loss_ce

        # 2. 计算 3D 连续性/平滑性惩罚项 (L_smooth_3D)

        # 从 Logits (outputs) 中获取前景类 (C=1) 的预测概率 P
        # 假设 outputs 形状为 (B, 2, D, H, W)，前景类是索引 1
        probabilities = torch.sigmoid(outputs[:, 1, :, :, :])  # 形状为 (B, D, H, W)

        # 惩罚项 L_smooth_3D = Sum( (P_x)^2 + (P_y)^2 + (P_z)^2 ) / N

        # 计算深度 (z) 方向的梯度差 (dim=1，因为索引 0 是 Batch)
        diff_z = torch.diff(probabilities, dim=1)
        smooth_loss_z = torch.mean(diff_z.pow(2))

        # 计算高度 (y) 方向的梯度差 (dim=2)
        diff_y = torch.diff(probabilities, dim=2)
        smooth_loss_y = torch.mean(diff_y.pow(2))

        # 计算宽度 (x) 方向的梯度差 (dim=3)
        diff_x = torch.diff(probabilities, dim=3)
        smooth_loss_x = torch.mean(diff_x.pow(2))

        # 3D 平滑性损失：三个方向的梯度平方均值之和
        loss_smooth_3d = smooth_loss_x + smooth_loss_y + smooth_loss_z

        # 3. 组合总损失
        # 这里的 smooth_lambda 是关键超参数
        smooth_lambda = getattr(args, 'smooth_lambda', 0.1)  # 默认值 0.01

        total_loss = loss_seg + smooth_lambda * loss_smooth_3d

        return total_loss
    elif args.loss_func == 'dice_plus_cldice':

        # 快速模式：减少clDice计算开销
        fast_mode = getattr(args, 'fast_mode', True)  # 默认启用快速模式

        CL_ITER = getattr(args, 'cl_iter', 20)
        ALPHA_CL = getattr(args, 'alpha_cl', 0.3)
        DOWNSAMPLE_SIZE = getattr(args, 'downsample_size', [64, 64, 64])

        # 初始化损失函数
        criterion_cldice = soft_cldice(iter_=CL_ITER, smooth=1., exclude_background=True).to(args.device)
        Patch_dice = DiceLoss().to(args.device)

        # --- 2. 数据准备 ---
        y_pred = F.softmax(outputs, dim=1)
        num_classes = outputs.size(1)

        if labels.dim() == outputs.dim() - 1:
            y_true_onehot = F.one_hot(labels.long(), num_classes=num_classes)
            y_true_onehot = y_true_onehot.permute(0, 4, 1, 2, 3).float()
        else:
            y_true_onehot = labels.float()

        # --- 3. 性能优化：对 clDice 的输入进行降采样 ---
        # 降采样预测概率 (使用三线性插值，保持连续性)
        y_pred_small = F.interpolate(
            y_pred,
            size=DOWNSAMPLE_SIZE,
            mode='trilinear',
            align_corners=False
        )

        # 降采样独热编码标签 (使用最近邻插值，保持 0/1 属性)
        y_true_onehot_small = F.interpolate(
            y_true_onehot,
            size=DOWNSAMPLE_SIZE,
            mode='nearest'
        )

        # --- 4. 计算损失 ---
        # clDice 损失 (soft_cldice已经返回损失值，不需要1-操作)
        loss_cldice = criterion_cldice(y_true_onehot_small, y_pred_small)

        # Dice 损失
        loss_dice = Patch_dice(outputs, labels)

        # 组合损失
        combined_loss = (1.0 - ALPHA_CL) * loss_dice + ALPHA_CL * loss_cldice

        # 可选：添加调试信息
        if getattr(args, 'debug_loss', False):
            print(
                f"Loss components - Dice: {loss_dice.item():.4f}, clDice: {loss_cldice.item():.4f}, Combined: {combined_loss.item():.4f}")

        return combined_loss
    elif args.loss_func == 'dice_plus_topo':
        # 组合 Dice 损失和拓扑损失
        Patch_dice = DiceLoss().to(args.device)

        # 拓扑损失参数
        lambda_topo = getattr(args, 'lambda_topo', 0.1)  # 拓扑损失权重
        dim_H0 = getattr(args, 'dim_H0', True)  # 是否使用连通性约束
        dim_H1 = getattr(args, 'dim_H1', True)  # 是否使用孔洞约束

        criterion_topo = TopologicalLoss(
            lambda_topo=lambda_topo,
            dim_H0=dim_H0,
            dim_H1=dim_H1
        ).to(args.device)

        # 计算 Dice 损失
        loss_dice = Patch_dice(outputs, labels)

        # 计算拓扑损失
        loss_topo = criterion_topo(outputs, labels)

        # 组合损失
        combined_loss = loss_dice + loss_topo

        # 可选：添加调试信息
        if getattr(args, 'debug_loss', False):
            print(
                f"Loss components - Dice: {loss_dice.item():.4f}, Topo: {loss_topo.item():.4f}, Combined: {combined_loss.item():.4f}")

        return combined_loss
    elif args.loss_func == 'dice_ce_topo':
        # 组合 Dice + CE + 拓扑损失
        Patch_dice = DiceLoss().to(args.device)

        # 计算权重
        neg = (1 - labels).sum()
        pos = labels.sum()
        beta = neg / (neg + pos)
        weight = torch.tensor([1 - beta, beta]).to(args.device)

        # 计算各种损失
        loss_dice = Patch_dice(outputs, labels)
        loss_ce = nn.CrossEntropyLoss(weight=weight, reduction='mean')(outputs, labels.long())

        # 拓扑损失参数
        lambda_topo = getattr(args, 'lambda_topo', 0.05)  # 较小的拓扑权重
        criterion_topo = TopologicalLoss(
            lambda_topo=lambda_topo,
            dim_H0=True,
            dim_H1=True
        ).to(args.device)

        loss_topo = criterion_topo(outputs, labels)

        # 组合损失
        combined_loss = loss_dice + loss_ce + loss_topo

        if getattr(args, 'debug_loss', False):
            print(
                f"Loss components - Dice: {loss_dice.item():.4f}, CE: {loss_ce.item():.4f}, Topo: {loss_topo.item():.4f}, Combined: {combined_loss.item():.4f}")

        return combined_loss

    else:
        raise ValueError("Only ['dice', 'cross_with_weight', 'dice_plus_ce', 'dice_plus_PatchDice', "
                         "'multi_scale_patch_dice', 'multi_scale_patch_dice_plus_ce', "
                         "'dice_ce_plus_smooth', 'dice_plus_cldice', 'dice_plus_topo', 'dice_ce_topo'] "
                         "loss is supported.")


def con_matrix(outputs, labels, args):
    y_pred = outputs.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    y_pred = y_pred.argmax(axis=1).flatten()
    y_true = y_true.flatten()

    num_class = args.out_channels
    current = confusion_matrix(y_true, y_pred, labels=range(num_class))  # confusion_matrix混淆矩阵，计算把xxx预测成xxx的次数

    TP = current[1][1]
    TN = current[0][0]
    FP = current[1][0]
    FN = current[0][1]

    Acc = (TP + TN) / (TP + TN + FP + FN)
    Pre = TP / (TP + FP)

    # compute mean iou
    intersection = np.diag(current)
    # 一维数组的形式返回混淆矩阵的对角线元素
    ground_truth_set = current.sum(axis=1)
    # 按行求和
    predicted_set = current.sum(axis=0)
    # 按列求和
    union = ground_truth_set + predicted_set - intersection + 1e-7
    IoU = intersection / union.astype(np.float32)
    union_dice = ground_truth_set + predicted_set + 1e-7
    DICE = 2 * intersection / union_dice.astype(np.float32)

    return np.mean(IoU), np.mean(DICE), np.mean(Acc), np.mean(Pre)


def save_train_info(args, train_RESULT, val_RESULT):
    if not os.path.exists('./EXP/' + args.exp + '/results/train/'):
        os.makedirs('./EXP/' + args.exp + '/results/train/')

    data_df = pd.DataFrame(train_RESULT)
    data_df.columns = ['train_loss', 'train_iou', 'train_dice']
    data_df.index = np.arange(0, args.epochs, 1)
    writer = pd.ExcelWriter('./EXP/' + args.exp + '/results/train/train_result.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer._save()
    writer.close()

    data_df_val = pd.DataFrame(val_RESULT)
    data_df_val.columns = ['val_loss', 'val_iou', 'val_dice']
    data_df_val.index = np.arange(0, args.epochs, 1)
    writer_val = pd.ExcelWriter('./EXP/' + args.exp + '/results/train/val_result.xlsx')
    data_df_val.to_excel(writer_val, 'page_1', float_format='%.5f')
    writer_val._save()


def save_result(args, segs, inputs, gts, val_loss, val_iou, val_dice):
    result_path = './EXP/' + args.exp + '/results/valid/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path + "valid_final_result.txt", 'a+') as f:
        f.write('valid loss:\t' + str(val_loss) + '\n')
        f.write('valid iou:\t' + str(val_iou) + '\n')
        f.write('valid dice:\t' + str(val_dice) + '\n')

    if not os.path.exists(result_path + '/numpy/'):
        os.makedirs(result_path + '/numpy/')
    if not os.path.exists(result_path + '/picture/'):
        os.makedirs(result_path + '/picture/')

    for i in range(len(inputs)):

        seg = segs[i].argmax(axis=1)
        img = inputs[i]
        gt = gts[i]
        seg = np.squeeze(seg)
        img = np.squeeze(img)
        gt = np.squeeze(gt)
        # save output
        np.save(result_path + '/numpy/' + str(i) + '_seg.npy', seg)
        np.save(result_path + '/numpy/' + str(i) + '_img.npy', img)
        np.save(result_path + '/numpy/' + str(i) + '_gt.npy', gt)
        # save picture

        index = np.arange(0, 128, 50)
        for idx in index:
            # dim 0
            plt.subplot(1, 3, 1)
            plt.imshow(img[idx, :, :])
            plt.axis('off')
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(gt[idx, :, :])
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(seg[idx, :, :])
            plt.axis('off')
            plt.title('Segmentation')

            plt.savefig(result_path + '/picture/No_' + str(i) + '_idx_' + str(idx) + '_dim_0.png')
            plt.close()
            # dim 1
            plt.subplot(1, 3, 1)
            plt.imshow(img[:, idx, :])
            plt.axis('off')
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(gt[:, idx, :])
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(seg[:, idx, :])
            plt.axis('off')
            plt.title('Segmentation')

            plt.savefig(result_path + '/picture/No_' + str(i) + '_idx_' + str(idx) + '_dim_1.png')
            plt.close()
            # dim 2
            plt.subplot(1, 3, 1)
            plt.imshow(img[:, :, idx])
            plt.axis('off')
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(gt[:, :, idx])
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(seg[:, :, idx])
            plt.axis('off')
            plt.title('Segmentation')

            plt.savefig(result_path + '/picture/No_' + str(i) + '_idx_' + str(idx) + '_dim_2.png')
            plt.close()


def load_pred_data(args):
    # path = args.pred_path
    data = np.load(args.pred_path)
    # data = np.transpose(data, (2, 0, 1))

    print(np.shape(data))
    # if args.pred_data_name == 'f3':
    #     print("Data use f3.")
    #     data = np.load('f3_data_path')
    #     return data
    # elif args.pred_data_name == 'kerry3d':
    #     print("Data use kerry.")
    #     data = np.load('D:\data\kerry3d.npy')
    #     data = np.transpose(data, (2, 0, 1))
    #     return data
    # else:
    #     raise ValueError("Only ['f3', 'kerry'] mode is supported.")

    return data


def save_pred_picture(gx, gy, save_path, pred_data_name):
    k1, k2, k3 = 80, 80, 80
    gx1 = gx[k1, :, :]
    gy1 = gy[k1, :, :]
    gx2 = gx[:, k2, :]
    gy2 = gy[:, k2, :]
    gx3 = gx[:, :, k3]
    gy3 = gy[:, :, k3]

    # xline slice
    plt.subplot(1, 2, 1)
    plt.imshow(gx1, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(gy1, cmap='gray')

    plt.savefig(save_path + pred_data_name + '_dim_0.png', dpi=600)

    # inline slice
    plt.subplot(1, 2, 1)
    plt.imshow(gx2, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(gy2, cmap='gray')

    plt.savefig(save_path + pred_data_name + '_dim_1.png', dpi=600)

    # time slice
    plt.subplot(1, 2, 1)
    plt.imshow(gx3, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(gy3, cmap='gray')

    plt.savefig(save_path + pred_data_name + '_dim_2.png', dpi=600)
