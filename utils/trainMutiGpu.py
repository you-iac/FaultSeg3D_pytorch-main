import datetime
import os
import time

import torch
from torch import nn
from tqdm import tqdm
from utils.tools import load_data, compute_loss, con_matrix, save_train_info, save_result
import torch.optim as optim
from models.faultseg3d import FaultSeg3D
import numpy as np


def trainMutiGpu(args):
    # ==================== 修复1: 正确的多GPU设备设置 ====================
    # 检测GPU数量
    num_gpus = torch.cuda.device_count()
    print("=" * 60)
    print(f"检测到 {num_gpus} 个GPU")
    print("=" * 60)

    # 获取所有GPU信息
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

    # 设置主设备（DataParallel会自动处理多GPU）
    if num_gpus > 0:
        # DataParallel需要将模型放在一个主GPU上，这里选择GPU 0
        main_gpu = 0
        args.device = f'cuda:{main_gpu}'
        device = torch.device(args.device)
        print(f"主设备设置为: {device}")
    else:
        print("错误: 未检测到可用GPU!")
        return -1

    # 加载数据
    print("---")
    print("加载数据 ... ")
    train_loader, val_loader = load_data(args)
    print('创建模型...')

    model = FaultSeg3D(args.in_channels, args.out_channels)

    # ==================== 修复2: 正确的多GPU配置 ====================
    if num_gpus > 1:
        print(f"使用 {num_gpus} 个GPU进行训练!")

        # 1. 设置性能优化
        torch.backends.cudnn.benchmark = True

        # 2. 使用所有可用GPU，不硬编码device_ids
        device_ids = list(range(num_gpus))
        print(f"使用的GPU设备ID: {device_ids}")

        # 3. 使用DataParallel包装模型
        model = nn.DataParallel(model, device_ids=device_ids)

        # 4. 设置主GPU（DataParallel默认使用device_ids[0]作为主GPU）
        torch.cuda.set_device(main_gpu)

        # 5. 清空所有GPU缓存
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()

        print(f"模型分布在GPU: {device_ids}上")
        print(f"主GPU (device_ids[0]): {main_gpu}")
    else:
        print("使用单个GPU进行训练")

    # 将模型移动到主设备
    model = model.to(device)

    # ==================== 修复3: 调整批处理大小 ====================
    # 多GPU时，总批处理大小 = args.batch_size × GPU数量
    # 确保每个GPU上的批处理大小合理
    if num_gpus > 1:
        effective_batch_size = args.batch_size * num_gpus
        print(f"批处理大小说明:")
        print(f"  - 配置的batch_size: {args.batch_size}")
        print(f"  - GPU数量: {num_gpus}")
        print(f"  - 有效总批处理大小: {effective_batch_size}")
        print(f"  - 每个GPU处理的批大小: ~{args.batch_size}")

    # 初始化优化器
    print("---")
    print("定义优化器 ... ")

    optimizer = optim.Adam(model.parameters(), lr=args.optim_lr)

    # 设置模型保存路径
    model_path = './EXP/' + args.exp + '/models/'
    print("---")
    print("模型保存路径: ", model_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # 开始训练
    print("---")
    print("开始训练 ... ")

    train_RESULT = []
    val_RESULT = []

    best_iou = 0.0

    step_counter = 0  # 修复变量名冲突，避免与time模块同名

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0

        # ==================== 修复4: 正确的数据移动 ====================
        for step, data in enumerate(tqdm(train_loader, desc='[Train] Epoch' + str(epoch + 1) + '/' + str(args.epochs))):
            # 正确的方式：将数据移动到主设备（DataParallel会自动分发到其他GPU）
            inputs, labels = data['x'].to(device), data['y'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)  # DataParallel自动处理多GPU前向传播
            loss = compute_loss(outputs, labels, args)
            iou, dice, acc, pre = con_matrix(outputs, labels, args)

            loss.backward()  # 梯度自动收集到主GPU
            optimizer.step()  # 优化器在主GPU上更新参数

            train_loss += loss.item()
            train_iou += iou
            train_dice += dice

            # 记录训练日志
            log_path = os.path.join('.', 'EXP', args.exp, 'log.txt')
            with open(log_path, 'a') as f:
                f.write(str(datetime.datetime.today()) + ' : epoch: ' + str(epoch) +
                        ' step:' + str(step_counter) + '\n')
            step_counter = step_counter + 1

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for step, data in enumerate(tqdm(val_loader, desc='[VALID] Valid ')):
                # 同样将数据移动到主设备
                inputs = data['x'].to(device)
                labels = data['y'].to(device)
                outputs = model(inputs)
                loss = compute_loss(outputs, labels, args)
                iou, dice, acc, pre = con_matrix(outputs, labels, args)

                val_loss += loss.item()
                val_iou += iou
                val_dice += dice

        # 打印训练结果
        print(
            " train loss: {:.4f}".format(train_loss / len(train_loader)),
            " train iou: {:.4f}".format(train_iou / len(train_loader)),
            " train dice:{:.4f}".format(train_dice / len(train_loader)),
            " val loss: {:.4f}".format(val_loss / len(val_loader)),
            " val iou: {:.4f}".format(val_iou / len(val_loader)),
            " val dice:{:.4f}".format(val_dice / len(val_loader))
        )

        train_result = np.append(train_loss / len(train_loader),
                                 [train_iou / len(train_loader), train_dice / len(train_loader)])
        train_RESULT.append(train_result)

        val_result = np.append(val_loss / len(val_loader),
                               [val_iou / len(val_loader), val_dice / len(val_loader)])
        val_RESULT.append(val_result)

        # ==================== 修复5: 保存模型时的注意事项 ====================
        if (val_iou / len(val_loader)) > best_iou:
            print("new best ({:.6f} --> {:.6f}). ".format(best_iou, val_iou / len(val_loader)))
            best_iou = val_iou / len(val_loader)

            # DataParallel包装的模型需要用module属性访问原始模型
            if num_gpus > 1:
                # 保存时去除DataParallel包装
                best_model_name = 'FaultSeg3D_BEST.pth'
                torch.save(model.module.state_dict(), model_path + best_model_name)
            else:
                best_model_name = 'FaultSeg3D_BEST.pth'
                torch.save(model.state_dict(), model_path + best_model_name)

            print(f"最佳模型已保存: {best_model_name}")

        # 定期保存检查点
        if (epoch + 1) % args.val_every == 0:
            model_name = 'FaultSeg3D_epoch_{}_iou_{:.4f}_CP.pth'.format(epoch + 1, val_iou / len(val_loader))

            if num_gpus > 1:
                torch.save(model.module.state_dict(), model_path + model_name)
            else:
                torch.save(model.state_dict(), model_path + model_name)

            print(f"检查点已保存: {model_name}")

    # 保存训练信息
    print("---")
    print("保存训练信息 ... ")
    save_train_info(args, train_RESULT, val_RESULT)
    print("---")
    print("训练完成! ")
    print("---")
    print("最终验证 ... ")

    # 最终验证
    valid(args, val_loader)

    return 0


def valid(args, val_loader=None):
    # ==================== 修复6: 验证函数的多GPU支持 ====================
    # 检测GPU数量
    num_gpus = torch.cuda.device_count()

    if num_gpus > 0:
        main_gpu = 0
        args.device = f'cuda:{main_gpu}'
        device = torch.device(args.device)
    else:
        print("错误: 未检测到可用GPU!")
        return

    print("---")
    print('设备: ', device)

    # 加载数据
    print("---")
    print("加载数据 ... ")
    if args.mode == 'valid_only':
        val_loader = load_data(args)

    # 加载模型
    print("---")
    print("加载模型 ... ")
    model = FaultSeg3D(args.in_channels, args.out_channels)

    model_path = './EXP/' + args.exp + '/models/' + args.pretrained_model_name

    # 加载模型权重
    model.load_state_dict(torch.load(model_path))

    # 多GPU支持
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        model = nn.DataParallel(model, device_ids=device_ids)
        torch.cuda.set_device(main_gpu)
        print(f"使用 {num_gpus} 个GPU进行验证")

    model = model.to(device)

    segs = []
    inputs = []
    gts = []

    print("---")
    print("开始验证 ... ")

    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    val_acc = 0.0
    val_pre = 0.0

    model.eval()
    with torch.no_grad():
        for step, data in enumerate(tqdm(val_loader, desc='[Valid] Valid')):
            x = data['x'].to(device)  # 移动到主设备
            y = data['y'].to(device)  # 移动到主设备

            outputs = model(x)
            loss = compute_loss(outputs, y, args)
            iou, dice, acc, pre = con_matrix(outputs, y, args)

            val_loss += loss.item()
            val_iou += iou
            val_dice += dice
            val_acc += acc
            val_pre += pre

            # 将结果移回CPU
            segs.append(outputs.detach().cpu().numpy())
            inputs.append(x.detach().cpu().numpy())
            gts.append(y.detach().cpu().numpy())

        print(
            " val loss: {:.4f}".format(val_loss / len(val_loader)),
            " val iou: {:.4f}".format(val_iou / len(val_loader)),
            " val dice:{:.4f}".format(val_dice / len(val_loader)),
            " val acc:{:.4f}".format(val_acc / len(val_loader)),
            " val pre:{:.4f}".format(val_pre / len(val_loader))
        )

        print("---")
        print("保存验证结果 ... ")

        save_result(args, segs, inputs, gts, val_loss / len(val_loader),
                    val_iou / len(val_loader), val_dice / len(val_loader))

        print("---")
        print("保存完成! ")