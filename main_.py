# Fault Segmentation Based on Pytorch
import os
import argparse

import numpy as np
import torch
from torch import nn

from models.faultseg3d import FaultSeg3D
from utils.train import train, valid
from utils.test import pred_Gaussian
from utils.test import prediction_all

from utils.tools import save_args_info



def add_args():
    parser = argparse.ArgumentParser(description="FaultSeg3D_pytorch")

    parser.add_argument("--exp", default="400_50_CEDNet_Unet_Dice+2Bcn", type=str, help="Name of each run")
    parser.add_argument("--device", default='cuda:0', type=str, help="GPU id for training")
    parser.add_argument("--mode", default='train', choices=['train', 'valid_only', 'pred', 'pred_all'], type=str, help='network run mode')
    parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
    parser.add_argument("--batch_size_not_train", default=1, type=int, help="number of batch size when not training")
    parser.add_argument("--epochs", default=100, type=int, help="max number of training epochs")
    parser.add_argument("--train_path", default="./data/data_3D_400/train/", type=str, help="dataset directory")
    parser.add_argument("--valid_path", default="./data/data_3D_400/valid/", type=str, help="dataset directory")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
    parser.add_argument("--loss_func", default="dice_plus_ce", choices=['dice', 'cross_with_weight','dice_plus_ce', 'dice_ce_plus_smooth',
                                                                                                                           'dice_plus_cldice', 'dice_plus_topo', 'dice_ce_topo'], type=str, help="choose loss function")
    parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
    parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--workers", default=0, type=int, help="number of workers")
    parser.add_argument("--pretrained_model_name", default="FaultSeg3D_BEST.pth", type=str, help="pretrained model name")
    parser.add_argument("--pred_data_name", default="f3", type=str, help="pretrained data name")
    parser.add_argument("--pred_path", default="f3", type=str, help="pretrained data path")
    parser.add_argument('--overlap', default=0.25, type=int, help='pred‘s overlap')
    parser.add_argument('--threshold', default=0.5, type=float, help='Classification threshold')
    parser.add_argument('--sigma', default=0.0, type=float, help='Gaussian filter sigma')


    args = parser.parse_args()


    print()
    print(">>>============= args ====================<<<")
    print()
    print(args)  # print command line args
    print()
    print(">>>=======================================<<<")

    return args


def main(args):
    # pred_Gaussian(args)

    # args.mode = "valid_only"
    # valid(args)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'valid_only':
        valid(args)
    elif args.mode == 'pred':
        pred_Gaussian(args)
    elif args.mode == 'pred_all':
        prediction_all(args)
    else:
        raise ValueError("Only ['train', 'valid_only', 'pred'] mode is supported.")
    save_args_info(args)


if __name__ == "__main__":
    args = add_args()

    main(args)

# 训练
# python main.py --mode train --exp *** --train_path ./data/train/ --valid_path ./data/valid/ --epochs 50
# python main.py --mode train --exp *** --train_path ./data/data_3D_800/train/ --valid_path ./data/data_3D_800/valid/ --epochs 50
# 预测
# python main.py --mode pred --exp  ***  --pred_data_name *** --pred_path D:/data/***
# 验证
# python main.py --mode valid_only --exp ***



# valid loss:	0.361565550416708
# valid iou:	0.7892225321753248
# valid dice:	0.8720652314570909

# 损失函数
# valid loss:	0.5467045091092586
# valid iou:	0.8509732605697133
# valid dice:	0.914122573012089

# CA 跳跃连接
# valid loss:	0.5347332391887903
# valid iou:	0.8499133697539666
# valid dice:	0.9135304354586736

# Unet
# valid loss:	0.5264931969344616
# valid iou:	0.8565751354414219
# valid dice:	0.9178771569514355




# 400_50_Unet_ant2_B+D_torch2.7.1 使用长方形卷积神经网络，733 533 新torch2.7.1版本
# 400_50_Unet_ant_B+D_torch2.7.1  使用长方形卷积神经网络，337 335 新torch2.7.1版本


#DCN
# valid loss:	0.5474629290401936
# valid iou:	0.8506082754387261
# valid dice:	0.9139575967608697

# ant
# valid loss:	0.5441791109740735
# valid iou:	0.8521923979105901
# valid dice:	0.9149401090752918

# Unet
# valid loss:	0.5467045091092586
# valid iou:	0.8509732605697133
# valid dice:	0.914122573012089