# Fault Segmentation Based on Pytorch
import os
import argparse

import torch
from torch import nn

from models.faultseg3d import FaultSeg3D
from utils.train import train, valid
from utils.test import pred_Gaussian
from utils.tools import save_args_info


def add_args():
    parser = argparse.ArgumentParser(description="FaultSeg3D_pytorch")

    parser.add_argument("--exp", default="test", type=str, help="Name of each run")
    parser.add_argument("--device", default='cuda:0', type=str, help="GPU id for training")
    parser.add_argument("--mode", default='train', choices=['train', 'valid_only', 'pred'], type=str, help='network run mode')
    parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
    parser.add_argument("--batch_size_not_train", default=1, type=int, help="number of batch size when not training")
    parser.add_argument("--epochs", default=25, type=int, help="max number of training epochs")
    parser.add_argument("--train_path", default="./data/train/", type=str, help="dataset directory")
    parser.add_argument("--valid_path", default="./data/valid/", type=str, help="dataset directory")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
    parser.add_argument("--loss_func", default="cross_with_weight", choices=['dice', 'cross_with_weight'], type=str, help="choose loss function")
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
    else:
        raise ValueError("Only ['train', 'valid_only', 'pred'] mode is supported.")
    save_args_info(args)


if __name__ == "__main__":
    args = add_args()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # print(torch.cuda.device_count())
    #
    # model = FaultSeg3D(args.in_channels, args.out_channels) #载入模型
    #
    # if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)  # 将模型对象转变为多GPU并行运算的模型
    #
    # model.to(args.device)  # 把并行的模型移动到GPU上

    main(args)

# 训练
# python main.py --mode train --exp *** --train_path ./data/train/ --valid_path ./data/valid/ --epochs 50
# python main.py --mode train --exp *** --train_path ./data/data_3D_800/train/ --valid_path ./data/data_3D_800/valid/ --epochs 50
# 预测
# python main.py --mode pred --exp  ***  --pred_data_name *** --pred_path D:/data/***
# 验证
# python main.py --mode valid_only --exp ***
