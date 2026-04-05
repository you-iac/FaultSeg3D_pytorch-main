import datetime
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from models.faultseg3d import FaultSeg3D
from utils.domain_adapt import CoralFeatureQueue, compute_da_loss
from utils.tools import con_matrix, compute_loss, load_data, save_result, save_train_info


def _parse_da_layers(raw_layers):
    if isinstance(raw_layers, str):
        layers = tuple(layer.strip() for layer in raw_layers.split(",") if layer.strip())
    elif isinstance(raw_layers, (list, tuple)):
        layers = tuple(str(layer).strip() for layer in raw_layers if str(layer).strip())
    else:
        layers = ("x4",)
    return layers if layers else ("x4",)


def _next_target_batch(target_iter, target_loader):
    try:
        target_batch = next(target_iter)
    except StopIteration:
        target_iter = iter(target_loader)
        target_batch = next(target_iter)
    return target_batch, target_iter


def _forward_with_features(model, x):
    """
    Forward once and always return (pred, feature_dict) for DA.
    Falls back to using output map as x4 if model does not expose features.
    """
    try:
        out = model(x, return_features=True)
        if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], dict):
            return out[0], out[1]
        pred = out
    except TypeError:
        pred = model(x)
    return pred, {"x4": pred}


def train(args, target_train_loader=None):
    # set device
    device = torch.device(args.device)
    print("---")
    print("Device is :", device)

    # Load data
    print("---")
    print("Loading data ... ")
    train_loader, val_loader = load_data(args)
    print("Create model...")

    model = FaultSeg3D(args.in_channels, args.out_channels).to(device)

    # Initialize optimizer
    print("---")
    print("Define optimizer ... ")
    optimizer = optim.Adam(model.parameters(), lr=args.optim_lr)

    # Set model save path   ./EXP/<exp>/models/
    model_path = "./EXP/" + args.exp + "/models/"
    print("---")
    print("The model is saved in :", model_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Domain adaptation config
    use_domain_adapt = bool(getattr(args, "use_domain_adapt", False))
    has_target_loader = target_train_loader is not None and len(target_train_loader) > 0
    da_enabled = use_domain_adapt and has_target_loader
    da_weight = float(getattr(args, "da_weight", 0.0))
    da_layers = _parse_da_layers(getattr(args, "da_layers", "x4"))
    da_use_queue = bool(getattr(args, "da_use_queue", False))

    da_queue = None
    target_iter = None
    if da_enabled:
        target_iter = iter(target_train_loader)
        if da_use_queue:
            da_queue = CoralFeatureQueue(
                layers=da_layers,
                queue_size=int(getattr(args, "da_queue_size", 20)),
                min_samples=int(getattr(args, "da_min_samples", 2)),
            )
        print(
            f"[DA] enabled, layers={da_layers}, weight={da_weight}, "
            f"use_queue={da_use_queue}, target_batches={len(target_train_loader)}"
        )
    elif use_domain_adapt:
        print("[DA] requested but target loader is unavailable. fallback to supervised-only training.")
    else:
        print("[DA] disabled, running supervised-only training.")

    # start training
    print("---")
    print("Start training ... ")

    train_RESULT = []
    val_RESULT = []
    best_iou = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_total_loss = 0.0
        train_seg_loss = 0.0
        train_da_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0

        for _, data in enumerate(
            tqdm(train_loader, desc="[Train] Epoch" + str(epoch + 1) + "/" + str(args.epochs))
        ):
            inputs = data["x"].to(device)
            labels = data["y"].to(device)

            optimizer.zero_grad()

            if da_enabled:
                target_data, target_iter = _next_target_batch(target_iter, target_train_loader)
                target_inputs = target_data["x"].to(device)

                outputs, feats_s = _forward_with_features(model, inputs)
                _, feats_t = _forward_with_features(model, target_inputs)

                seg_loss = compute_loss(outputs, labels, args)
                if da_queue is not None:
                    coral_loss = da_queue.step(feats_s, feats_t)
                else:
                    coral_loss = compute_da_loss(feats_s, feats_t, layers=da_layers)
                total_loss = seg_loss + da_weight * coral_loss
            else:
                outputs = model(inputs)
                seg_loss = compute_loss(outputs, labels, args)
                coral_loss = seg_loss.new_zeros(())
                total_loss = seg_loss

            iou, dice, _, _ = con_matrix(outputs, labels, args)

            total_loss.backward()
            optimizer.step()

            train_total_loss += total_loss.item()
            train_seg_loss += seg_loss.item()
            train_da_loss += coral_loss.item()
            train_iou += iou
            train_dice += dice

            # record train log
            with open(".\\EXP\\" + args.exp + "\\log.txt", "a") as f:
                f.write(
                    str(datetime.datetime.today())
                    + " : epoch: "
                    + str(epoch)
                    + " Times:"
                    + str(global_step)
                    + "\n"
                )
            global_step += 1

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for _, data in enumerate(tqdm(val_loader, desc="[VALID] Valid ")):
                inputs = data["x"].to(device)
                labels = data["y"].to(device)
                outputs = model(inputs)
                loss = compute_loss(outputs, labels, args)
                iou, dice, _, _ = con_matrix(outputs, labels, args)

                val_loss += loss.item()
                val_iou += iou
                val_dice += dice

        print(
            " train total loss: {:.4f}".format(train_total_loss / len(train_loader)),
            " train seg loss: {:.4f}".format(train_seg_loss / len(train_loader)),
            " train coral loss: {:.4f}".format(train_da_loss / len(train_loader)),
            " train iou: {:.4f}".format(train_iou / len(train_loader)),
            " train dice:{:.4f}".format(train_dice / len(train_loader)),
            " val loss: {:.4f}".format(val_loss / len(val_loader)),
            " val iou: {:.4f}".format(val_iou / len(val_loader)),
            " val dice:{:.4f}".format(val_dice / len(val_loader)),
        )

        train_result = np.append(
            train_total_loss / len(train_loader),
            [train_iou / len(train_loader), train_dice / len(train_loader)],
        )
        train_RESULT.append(train_result)

        val_result = np.append(
            val_loss / len(val_loader),
            [val_iou / len(val_loader), val_dice / len(val_loader)],
        )
        val_RESULT.append(val_result)

        if (val_iou / len(val_loader)) > best_iou:
            print("new best ({:.6f} --> {:.6f}). ".format(best_iou, val_iou / len(val_loader)))
            best_iou = val_iou / len(val_loader)
            best_model_name = "FaultSeg3D_BEST.pth".format(epoch + 1, val_iou / len(val_loader))
            torch.save(model.state_dict(), model_path + best_model_name)

        if (epoch + 1) % args.val_every == 0:
            model_name = "FaultSeg3D_epoch_{}_iou_{:.4f}_CP.pth".format(
                epoch + 1, val_iou / len(val_loader)
            )  # CP means checkpoints
            torch.save(model.state_dict(), model_path + model_name)

    # Save training information
    print("---")
    print("Save training information ... ")
    save_train_info(args, train_RESULT, val_RESULT)
    print("---")
    print("Train Finish ! ")
    print("---")
    print("---")
    print("Last validation ... ")
    valid(args, val_loader)

    return 0


def valid(args, val_loader=None):
    device = torch.device(args.device)
    print("---")
    print("Device is :", device)
    # Load data
    print("---")
    print("Loading data ... ")
    if args.mode == "valid_only":
        val_loader = load_data(args)
    # Load Model
    print("---")
    print("Loading Model ... ")
    model = FaultSeg3D(args.in_channels, args.out_channels).to(device)

    model_path = "./EXP/" + args.exp + "/models/" + args.pretrained_model_name
    model.load_state_dict(torch.load(model_path, map_location=device))

    segs = []
    inputs = []
    gts = []

    print("---")
    print("Start validation ... ")

    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    val_acc = 0.0
    val_pre = 0.0

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader, desc="[Valid] Valid")):
            x = data["x"].to(device)
            y = data["y"].to(device)

            outputs = model(x)
            loss = compute_loss(outputs, y, args)
            iou, dice, acc, pre = con_matrix(outputs, y, args)

            val_loss += loss.item()
            val_iou += iou
            val_dice += dice
            val_acc += acc
            val_pre += pre

            segs.append(outputs.detach().cpu().numpy())
            inputs.append(x.detach().cpu().numpy())
            gts.append(y.detach().cpu().numpy())

        print(
            " val loss: {:.4f}".format(val_loss / len(val_loader)),
            " val iou: {:.4f}".format(val_iou / len(val_loader)),
            " val dice:{:.4f}".format(val_dice / len(val_loader)),
            " val acc:{:.4f}".format(val_acc / len(val_loader)),
            " val pre:{:.4f}".format(val_pre / len(val_loader)),
        )

        print("---")
        print("Save result of validation ... ")

        save_result(
            args,
            segs,
            inputs,
            gts,
            val_loss / len(val_loader),
            val_iou / len(val_loader),
            val_dice / len(val_loader),
        )

        print("---")
        print("Save Finished ! ")
