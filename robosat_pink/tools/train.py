import os
import sys
import argparse

import torch
import torch.backends.cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from tqdm import tqdm

import pkgutil
from importlib import import_module

from robosat_pink.transforms import (
    JointCompose,
    JointTransform,
    JointResize,
    JointRandomFlipOrRotate,
    ImageToTensor,
    MaskToTensor,
    AsType
)

from robosat_pink.datasets import PairedTiles
from robosat_pink.metrics import Metrics
from robosat_pink.config import load_config
from robosat_pink.logs import Logs
import robosat_pink.losses
import robosat_pink.models

from numpy import floor, array
from numpy.random import randint

import albumentations as A


def add_parser(subparser):
    parser = subparser.add_parser(
        "train", help="trains a model on a dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config", type=str, required=True, help="path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=False, help="path to a model checkpoint (to retrain)")
    parser.add_argument("--resume", action="store_true", help="resume training (imply to provide a checkpoint)")
    parser.add_argument("--workers", type=int, default=0, help="number of workers pre-processing images")
    parser.add_argument("--dataset", type=str, help="if set, override dataset path value from config file")
    parser.add_argument("--epochs", type=int, help="if set, override epochs value from config file")
    parser.add_argument("--batch_size", type=int, help="if set, override batch_size value from config file")
    parser.add_argument("--lr", type=float, help="if set, override learning rate value from config file")
    parser.add_argument("out", type=str, help="directory to save checkpoint .pth files and log")

    parser.set_defaults(func=main)


def main(args):
    config = load_config(args.config)
    config["dataset"]["path"] = args.dataset if args.dataset else config["dataset"]["path"]
    config["model"]["lr"] = args.lr if args.lr else config["model"]["lr"]
    config["model"]["epochs"] = args.epochs if args.epochs else config["model"]["epochs"]
    config["model"]["batch_size"] = args.batch_size if args.batch_size else config["model"]["batch_size"]

    log = Logs(os.path.join(args.out, "log"))

    if torch.cuda.is_available():
        device = torch.device("cuda")

        torch.backends.cudnn.benchmark = True
        log.log("RoboSat - training on {} GPUs, with {} workers".format(torch.cuda.device_count(), args.workers))
    else:
        device = torch.device("cpu")
        log.log("RoboSat - training on CPU, with {} workers".format(args.workers))

    num_classes = len(config["classes"])
    num_channels = 0
    for channel in config["channels"]:
        num_channels += len(channel["bands"])
    pretrained = config["model"]["pretrained"]
    encoder = config["model"]["encoder"]

    models = [name for _, name, _ in pkgutil.iter_modules([os.path.dirname(robosat_pink.models.__file__)])]
    if config["model"]["name"] not in [model for model in models]:
        sys.exit("Unknown model, thoses available are {}".format([model for model in models]))

    model_module = import_module("robosat_pink.models.{}".format(config["model"]["name"]))
    net = getattr(model_module, "{}".format(config["model"]["name"].title()))(
        num_classes=num_classes, num_channels=num_channels, encoder=encoder, pretrained=pretrained
    ).to(device)

    net = torch.nn.DataParallel(net)
    optimizer = Adam(net.parameters(), lr=config["model"]["lr"], weight_decay=config["model"]["decay"])

    resume = 0
    if args.checkpoint:

        def map_location(storage, _):
            return storage.cuda() if torch.cuda.is_available() else storage.cpu()

        # https://github.com/pytorch/pytorch/issues/7178
        chkpt = torch.load(args.checkpoint, map_location=map_location)
        net.load_state_dict(chkpt["state_dict"])
        log.log("Using checkpoint: {}".format(args.checkpoint))

        if args.resume:
            optimizer.load_state_dict(chkpt["optimizer"])
            resume = chkpt["epoch"]

    losses = [name for _, name, _ in pkgutil.iter_modules([os.path.dirname(robosat_pink.losses.__file__)])]
    if config["model"]["loss"] not in [loss for loss in losses]:
        sys.exit("Unknown loss, thoses available are {}".format([loss for loss in losses]))

    loss_module = import_module("robosat_pink.losses.{}".format(config["model"]["loss"]))
    criterion = getattr(loss_module, "{}".format(config["model"]["loss"].title()))().to(device)

    train_loader, val_loader = get_dataset_loaders(config["dataset"]["path"], config, args.workers)

    if resume >= config["model"]["epochs"]:
        sys.exit(
            "Error: Epoch {} set in {} already reached by the checkpoint provided".format(
                config["model"]["epochs"], args.config
            )
        )

    log.log("")
    log.log("--- Input tensor from Dataset: {} ---".format(config["dataset"]["path"]))
    num_channel = 1
    for channel in config["channels"]:
        for band in channel["bands"]:
            log.log("Channel {}:\t\t {}[band: {}]".format(num_channel, channel["sub"], band))
            num_channel += 1
    log.log("")
    log.log("--- Hyper Parameters ---")
    log.log("Model:\t\t\t {}".format(config["model"]["name"]))
    log.log("Encoder model:\t\t {}".format(config["model"]["encoder"]))
    log.log("Loss function:\t\t {}".format(config["model"]["loss"]))
    log.log("ResNet pre-trained:\t {}".format(config["model"]["pretrained"]))
    log.log("Batch Size:\t\t {}".format(config["model"]["batch_size"]))
    log.log("Tile Size:\t\t {}".format(config["model"]["tile_size"]))
    log.log("Data Augmentation:\t {}".format(config["model"]["data_augmentation"]))
    log.log("Learning Rate:\t\t {}".format(config["model"]["lr"]))
    log.log("Weight Decay:\t\t {}".format(config["model"]["decay"]))
    log.log("")

    for epoch in range(resume, config["model"]["epochs"]):

        log.log("---")
        log.log("Epoch: {}/{}".format(epoch + 1, config["model"]["epochs"]))

        train_hist = train(train_loader, num_classes, device, net, optimizer, criterion)
        log.log(
            "Train    loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
                train_hist["loss"],
                train_hist["miou"],
                config["classes"][1]["title"],
                train_hist["fg_iou"],
                train_hist["mcc"],
            )
        )

        val_hist = validate(val_loader, num_classes, device, net, criterion)
        log.log(
            "Validate loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}".format(
                val_hist["loss"], val_hist["miou"], config["classes"][1]["title"], val_hist["fg_iou"], val_hist["mcc"]
            )
        )

        states = {"epoch": epoch + 1, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict()}
        checkpoint_path = os.path.join(
            args.out, "checkpoint-{:05d}-of-{:05d}.pth".format(epoch + 1, config["model"]["epochs"])
        )
        torch.save(states, checkpoint_path)


def train(loader, num_classes, device, net, optimizer, criterion):
    num_samples = 0
    running_loss= 0

    metrics = Metrics()

    net.train()

    for images, masks, _tile in tqdm(loader, desc="Train", unit="batch", ascii=True):
        print(images)

        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))

        optimizer.zero_grad()
        outputs = net(images)

        assert outputs.size()[2:] == masks.size()[1:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

        loss = criterion(outputs, masks)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            prediction = output.detach()
            metrics.add(mask, prediction)

    assert num_samples > 0, "dataset contains training images and labels"

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }


def validate(loader, num_classes, device, net, criterion):

    num_samples = 0
    running_loss = 0

    metrics = Metrics()
    net.eval()

    with torch.no_grad():
        for images, masks, _tile in tqdm(loader, desc="Validate", unit="batch", ascii=True):
            images = images.to(device)
            masks = masks.to(device)

            assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

            num_samples += int(images.size(0))
            outputs = net(images)

            assert outputs.size()[2:] == masks.size()[1:], "resolutions for predictions and masks are in sync"
            assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

            loss = criterion(outputs, masks)
            running_loss += loss.item()

            for mask, output in zip(masks, outputs):
                metrics.add(mask, output)

    assert num_samples > 0, "dataset contains validation images and labels"

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }


def get_dataset_loaders(dataset_path, config, workers):


    batch_size = config['model']["batch_size"]
    train_percent = config['dataset']['train_percent']
    path = dataset_path
    #
    # transform = JointCompose(
    #     [
    #         #JointRandomFlipOrRotate(config["model"]["data_augmentation"]),
    #         JointTransform(AsType(float), AsType(float)),
    #         JointTransform(ImageToTensor(), MaskToTensor()),
    #         #JointTransform(Normalize(mean=mean, std=std), None),
    #     ]
    # )
    #
    mean = array([[[8237.95084794]],

                   [[6467.98702156]],

                   [[6446.61743148]],

                   [[4520.95360105]]])

    std  = array([[[12067.03414753]],

                   [[ 8810.00542703]],

                   [[10710.64289882]],

                   [[ 9024.92028515]]])

    transform = A.Compose([
        #A.ToFloat(p = 1),
        # A.RandomRotate90(p = 0.5),
        # A.RandomRotate90(p = 0.5),
        # A.RandomRotate90(p = 0.5), #these do something bad to the bands
        A.Normalize(mean = mean, std = std, max_pixel_value = 1),
        A.HorizontalFlip(p = 0.5),
        A.VerticalFlip(p = 0.5),
        A.ToFloat(p = 1)
    ])


    data_tiles = PairedTiles(
        os.path.join(path, "images"),
        os.path.join(path, "mask"), transform
    )

    num_images = len(data_tiles)

    all_indices = set(range(num_images))

    num_train = int(floor(num_images * train_percent))

    train_indices = randint(0, num_images, num_train)

    test_indices = all_indices - set(train_indices)

    train_tiles = PairedTiles(os.path.join(path, "images"),
                              os.path.join(path, "mask"),
                              transform, list(train_indices))

    test_tiles = PairedTiles(os.path.join(path, "images"),
                             os.path.join(path, "mask"),
                             transform, list(test_indices))


    train_loader = DataLoader(train_tiles, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    val_loader = DataLoader(test_tiles, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers)


    return train_loader, val_loader


# def get_dataset_loaders(path, config, workers):
#
#     mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # Values computed on ImageNet DataSet
#
#     transform = JointCompose(
#         [
#             JointResize(config["model"]["tile_size"]),
#             JointRandomFlipOrRotate(config["model"]["data_augmentation"]),
#             JointTransform(ImageToTensor(), MaskToTensor()),
#             JointTransform(Normalize(mean=mean, std=std), None),
#         ]
#     )
#
#     train_dataset = SlippyMapTilesConcatenation(
#         os.path.join(path, "training"),
#         config["channels"],
#         os.path.join(path, "training", "labels"),
#         joint_transform=transform,
#     )
#
#     val_dataset = SlippyMapTilesConcatenation(
#         os.path.join(path, "validation"),
#         config["channels"],
#         os.path.join(path, "validation", "labels"),
#         joint_transform=transform,
#     )
#
#     batch_size = config["model"]["batch_size"]
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers)
#
#     return train_loader, val_loader
