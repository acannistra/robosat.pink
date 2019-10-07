import os
import io
import sys
import argparse
sys.path.append("../model/robosat_pink/")
os.environ['CURL_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'

import pkgutil
from importlib import import_module

import numpy as np

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from tqdm import tqdm
from PIL import Image

import robosat_pink.models
from robosat_pink.datasets import SlippyMapTiles, BufferedSlippyMapDirectory, S3SlippyMapTiles
from robosat_pink.tiles import tiles_from_slippy_map
from robosat_pink.config import load_config
from robosat_pink.colors import make_palette
from robosat_pink.transforms import ImageToTensor

import albumentations as A

import boto3
import s3fs

def add_parser(subparser):
    parser = subparser.add_parser(
        "predict",
        help="from a trained model and predict inputs, predicts masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="model checkpoint to load")
    parser.add_argument("--workers", type=int, default=0, help="number of workers pre-processing images")

    parser.add_argument("--config", type=str, required=True, help="path to configuration file")
    parser.add_argument("--batch_size", type=int, help="if set, override batch_size value from config file")

    parser.add_argument("--aws_profile", help='aws profile for use in s3 access')

    parser.add_argument("--threshold", help='probability threshold for binarization of predictions (default = 0.0)', default = 0.0)

    parser.add_argument("tiles", type=str, help="directory to read slippy map image tiles from")
    parser.add_argument("preds", type=str, help="directory to save slippy map prediction masks to")

    parser.set_defaults(func=main)

def _write_png(tile, data, outputdir, palette):
    out = Image.fromarray(data, mode="P")
    out.putpalette(palette)

    x = tile[0].item()
    y = tile[1].item()
    z = tile[2].item()

    os.makedirs(os.path.join(outputdir, str(z), str(x)), exist_ok=True)
    path = os.path.join(outputdir, str(z), str(x), str(y) + ".png")

    out.save(path, optimize=True)


def _write_tif(tile, data, outputdir):
    tile_xy_bounds = xy_bounds(tile)
    tile_latlon_bounds = bounds(tile)

    bands = 1
    height, width = data.shape

    new_transform = rio.transform.from_bounds(*tile_latlon_bounds, width, height)

    profile = {
        'driver' : 'GTiff',
        'dtype' : data.dtype,
        'height' : height,
        'width' : width,
        'count' : bands,
        'crs' : {'init' : 'epsg:4326'},
        'transform' : new_transform
    }
    x, y, z = tile[0].item(), tile[1].item(), tile[2].item()

    tile_file = os.path.join(outputdir, str(z), str(x), str(y) + ".tif")

#    try:


    # write data to file
    with rio.open(tile_file, 'w', **profile) as dst:
        for band in range(0, bands ):
            dst.write(data[band], band+1)


    return tile, True


def main(args):
    config = load_config(args.config)
    num_classes = len(config["classes"])
    batch_size = args.batch_size if args.batch_size else config["model"]["batch_size"]
    tile_size = config["model"]["tile_size"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    def map_location(storage, _):
        return storage.cuda() if torch.cuda.is_available() else storage.cpu()

    # check checkpoint situation  + load if ncessary
    chkpt = None # no checkpoint
    if args.checkpoint: # command line checkpoint
        chkpt = args.checkpoint
    try: # config file checkpoint
        chkpt = config["checkpoint"]['path']
    except:
        # no checkpoint in config file
        pass

    S3_CHECKPOINT = False
    if chkpt.startswith("s3://"):
        S3_CHECKPOINT = True
        # load from s3
        chkpt = chkpt[5:]

    models = [name for _, name, _ in pkgutil.iter_modules([os.path.dirname(robosat_pink.models.__file__)])]
    if config["model"]["name"] not in [model for model in models]:
        sys.exit("Unknown model, thoses available are {}".format([model for model in models]))

    num_channels = 0
    for channel in config["channels"]:
        num_channels += len(channel["bands"])

    pretrained = config["model"]["pretrained"]
    encoder = config["model"]["encoder"]

    model_module = import_module("robosat_pink.models.{}".format(config["model"]["name"]))

    net = getattr(model_module, "{}".format(config["model"]["name"].title()))(
        num_classes=num_classes, num_channels=num_channels, encoder=encoder, pretrained=pretrained
    ).to(device)

    net = torch.nn.DataParallel(net)


    try:
        if S3_CHECKPOINT:
            sess = boto3.Session(profile_name=args.aws_profile)
            fs = s3fs.S3FileSystem(session=sess)
            with s3fs.S3File(fs, chkpt, 'rb') as C:
                state = torch.load(io.BytesIO(C.read()), map_location = map_location)
        else:
            state = torch.load(chkpt, map_location= map_location)
        net.load_state_dict(state['state_dict'])
        net.to(device)
    except FileNotFoundError as f:
        print("{} checkpoint not found.".format(chkpt))


    net.eval()

    if args.tiles.startswith('s3://'):
        directory = S3SlippyMapTiles(args.tiles, mode='multibands', transform=None, aws_profile = args.aws_profile)
    else:
        directory = SlippyMapTiles(args.tiles, mode="multibands", transform = transform)
    # directory = BufferedSlippyMapDirectory(args.tiles, transform=transform, size=tile_size, overlap=args.overlap)
    loader = DataLoader(directory, batch_size=batch_size, num_workers=args.workers)

    print(len(directory))

    palette = make_palette(config["classes"][0]["color"])


    # don't track tensors with autograd during prediction
    with torch.no_grad():
        for tiles, images in tqdm(loader, desc="Eval", unit="batch", ascii=True):
            tiles = list(zip(tiles[0], tiles[1], tiles[2]))
            images = images.to(device)
            outputs = net(images)


            print(len(tiles), len(outputs))
            for i, (tile, prob) in enumerate(zip(tiles, outputs)):
                print(tile)
                print("Saving tile {}...".format(i))
                savedir = args.preds

                # manually compute segmentation mask class probabilities per pixel
                image = (prob > args.threshold).cpu().numpy().astype(np.uint8).squeeze()

                _write_png(tile, image, savedir, palette)
