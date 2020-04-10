import os
import io
import sys
import argparse
sys.path.append("../model/robosat_pink/")
os.environ['CURL_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'

import pkgutil
from importlib import import_module

import numpy as np
import rasterio as rio

import pprint
from re import match

import pandas as pd

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Compose, Normalize

from tqdm import tqdm
from PIL import Image

from mercantile import Tile, xy_bounds, bounds

import robosat_pink.models
from robosat_pink.datasets import SlippyMapTiles, BufferedSlippyMapDirectory, S3SlippyMapTiles, SlippyMapTilesConcatenation
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

    parser.add_argument("--create_tif", action='store_true', help="create .tif tiles for mask")

    parser.add_argument("--aws_profile", help='aws profile for use in s3 access')

    parser.add_argument("--threshold", help='probability threshold for binarization of predictions (default = 0.0)', default = 0.0)

    parser.add_argument("--tiles", type=str, help="directory to read slippy map image tiles from. Will use config if not provided.")
    parser.add_argument("preds", type=str, help="directory to save slippy map prediction masks to")

    parser.add_argument("--buffer", action='store_true',
     help="Buffer images before prediction to avoid artifacts. "
    )

    parser.add_argument("--buffer_overlap", type=int, default=64,
    help="Neighboring-tile overlap width in pixels for buffer.")

    parser.add_argument("--tile_ids", type=str, help="File containing image ids to use for prediction.")

    parser.set_defaults(func=main)

def _write_png(tile, data, outputdir, palette):
    out = Image.fromarray(data, mode="P")
    out.putpalette(palette)

    os.makedirs(os.path.join(outputdir, str(tile.z), str(tile.x)), exist_ok=True)
    path = os.path.join(outputdir, str(tile.z), str(tile.x), str(tile.y) + ".png")

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

    tile_file = os.path.join(outputdir, str(tile.z), str(tile.x), str(tile.y) + ".tif")

#    try:


    # write data to file
    with rio.open(tile_file, 'w', **profile) as dst:
        for band in range(0, bands):
            dst.write(data, band+1)


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
    else:
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
        net.load_state_dict(state['state_dict'], strict=False)
        net.to(device)
    except FileNotFoundError as f:
        print("{} checkpoint not found.".format(chkpt))


    net.eval()

    tile_ids_filter = None
    if args.tile_ids is not None:
        tile_ids_filter = pd.read_csv(args.tile_ids, names=['ids']).ids.values



    ## Construct torch Dataset, either from single directory (if args.tiles is given) or from config. Used --tile_ids argument
    ## to determine how to filter resulting tiles (e.g. to only run prediction on a test set)
    if args.tiles is not None:
        imagery_locs = [args.tiles]
        # use tiledir  provided
        if args.tiles.startswith('s3://'):
            allImageryDatasets = [S3SlippyMapTiles(args.tiles, mode='multibands', transform=None, aws_profile = args.aws_profile, ids = tile_ids_filter, buffered=args.buffer, buffered_overlap=args.buffer_overlap, tilesize=tile_size, bands=num_channels)]
        else:
            allImageryDatasets = [SlippyMapTiles(args.tiles, mode="multibands", transform = None)]
        # directory = BufferedSlippyMapDirectory(args.tiles, transform=transform, size=tile_size,re overlap=args.overlap)
    else: # use config to search for tiles
        fs = s3fs.S3FileSystem(session = boto3.Session(profile_name = config['dataset']['aws_profile']))
        p = pprint.PrettyPrinter()
        imagery_searchpath = config['dataset']['image_bucket']  + '/' +  config['dataset']['imagery_directory_regex']
        print("Searching for imagery...({})".format(imagery_searchpath))
        imagery_candidates = fs.ls(config['dataset']['image_bucket'])
        print("candidates:")
        p.pprint(imagery_candidates)
        imagery_locs = [c for c in imagery_candidates if match(imagery_searchpath, c)]
        print("result:")
        p.pprint(imagery_locs)

        allImageryDatasets = [
            S3SlippyMapTiles("s3://" +  loc, mode='multibands', transform=None, aws_profile=args.aws_profile, ids=tile_ids_filter)
            for loc in imagery_locs
        ]


    palette = make_palette(config["classes"][0]["color"])


    # don't track tensors with autograd during prediction
    with torch.no_grad():
        for dataset, imageloc in zip(allImageryDatasets, imagery_locs):
            print("Prediction: {}".format(imageloc))
            imageloc_path = imageloc.replace("/", ":") # to not recreate directory structure when saving
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=args.workers)
            for tiles, images in tqdm(loader, desc="Eval", unit="batch", ascii=True):
                tiles = list(zip(tiles[0], tiles[1], tiles[2]))
                images = images.to(device)
                outputs = net(images)


                for i, (tile, prob) in enumerate(zip(tiles, outputs)):
                    tile = Tile(tile[0].item(), tile[1].item(), tile[2].item())
                    savedir = args.preds

                    # manually compute segmentation mask class probabilities per pixel
                    image = (prob > args.threshold).cpu().numpy().astype(np.uint8)

                    if args.buffer:
                        image = allImageryDatasets[0].unbuffer(image)

                    image = image.squeeze()
                    
                    _write_png(tile, image, os.path.join(savedir, imageloc_path), palette)

                    if(args.create_tif):
                        _write_tif(tile, image, os.path.join(savedir, imageloc_path))
