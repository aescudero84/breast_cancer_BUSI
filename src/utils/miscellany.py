import os
import json
import torch
import random
import pprint
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Tuple
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score
from SimpleITK import GetImageFromArray, GetArrayFromImage, WriteImage, ReadImage
# from ..utils.metrics import METRICS
# from ..utils.metrics import calculate_metrics
# from ..dataset.BraTS_dataset import recover_initial_resolution
import matplotlib.pyplot as plt
import seaborn as sns


def save_args(args: argparse.Namespace):
    """
    This function saves parsed arguments into config file.


    Parameters
    ----------
    args (dict{arg:value}): Arguments for this run

    """

    config = vars(args).copy()
    del config['save_folder'], config['seg_folder']

    logging.info(f"Execution for configuration:")
    pprint.pprint(config)

    config_file = args.save_folder / "config_file.json"
    with config_file.open("w") as file:
        json.dump(config, file, indent=4)


def init_log(log_name: str):
    """
    This function initializes a log file.

    Params:
    *******
        - log_name (str): log name

    """
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] - [%(levelname)s] - [%(filename)s:%(lineno)s] --- %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_name,
        filemode='a',
        force=True
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


def seed_everything(seed: int):

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def count_pixels(segmentation):

    unique, counts = np.unique(segmentation, return_counts=True)
    pixels_dict = dict(zip(unique, counts))

    if 4.0 not in pixels_dict:
        pixels_dict[4.0] = 0

    return pixels_dict
