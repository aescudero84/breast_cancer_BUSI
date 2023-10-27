import os
import json
import torch
import random
import pprint
import logging
import argparse
import numpy as np
import pandas as pd
import glob
import yaml
from pprint import pformat


def load_config_file(path: str):
    """
    This function load a config file and return the different sections.


    Parameters
    ----------
    path: Path where the config file is stored

    """
    with open(path) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)
        logging.info(pformat(config))
    return config['model'], config['optimizer'], config['loss'], config['training'], config['data']


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


def seed_everything(seed: int, cuda_benchmark: bool = False):
    """
    This function initializes all the seeds

    Params:
    *******
        - seed: seed number
        - cuda_benchmark: flag to activate/deactivate CUDA optimization algorithms

    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cuda_benchmark


def save_segmentation_results(path):
    results = []
    for n, f in enumerate(sorted(glob.glob(path + "/fold*/results.csv"))):
        df = pd.read_csv(f)
        df["fold"] = n
        results.append(df)

    df = pd.concat(results)
    df_grouped = df.drop(columns="patient_id").groupby('fold').mean().reset_index().drop(columns='fold').T
    df_grouped.columns = [f"fold {c}" for c in df_grouped.columns]
    df_grouped["mean"] = df_grouped.mean(axis=1)
    df_grouped["std"] = df_grouped.std(axis=1)
    df_grouped["latex"] = (round(df_grouped["mean"], 3).astype(str).str.ljust(5, '0') + " $pm$ " +
                           round(df_grouped["std"], 3).astype(str).str.ljust(5, '0'))
    df_grouped.to_excel(path + '/segmentation_results.xlsx', index=False)
