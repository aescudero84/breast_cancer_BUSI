import warnings
warnings.filterwarnings('ignore')
import numpy as np
import logging
import os
import sys
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.utils.metrics import calculate_metrics
from pathlib import Path
from src.models.segmentation.BTS_UNet import BTSUNet
from src.models.segmentation.Test_UNet import TestUNet
from src.models.segmentation.BTS_HDS_UNet import BTS_HDS_UNet
from monai.networks.nets import UNet, VNet, SegResNet
from monai.losses import DiceLoss, DiceFocalLoss, GeneralizedDiceLoss, DiceCELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


def load_pretrained_model(model: nn.Module, ckpt_path: str):
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Loaded checkpoint '{ckpt_path}'. Last epoch: {checkpoint['epoch']}")
    else:
        raise ValueError(f"\n\t-> No checkpoint found at '{ckpt_path}'")

    return model


def inference(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, path: str, device: str = 'cpu'):
    results = pd.DataFrame(columns=['patient_id', 'Haussdorf distance', 'DICE', 'Sensitivity', 'Specificity',
                                    'Accuracy', 'Jaccard index', 'Precision', 'class'])

    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        label = test_data['label'][0]
        test_images = test_data['image'].to(device)
        test_masks = test_data['mask'].to(device)

        # generating segmentation
        test_outputs = model(test_images)
        if type(test_outputs) == list:
            test_outputs = test_outputs[-1]  # in case that deep supervision is being used we got the last output
        test_outputs = (torch.sigmoid(test_outputs) > .5).float()
        # test_loss = loss_fn(test_outputs, test_masks)

        # converting tensors to numpy arrays
        test_masks = test_masks.detach().cpu().numpy()
        test_outputs = test_outputs.detach().cpu().numpy()

        # getting metrics
        metrics = calculate_metrics(test_masks, test_outputs, patient_id)
        metrics['class'] = label
        results = results.append(metrics, ignore_index=True)

        #
        # logging.info(count_pixels(test_masks[0, 0, :, :].cpu().numpy()))
        # logging.info(count_pixels(test_outputs[0, 0, :, :].cpu().numpy()))
        # showing results
        # plt.imshow(test_images[0, 0, :, :].cpu().numpy(), cmap='gray')
        # plt.show()
        # plt.imshow(test_masks[0, 0, :, :], cmap='gray')
        # plt.show()
        # plt.imshow(test_outputs[0, 0, :, :], cmap='gray')
        # plt.show()

        # saving segmentation
        save_segmentation(seg=test_outputs, path=f"{path}/segs/{label}_{patient_id}_seg.png")

    results.to_csv(f'{path}/results.csv', index=False)

    return results


def save_segmentation(seg: np.array, path: str):
    seg = seg[0, 0, :, :].astype(int)
    seg[seg > 0] = 255
    cv2.imwrite(path, seg)


def count_parameters(model: torch.nn.Module) -> int:
    """
    This function counts the trainable parameters in a model.

    Params:
    *******
        - model (torch.nn.Module): Torch model

    Return:
    *******
        - Int: Number of parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_segmentation_model(
        architecture: str,
        sequences: int = 1,
        regions: int = 1,
        width: int = 48,
        save_folder: Path = None,
        deep_supervision: bool = False
) -> torch.nn.Module:
    """
    This function implement the architecture chosen.

    Params:
    *******
        - architecture: architecture chosen

    Return:
    *******
        - et_present: it is true if the segmentation possess the ET region
        - img_segmentation: stack of images of the regions
    """

    logging.info(f"Creating {architecture} model")
    logging.info(f"The model will be fed with {sequences} sequences")

    if architecture == 'BTSUNet':
        model = BTSUNet(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'BTS_HDS_UNet':
        model = BTS_HDS_UNet(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'UNet':
        model = UNet(spatial_dims=2, in_channels=sequences, out_channels=regions,
                     channels=(width, 2*width, 4*width, 8*width), strides=(2, 2, 2))
    elif architecture == 'VNet':
        model = VNet(spatial_dims=2, in_channels=sequences, out_channels=regions)
    elif architecture == 'SegResNet':
        model = SegResNet(spatial_dims=2, init_filters=width, in_channels=sequences, out_channels=regions)
    else:
        model = torch.nn.Module()
        assert "The model selected does not exist. " \
               "Please, chose some of the following architectures: 3DUNet, VNet, ResidualUNet, ShallowUNet, DeepUNet"

    # Saving the model scheme in a .txt file
    if save_folder is not None:
        model_file = save_folder / "model.txt"
        with model_file.open("w") as f:
            print(model, file=f)

    logging.info(model)
    logging.info(f"Total number of trainable parameters: {count_parameters(model)}")

    return model


def init_optimizer(model: torch.nn.Module, optimizer: str, learning_rate: float = 0.001) -> torch.optim:
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

    return optimizer


def init_loss_function(loss_function: str = "dice") -> torch.nn.Module:
    if loss_function == 'DICE':
        loss_function_criterion = DiceLoss(include_background=True, sigmoid=True, smooth_dr=1, smooth_nr=1,
                                           squared_pred=True)
    elif loss_function == "FocalDICE":
        loss_function_criterion = DiceFocalLoss(include_background=True, sigmoid=True, squared_pred=True)
    elif loss_function == "GeneralizedDICE":
        loss_function_criterion = GeneralizedDiceLoss(include_background=True, sigmoid=True)
    elif loss_function == "CrossentropyDICE":
        loss_function_criterion = DiceCELoss(include_background=True, sigmoid=True, squared_pred=True)
    elif loss_function == "Jaccard":
        loss_function_criterion = DiceLoss(include_background=True, sigmoid=True, jaccard=True, reduction="sum")
    else:
        print("Select a loss function allowed: ['DICE', 'FocalDICE', 'GeneralizedDICE', 'CrossentropyDICE', 'Jaccard']")
        sys.exit()

    return loss_function_criterion


def init_lr_scheduler(optimizer, scheduler: str = "dice", T_max=20, min_lr=1e-6, patience=10) -> torch.optim.lr_scheduler:
    if scheduler == 'plateau':
        sche = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, min_lr=min_lr, verbose=True)
    elif scheduler == "cosine":
        sche = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)
    else:
        print("Select a loss function allowed: ['DICE', 'FocalDICE', 'GeneralizedDICE', 'CrossentropyDICE', 'Jaccard']")
        sys.exit()

    return sche
