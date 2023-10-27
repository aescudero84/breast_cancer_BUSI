import warnings
warnings.filterwarnings('ignore')
import logging
import sys
import torch
from pathlib import Path
from src.models.segmentation.BTS_UNet import BTSUNet
from src.models.segmentation.AAU_Net import AAUNet
from src.models.classification.BTS_UNET_classifier import BTSUNetClassifier
from src.models.segmentation.FSB_BTS_UNet import FSB_BTS_UNet
from src.models.segmentation.FSB_BTS_UNet_ import FSB_BTS_UNet_
from src.models.segmentation.FSB_BTS_UNet__ import FSB_BTS_UNet__
from src.models.segmentation.FSB_BTS_UNet___ import FSB_BTS_UNet___
from src.models.segmentation.FSB_BTS_UNet_a import FSB_BTS_UNet_a
from src.models.segmentation.TransUNet import TransUNet
from src.models.multitask.Multi_BTS_UNet import Multi_BTS_UNet
from src.models.multitask.Multi_FSB_BTS_UNet import Multi_FSB_BTS_UNet
from src.models.multitask.Multi_FSB_BTS_UNet__ import Multi_FSB_BTS_UNet_
from src.models.multitask.Multi_FSB_BTS_UNet_v2 import Multi_FSB_BTS_UNet_v2
from src.models.multitask.ExtendedUNetPlusPlus import ExtendedUNetPlusPlus
from src.models.segmentation.FSB_BTS_UNet_test1 import FSB_BTS_UNet_test1
from monai.networks.nets import UNet, AttentionUnet, BasicUnetPlusPlus
from monai.losses import DiceLoss, DiceFocalLoss, GeneralizedDiceLoss, DiceCELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from monai.networks.nets import SwinUNETR, SegResNet
from src.utils.models import count_parameters


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

    :param architecture: architecture chosen
    :param sequences: number of channels for the input layer
    :param regions: number of channels for the output layer
    :param width: number of channels to use in the first convolutional module
    :param deep_supervision: whether deep supervision is active
    :param save_folder: path to store the model
    :return: PyTorch module
    """

    logging.info(f"Creating {architecture} model")
    logging.info(f"The model will be fed with {sequences} sequences")

    if architecture == 'BTSUNet':
        model = BTSUNet(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'FSB_BTS_UNet':
        model = FSB_BTS_UNet(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'FSB_BTS_UNet_':
        model = FSB_BTS_UNet_(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'FSB_BTS_UNet__':
        model = FSB_BTS_UNet__(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'FSB_BTS_UNet___':
        model = FSB_BTS_UNet___(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'FSB_BTS_UNet_a':
        model = FSB_BTS_UNet_a(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'UNet':
        model = UNet(spatial_dims=2, in_channels=sequences, out_channels=regions,
                     channels=(width, 2*width, 4*width, 8*width), strides=(2, 2, 2))
    elif architecture == "AttentionUNet":
        model = AttentionUnet(spatial_dims=2, in_channels=sequences, out_channels=regions,
                              channels=(width, 2*width, 4*width, 8*width), strides=(2, 2, 2))
    elif architecture == "AAUnet":
        model = AAUNet(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == "BasicUnetPlusPlus":
        model = BasicUnetPlusPlus(spatial_dims=2, in_channels=sequences, out_channels=regions,
                                  deep_supervision=deep_supervision)
    elif architecture == "FSB_BTS_UNet_test1":
        model = FSB_BTS_UNet_test1(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == "SwinUNETR":
        model = SwinUNETR(img_size=(128, 128), in_channels=1, out_channels=1, use_checkpoint=True, spatial_dims=2)
    elif architecture == "TransUNet":
        model = TransUNet(img_dim=128, in_channels=sequences, class_num=regions, out_channels=128, head_num=4,
                          mlp_dim=256, block_num=8, patch_dim=4)
    elif architecture == "SegResNet":
        model = SegResNet(spatial_dims=2, init_filters=width, in_channels=sequences, out_channels=1)
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


def init_classification_model(
        architecture: str,
        sequences: int = 1,
        classes: int = 1,
        width: int = 48,
        save_folder: Path = None,
        deep_supervision: bool = False
) -> torch.nn.Module:
    """
    This function implement the architecture chosen.

    :param architecture: architecture chosen
    :param sequences: number of channels for the input layer
    :param classes: number of channels for the output layer
    :param width: number of channels to use in the first convolutional module
    :param deep_supervision: whether deep supervision is active
    :param save_folder: path to store the model
    :return: PyTorch module
    """

    logging.info(f"Creating {architecture} model")
    logging.info(f"The model will be fed with {sequences} sequences")

    if architecture == 'BTSUNetClassifier':
        model = BTSUNetClassifier(sequences=sequences, classes=classes, width=width, deep_supervision=deep_supervision)
    elif architecture == 'EfficientNet':
        from monai.networks.nets import EfficientNetBN
        model = EfficientNetBN("efficientnet-b0", pretrained=True, progress=True, spatial_dims=2,
                               in_channels=1, num_classes=1, norm=('batch', {'eps': 0.001, 'momentum': 0.01}),
                               adv_prop=False)
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


def init_multitask_model(
        architecture: str,
        sequences: int = 1,
        regions: int = 1,
        n_classes: int = 2,
        width: int = 48,
        save_folder: Path = None,
        deep_supervision: bool = False
) -> torch.nn.Module:
    """

    :param architecture: architecture chosen
    :param sequences: number of channels for the input layer
    :param regions: number of channels for the output layer
    :param n_classes: number of classes to predict
    :param width: number of channels to use in the first convolutional module
    :param deep_supervision: whether deep supervision is active
    :param save_folder: path to store the model
    :return: PyTorch module
    """

    logging.info(f"Creating {architecture} model")
    logging.info(f"The model will be fed with {sequences} sequences")
    if architecture == 'Multi_BTSUNet':
        model = Multi_BTS_UNet(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'Multi_FSB_BTS_UNet':
        model = Multi_FSB_BTS_UNet(sequences=sequences, regions=regions, n_classes=n_classes, width=width, deep_supervision=deep_supervision)
    elif architecture == 'Multi_FSB_BTS_UNet_':
        model = Multi_FSB_BTS_UNet_(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'Multi_FSB_BTS_UNet_v2':
        model = Multi_FSB_BTS_UNet_v2(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'ExtendedUNetPlusPlus':
        model = ExtendedUNetPlusPlus(in_channels=sequences, out_channels=regions, deep_supervision=deep_supervision)
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


def init_criterion_segmentation(loss_function: str = "dice") -> torch.nn.Module:
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
    elif loss_function == "BCE":
        loss_function_criterion = torch.nn.BCEWithLogitsLoss()
    else:
        print("Select a loss function allowed: ['DICE', 'FocalDICE', 'GeneralizedDICE', 'CrossentropyDICE', 'Jaccard']")
        sys.exit()

    return loss_function_criterion


def init_criterion_classification(n_classes: int = 2, classes_weighted=None) -> torch.nn.Module:
    if n_classes == 2:
        loss_function_criterion = torch.nn.BCEWithLogitsLoss()
    else:
        if classes_weighted:
            class_frequencies = torch.tensor(classes_weighted)

            # Calculate class weights
            class_weights = 1.0 / class_frequencies

            # Create a tensor for the Normalize weights
            weight_tensor = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float)

            # Define the loss function with class weights
            loss_function_criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor.to("cuda"))
        else:
            loss_function_criterion = torch.nn.CrossEntropyLoss()

    return loss_function_criterion


def init_lr_scheduler(
        optimizer,
        scheduler: str = "cosine",
        t_max: int = 20,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        patience: int = 20
) -> torch.optim.lr_scheduler:

    if scheduler == 'plateau':
        sche = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr, verbose=True)
    elif scheduler == "cosine":
        sche = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
    else:
        print("Select a loss function allowed: ['DICE', 'FocalDICE', 'GeneralizedDICE', 'CrossentropyDICE', 'Jaccard']")
        sys.exit()

    return sche
