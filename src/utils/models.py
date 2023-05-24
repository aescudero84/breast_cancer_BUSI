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
from src.models.classification.BTS_UNET_classifier import BTSUNetClassifier
from src.models.segmentation.FSB_BTS_UNet import FSB_BTS_UNet
from src.models.multitask.Multi_BTS_UNet import Multi_BTS_UNet
from src.models.multitask.Multi_FSB_BTS_UNet import Multi_FSB_BTS_UNet
from monai.networks.nets import UNet, AttentionUnet
from monai.losses import DiceLoss, DiceFocalLoss, GeneralizedDiceLoss, DiceCELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from src.utils.metrics import calculate_metrics_multiclass_segmentation
from src.utils.miscellany import count_pixels
from src.utils.miscellany import postprocess_semantic_segmentation


def load_pretrained_model(model: nn.Module, ckpt_path: str):
    """
    It restores a pretrained state model

    :param model: PyTorch module to be used
    :param ckpt_path: Path to the checkpoint

    :return: Model with a state loaded
    """
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Loaded checkpoint '{ckpt_path}'. Last epoch: {checkpoint['epoch']}")
    else:
        raise ValueError(f"\n\t-> No checkpoint found at '{ckpt_path}'")

    return model


def inference_binary_segmentation(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu'
):
    """
    It performs binary inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled as 0 or 1.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU

    :return: CSV file containing the main metrics
    """

    results = pd.DataFrame(columns=['patient_id', 'Haussdorf distance', 'DICE', 'Sensitivity', 'Specificity',
                                    'Accuracy', 'Jaccard index', 'Precision', 'class'])

    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        label = test_data['class'][0]
        test_images = test_data['image'].to(device)
        test_masks = test_data['mask'].to(device)

        # generating segmentation
        features_map = model(test_images)
        if type(features_map) == list:
            for n, ds in enumerate(reversed(features_map)):
                save_features_map(seg=ds, path=f"{path}/features_map/{label}_{patient_id}_ds_{n}.png")
            features_map = features_map[-1]  # in case that deep supervision is being used we got the last output
        else:
            save_features_map(seg=features_map, path=f"{path}/features_map/{label}_{patient_id}_seg.png")
        test_outputs = (torch.sigmoid(features_map) > .5).float()

        # converting tensors to numpy arrays
        test_masks = test_masks.detach().cpu().numpy()
        test_outputs = test_outputs.detach().cpu().numpy()

        # getting metrics
        metrics = calculate_metrics(test_masks, test_outputs, patient_id)
        metrics['class'] = label
        results = results.append(metrics, ignore_index=True)

        # saving segmentation
        save_binary_segmentation(seg=test_outputs, path=f"{path}/segs/{label}_{patient_id}_seg.png")

    # saving metrics results
    results.to_csv(f'{path}/results.csv', index=False)

    return results


def inference_multilabel_segmentation(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu',
        postprocessing: bool = False
):
    """
    It performs multilabel inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled from 0 to the number of classes.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU
    :param postprocessing: boolean to decide whether labelling all the pixels as the majority class

    :return: CSV file containing the main metrics
    """

    results = pd.DataFrame(columns=['patient_id', 'Haussdorf distance', 'DICE', 'Sensitivity', 'Specificity',
                                    'Accuracy', 'Jaccard index', 'Precision', 'class', 'predicted_class'])

    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        label = test_data['class'][0]
        test_images = test_data['image'].to(device)
        test_masks = test_data['mask'].to(device)

        # generating segmentation
        features_map = model(test_images)
        if type(features_map) == list:
            for n, ds in enumerate(reversed(features_map)):
                save_features_map(seg=ds, path=f"{path}/features_map/{label}_{patient_id}_ds_{n}.png")
            features_map = features_map[-1]  # in case that deep supervision is being used we got the last output
        else:
            save_features_map(seg=features_map, path=f"{path}/features_map/{label}_{patient_id}_seg.png")
        test_outputs = torch.nn.functional.softmax(features_map)

        # converting tensors to numpy arrays
        test_masks = torch.argmax(test_masks, dim=1, keepdim=True).float().detach().cpu().numpy()
        test_outputs = torch.argmax(test_outputs, dim=1, keepdim=True).float().detach().cpu().numpy()
        if postprocessing:
            test_outputs_postprocessed = postprocess_semantic_segmentation(test_outputs)

        # getting predicted class
        counter = count_pixels(test_outputs)
        benign_pixels, malignant_pixels = counter.get(1, 0), counter.get(2, 0)
        if benign_pixels >= malignant_pixels:
            predicted_class = 'benign'
        else:
            predicted_class = 'malignant'

        # getting segmentation metrics
        if postprocessing:
            metrics = calculate_metrics_multiclass_segmentation(test_masks, test_outputs_postprocessed, patient_id)
        else:
            metrics = calculate_metrics_multiclass_segmentation(test_masks, test_outputs, patient_id)
        metrics['class'] = label
        metrics['predicted_class'] = predicted_class
        results = results.append(metrics, ignore_index=True)

        # saving segmentation
        save_multilabel_segmentation(seg=test_outputs, path=f"{path}/segs/{label}_{patient_id}_seg.png")
        if postprocessing:
            save_multilabel_segmentation(seg=test_outputs_postprocessed,
                                         path=f"{path}/segs/{label}_{patient_id}_seg_postprocessed.png")

    # applying mapping for classification
    mapping_class = {
        'benign': 0,
        'malignant': 1
    }
    results['numerical_class'] = results['class'].map(mapping_class)
    results['numerical_class_predicted'] = results['predicted_class'].map(mapping_class)

    results.to_csv(f'{path}/results.csv', index=False)

    return results


def inference_multitask(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu'
):
    """
    It performs multitask inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled as 0 or 1 as well as the image will be classified as benign or malignant.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU

    :return: CSV file containing the main metrics
    """

    results = pd.DataFrame(columns=['patient_id', 'Haussdorf distance', 'DICE', 'Sensitivity', 'Specificity',
                                    'Accuracy', 'Jaccard index', 'Precision', 'class'])

    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        label = test_data['class'][0]
        test_images = test_data['image'].to(device)
        test_masks = test_data['mask'].to(device)

        # generating segmentation
        pred_class, features_map = model(test_images)
        if type(features_map) == list:
            for n, ds in enumerate(reversed(features_map)):
                save_features_map(seg=ds, path=f"{path}/features_map/{label}_{patient_id}_ds_{n}.png")
            features_map = features_map[-1]  # in case that deep supervision is being used we got the last output
        else:
            save_features_map(seg=features_map, path=f"{path}/features_map/{label}_{patient_id}_seg.png")
        test_outputs = (torch.sigmoid(features_map) > .5).float()

        # converting tensors to numpy arrays
        test_masks = test_masks.detach().cpu().numpy()
        test_outputs = test_outputs.detach().cpu().numpy()

        # getting metrics
        metrics = calculate_metrics(test_masks, test_outputs, patient_id)
        metrics['class'] = label
        results = results.append(metrics, ignore_index=True)

        # saving segmentation
        save_binary_segmentation(seg=test_outputs, path=f"{path}/segs/{label}_{patient_id}_seg.png")

    results.to_csv(f'{path}/results_segmentation.csv', index=False)

    # classification
    patients = []
    ground_truth_label = []
    predicted_label = []
    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        label = test_data['label'][0]
        test_images = test_data['image'].to(device)

        # generating segmentation
        test_outputs, segs = model(test_images)
        test_outputs = (torch.sigmoid(test_outputs) > .5).double()

        # converting tensors to numpy arrays
        patients.append(patient_id)
        ground_truth_label.append(label.detach().cpu().numpy()[0])
        predicted_label.append(test_outputs.detach().cpu().numpy()[0][0])

    # getting metrics
    metrics = pd.DataFrame({
        'patient_id': patients,
        'ground_truth': ground_truth_label,
        'predicted_label': predicted_label
    })

    metrics.to_csv(f'{path}/results_classification.csv', index=False)

    return results, metrics


def inference_classification(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu'
):
    """
    It performs binary classification inference over PyTorch dataloader by means of a trained model. It means the image
    will be classified as benign or malignant.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU

    :return: CSV file containing the main metrics
    """

    patients = []
    ground_truth_label = []
    predicted_label = []
    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        label = test_data['label'][0]
        test_images = test_data['image'].to(device)

        # generating segmentation
        test_outputs = model(test_images)
        test_outputs = (torch.sigmoid(test_outputs) > .5).double()

        # converting tensors to numpy arrays
        patients.append(patient_id)
        ground_truth_label.append(label.detach().cpu().numpy()[0])
        predicted_label.append(test_outputs.detach().cpu().numpy()[0][0])

        # getting metrics
    metrics = pd.DataFrame({
        'patient_id': patients,
        'ground_truth': ground_truth_label,
        'predicted_label': predicted_label
    })

    metrics.to_csv(f'{path}/results.csv', index=False)

    return metrics


def save_binary_segmentation(seg: np.array, path: str, value_non_zero: int = 255):
    """
    It saves a NumPy array as a binary image

    :param seg: Image to be saved
    :param path: path to save the image
    :param value_non_zero: value to assign all non-zero values. Typically, it will be 255 or 1.
    """

    n_dims = len(seg.shape)

    assert n_dims <= 4, "Numpy array must have less than 5 dimensions to be able to be stored"

    if n_dims == 4:
        seg = seg[0, 0, :, :].astype(int)
    elif n_dims == 3:
        seg = seg[0, :, :].astype(int)

    seg[seg > 0] = value_non_zero
    cv2.imwrite(path, seg)


def save_multilabel_segmentation(seg: np.array, path: str):
    """
    It saves a NumPy array as multilabel image

    :param seg: Image to be saved
    :param path: path to save the image
    """

    n_dims = len(seg.shape)

    assert n_dims <= 4, "Numpy array must have less than 5 dimensions to be able to be stored"

    if n_dims == 4:
        seg = seg[0, 0, :, :].astype(int)
    elif n_dims == 3:
        seg = seg[0, :, :].astype(int)

    seg = seg[0, 0, :, :].astype(int)
    cv2.imwrite(path, seg)


def save_features_map(seg: np.array, path: str):
    seg = seg.detach().cpu().numpy()
    seg = seg[0, 0, :, :].astype(float)
    cv2.imwrite(path, seg)


def count_parameters(model: torch.nn.Module) -> int:
    """
    This function counts the trainable parameters of a model.

    :param model: Torch model
    :return: Number of parameters
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
    elif architecture == 'UNet':
        model = UNet(spatial_dims=2, in_channels=sequences, out_channels=regions,
                     channels=(width, 2*width, 4*width, 8*width), strides=(2, 2, 2))
    elif architecture == "AttentionUNet":
        model = AttentionUnet(spatial_dims=2, in_channels=sequences, out_channels=regions,
                              channels=(width, 2*width, 4*width, 8*width), strides=(2, 2, 2))
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
        width: int = 48,
        save_folder: Path = None,
        deep_supervision: bool = False
) -> torch.nn.Module:
    """

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
    if architecture == 'Multi_BTSUNet':
        model = Multi_BTS_UNet(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'Multi_FSB_BTS_UNet':
        model = Multi_FSB_BTS_UNet(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
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
    elif loss_function == "BCE":
        loss_function_criterion = torch.nn.BCEWithLogitsLoss()
    else:
        print("Select a loss function allowed: ['DICE', 'FocalDICE', 'GeneralizedDICE', 'CrossentropyDICE', 'Jaccard']")
        sys.exit()

    return loss_function_criterion


def init_lr_scheduler(
        optimizer,
        scheduler: str = "dice",
        t_max: int = 20,
        min_lr: float = 1e-6,
        patience: int = 10
) -> torch.optim.lr_scheduler:

    if scheduler == 'plateau':
        sche = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, min_lr=min_lr, verbose=True)
    elif scheduler == "cosine":
        sche = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
    else:
        print("Select a loss function allowed: ['DICE', 'FocalDICE', 'GeneralizedDICE', 'CrossentropyDICE', 'Jaccard']")
        sys.exit()

    return sche
