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
from src.models.segmentation.FSB_BTS_UNet_ import FSB_BTS_UNet_
from src.models.multitask.Multi_BTS_UNet import Multi_BTS_UNet
from src.models.multitask.Multi_FSB_BTS_UNet import Multi_FSB_BTS_UNet
from src.models.multitask.ExtendedUNetPlusPlus import ExtendedUNetPlusPlus
from src.models.multitask.Multi_FSB_BTS_UNet_ import Multi_FSB_BTS_UNet_
from src.models.segmentation.FSB_BTS_UNet_bkp import FSB_BTS_UNet_bkp
from monai.networks.nets import UNet, VNet, SegResNet, AttentionUnet, UNETR, HighResNet, BasicUNetPlusPlus
from monai.losses import DiceLoss, DiceFocalLoss, GeneralizedDiceLoss, DiceCELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from src.utils.metrics import calculate_metrics_multiclass_segmentation
from src.utils.miscellany import count_pixels
from src.utils.miscellany import postprocess_semantic_segmentation


def load_pretrained_model(model: nn.Module, ckpt_path: str):
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Loaded checkpoint '{ckpt_path}'. Last epoch: {checkpoint['epoch']}")
    else:
        raise ValueError(f"\n\t-> No checkpoint found at '{ckpt_path}'")

    return model


def inference_segmentation(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, path: str, device: str = 'cpu'):
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


def inference_semantic_segmentation(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, path: str, device: str = 'cpu'):
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
        test_outputs_postprocessed = postprocess_semantic_segmentation(test_outputs)

        # getting predicted class
        counter = count_pixels(test_outputs)
        counter_2 = count_pixels(test_outputs_postprocessed)
        print(counter, counter_2)
        benign_pixels, malignant_pixels = counter.get(1, 0), counter.get(2, 0)
        if benign_pixels >= malignant_pixels:
            predicted_class = 'benign'
        else:
            predicted_class = 'malignant'

        # getting segmentation metrics
        metrics = calculate_metrics_multiclass_segmentation(test_masks, test_outputs_postprocessed, patient_id)
        metrics['class'] = label
        metrics['predicted_class'] = predicted_class
        results = results.append(metrics, ignore_index=True)

        # saving segmentation
        save_semantic_segmentation(seg=test_outputs, path=f"{path}/segs/{label}_{patient_id}_seg.png")
        save_semantic_segmentation(seg=test_outputs_postprocessed, path=f"{path}/segs/{label}_{patient_id}_seg_postprocessed.png")

    # applying mapping for classification
    mapping_class = {
        'benign': 0,
        'malignant': 1
    }
    results['numerical_class'] = results['class'].map(mapping_class)
    results['numerical_class_predicted'] = results['predicted_class'].map(mapping_class)

    results.to_csv(f'{path}/results.csv', index=False)

    return results


def inference_multitask(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, path: str, device: str = 'cpu'):
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
        # test_loss = loss_fn(test_outputs, test_masks)

        # converting tensors to numpy arrays
        test_masks = test_masks.detach().cpu().numpy()
        test_outputs = test_outputs.detach().cpu().numpy()

        # getting metrics
        metrics = calculate_metrics(test_masks, test_outputs, patient_id)
        metrics['class'] = label
        results = results.append(metrics, ignore_index=True)

        # saving segmentation
        save_segmentation(seg=test_outputs, path=f"{path}/segs/{label}_{patient_id}_seg.png")

    results.to_csv(f'{path}/results_segmentation.csv', index=False)

    # classification
    metrics = pd.DataFrame(columns=['patient_id', 'ground_truth', 'predicted_label'])
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
        # print(label.detach().cpu().numpy()[0])
        # print(test_outputs.detach().cpu().numpy()[0][0])
        ground_truth_label.append(label.detach().cpu().numpy()[0])
        predicted_label.append(test_outputs.detach().cpu().numpy()[0][0])

        # getting metrics
    metrics = pd.DataFrame({'patient_id': patients, 'ground_truth': ground_truth_label, 'predicted_label': predicted_label})

    metrics.to_csv(f'{path}/results_classification.csv', index=False)

    return results, metrics


def inference_classification(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, path: str, device: str = 'cpu'):
    metrics = pd.DataFrame(columns=['patient_id', 'ground_truth', 'predicted_label'])

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
        # print(label.detach().cpu().numpy()[0])
        # print(test_outputs.detach().cpu().numpy()[0][0])
        ground_truth_label.append(label.detach().cpu().numpy()[0])
        predicted_label.append(test_outputs.detach().cpu().numpy()[0][0])

        # getting metrics
    metrics = pd.DataFrame({'patient_id': patients, 'ground_truth': ground_truth_label, 'predicted_label': predicted_label})

    metrics.to_csv(f'{path}/results.csv', index=False)

    return metrics



def save_segmentation(seg: np.array, path: str):
    seg = seg[0, 0, :, :].astype(int)
    seg[seg > 0] = 255
    cv2.imwrite(path, seg)


def save_semantic_segmentation(seg: np.array, path: str):
    seg = seg[0, 0, :, :].astype(int)
    cv2.imwrite(path, seg)


def save_features_map(seg: np.array, path: str):
    seg = seg.detach().cpu().numpy()
    seg = seg[0, 0, :, :].astype(float)
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
    elif architecture == 'FSB_BTS_UNet':
        model = FSB_BTS_UNet(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'FSB_BTS_UNet_':
        model = FSB_BTS_UNet_(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'UNet':
        model = UNet(spatial_dims=2, in_channels=sequences, out_channels=regions,
                     channels=(width, 2*width, 4*width, 8*width), strides=(2, 2, 2))
    elif architecture == 'FSB_BTS_UNet_bkp':
        model = FSB_BTS_UNet_bkp(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == "AttentionUNet":
        model = AttentionUnet(spatial_dims=2, in_channels=sequences, out_channels=regions,
                              channels=(width, 2*width, 4*width, 8*width), strides=(2, 2, 2))
    elif architecture == "HighResNet":
        model = HighResNet(spatial_dims=2, in_channels=1, out_channels=1)
    elif architecture == "UNETR":
        model = UNETR(in_channels=1, out_channels=1, img_size=128, spatial_dims=2)
    elif architecture == "UNetPlusPlus":
        model = BasicUNetPlusPlus(in_channels=1, out_channels=1, spatial_dims=2, deep_supervision=deep_supervision)
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
    elif loss_function == "BCE":
        loss_function_criterion = torch.nn.BCEWithLogitsLoss()
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
        classes: int = 1,
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
    if architecture == 'Multi_BTSUNet':
        model = Multi_BTS_UNet(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'Multi_FSB_BTS_UNet':
        model = Multi_FSB_BTS_UNet(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'Multi_FSB_BTS_UNet_':
        model = Multi_FSB_BTS_UNet_(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == "ExtendedUNetPlusPlus":
        model = ExtendedUNetPlusPlus(in_channels=sequences, out_channels=regions, spatial_dims=2, deep_supervision=deep_supervision)
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