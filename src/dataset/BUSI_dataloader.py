import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset.BUSI_dataset import BUSI


def BUSI_dataloader(seed, batch_size, transforms, remove_outliers=False, augmentations=None, normalization=None,
                    train_size=0.8, classes=None, path_images="./Datasets/Dataset_BUSI_with_GT_postprocessed_128/",
                    oversampling=True, semantic_segmentation=False):

    # classes to use by default
    if classes is None:
        classes = ['benign', 'malignant']

    # Checking if the path, where the images are, exists
    path_images = Path(path_images).resolve()
    assert path_images.exists(), f"Path '{path_images}' it doesn't exist"
    logging.info(f"Images are contained in the following path: {path_images}")

    # loading mapping file
    mapping = pd.read_csv(f"{path_images}/mapping.csv")

    # filtering specific classes
    mapping = mapping[mapping['class'].isin(classes)]

    # Splitting the mapping dataset into train_mapping, val_mapping and test_mapping
    train_mapping, val_mapping_ = train_test_split(mapping, train_size=train_size, random_state=int(seed), shuffle=True,
                                                   stratify=mapping['class'])
    val_mapping, test_mapping = train_test_split(val_mapping_, test_size=0.5, random_state=int(seed), shuffle=True,
                                                 stratify=val_mapping_['class'])

    if remove_outliers:
        train_mapping = filter_anomalous_cases(train_mapping)
        val_mapping = filter_anomalous_cases(val_mapping)
        test_mapping = filter_anomalous_cases(test_mapping)

    if oversampling:
        train_mapping_malignant = train_mapping[train_mapping['class'] == 'malignant']
        train_mapping = pd.concat([train_mapping, train_mapping_malignant])

    # logging datasets
    logging.info(train_mapping)
    logging.info(val_mapping)
    logging.info(test_mapping)

    # Creating the train-validation-test datasets
    train_dataset = BUSI(mapping_file=train_mapping, transforms=transforms, augmentations=augmentations,
                         normalization=normalization, semantic_segmentation=semantic_segmentation)
    val_dataset = BUSI(mapping_file=val_mapping, transforms=None, augmentations=augmentations,
                       normalization=normalization, semantic_segmentation=semantic_segmentation)
    test_dataset = BUSI(mapping_file=test_mapping, transforms=None, augmentations=augmentations,
                        normalization=normalization, semantic_segmentation=semantic_segmentation)

    logging.info(f"Size of train dataset: {train_dataset.__len__()}")
    logging.info(f"Shape of images used for training: {train_dataset.__getitem__(0)['image'].shape}")
    logging.info(f"Size of validation dataset: {val_dataset.__len__()}")
    logging.info(f"Shape of images used for validating: {val_dataset.__getitem__(0)['image'].shape}")
    logging.info(f"Size of test dataset: {test_dataset.__len__()}")
    logging.info(f"Shape of images used for testing: {test_dataset.__getitem__(0)['image'].shape}")

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def BUSI_dataloader_CV(seed, batch_size, transforms, remove_outliers=False, augmentations=None, normalization=None,
                       train_size=0.8, classes=None, n_folds=5, oversampling=True,
                       path_images="./Datasets/Dataset_BUSI_with_GT_postprocessed_128/", semantic_segmentation=False):

    # classes to use by default
    if classes is None:
        classes = ['benign', 'malignant']

    # Checking if the path, where the images are, exists
    path_images = Path(path_images).resolve()
    assert path_images.exists(), f"Path '{path_images}' it doesn't exist"
    logging.info(f"Images are contained in the following path: {path_images}")

    # loading mapping file
    mapping = pd.read_csv(f"{path_images}/mapping.csv")

    # filtering specific classes
    mapping = mapping[mapping['class'].isin(classes)]

    # splitting dataset into train-val-test CV
    fold_trainset, fold_valset, fold_testset = [], [], []
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    for n, (train_ix, test_ix) in enumerate(kfold.split(mapping, mapping['class'])):
        train_val_mapping, test_mapping = mapping.iloc[train_ix], mapping.iloc[test_ix]
        test_mapping['fold'] = [n] * len(test_mapping)

        # Splitting the mapping dataset into train_mapping, val_mapping and test_mapping
        train_mapping, val_mapping = train_test_split(train_val_mapping, train_size=train_size, random_state=int(seed),
                                                      shuffle=True, stratify=train_val_mapping['class'])

        if remove_outliers:
            train_mapping = filter_anomalous_cases(train_mapping)
            val_mapping = filter_anomalous_cases(val_mapping)
            test_mapping = filter_anomalous_cases(test_mapping)

        if oversampling:
            train_mapping_malignant = train_mapping[train_mapping['class'] == 'malignant']
            train_mapping = pd.concat([train_mapping, train_mapping_malignant])

        logging.info(train_mapping)
        logging.info(val_mapping)
        logging.info(test_mapping)

        # append the corresponding subset to train-val-test sets for each CV
        fold_trainset.append(BUSI(mapping_file=train_mapping, transforms=transforms, augmentations=augmentations,
                                  normalization=normalization, semantic_segmentation=semantic_segmentation))
        fold_valset.append(BUSI(mapping_file=val_mapping, transforms=None, augmentations=augmentations,
                                normalization=normalization, semantic_segmentation=semantic_segmentation))
        fold_testset.append(BUSI(mapping_file=test_mapping, transforms=None, augmentations=augmentations,
                                 normalization=normalization, semantic_segmentation=semantic_segmentation))

    # Creating a list of dataloaders. Each component of the list corresponds to a CV fold
    train_loader = [DataLoader(fold, batch_size=batch_size, shuffle=True) for fold in fold_trainset]
    val_loader = [DataLoader(fold, batch_size=batch_size, shuffle=True) for fold in fold_valset]
    test_loader = [DataLoader(fold, batch_size=1) for fold in fold_testset]

    return train_loader, val_loader, test_loader


def BUSI_dataloader_CV_prod(seed, batch_size, transforms, remove_outliers=False, augmentations=None, normalization=None,
                       train_size=0.8, classes=None, n_folds=5, oversampling=True,
                       path_images="./Datasets/Dataset_BUSI_with_GT_postprocessed_128/", semantic_segmentation=False):

    # classes to use by default
    if classes is None:
        classes = ['benign', 'malignant']

    # Checking if the path, where the images are, exists
    path_images = Path(path_images).resolve()
    assert path_images.exists(), f"Path '{path_images}' it doesn't exist"
    logging.info(f"Images are contained in the following path: {path_images}")

    # loading mapping file
    mapping = pd.read_csv(f"{path_images}/mapping.csv")

    # filtering specific classes
    mapping = mapping[mapping['class'].isin(classes)]

    # splitting dataset into train-val-test CV
    fold_trainset, fold_valset, fold_testset = [], [], []
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    for n, (train_ix, test_ix) in enumerate(kfold.split(mapping, mapping['class'])):
        train_val_mapping, test_mapping = mapping.iloc[train_ix], mapping.iloc[test_ix]
        test_mapping['fold'] = [n] * len(test_mapping)

        # Splitting the mapping dataset into train_mapping, val_mapping and test_mapping
        train_mapping, val_mapping = train_test_split(train_val_mapping, train_size=train_size, random_state=int(seed),
                                                      shuffle=True, stratify=train_val_mapping['class'])

        if remove_outliers:
            train_mapping = filter_anomalous_cases(train_mapping)
            val_mapping = filter_anomalous_cases(val_mapping)
            test_mapping = filter_anomalous_cases(test_mapping)

        if oversampling:
            train_mapping_malignant = train_mapping[train_mapping['class'] == 'malignant']
            train_mapping = pd.concat([train_mapping, train_mapping_malignant])

        train_mapping = pd.concat([train_mapping, val_mapping])
        logging.info(train_mapping)
        logging.info(test_mapping)

        # append the corresponding subset to train-val-test sets for each CV
        fold_trainset.append(BUSI(mapping_file=train_mapping, transforms=transforms,
                                  augmentations=augmentations,
                                  normalization=normalization, semantic_segmentation=semantic_segmentation))
        fold_testset.append(BUSI(mapping_file=test_mapping, transforms=None, augmentations=augmentations,
                                 normalization=normalization, semantic_segmentation=semantic_segmentation))

    # Creating a list of dataloaders. Each component of the list corresponds to a CV fold
    train_loader = [DataLoader(fold, batch_size=batch_size, shuffle=True) for fold in fold_trainset]
    test_loader = [DataLoader(fold, batch_size=1) for fold in fold_testset]

    return train_loader, test_loader


def filter_anomalous_cases(mapping):
    logging.info("Filtering anomalous cases")
    anomalous_cases = {
        'benign': [435, 433, 42, 131, 437, 269, 333, 399, 403, 406, 85, 164, 61, 94, 108, 114, 116, 119, 122, 201, 302,
                   394, 402, 199, 248, 242, 288, 236, 247, 233, 299, 4, 321, 25, 153],
        'malignant': [145, 51, 77, 78, 93, 94, 52, 106, 107, 18, 116],
        'normal': [34, 1]
    }

    for cls, ids in anomalous_cases.items():
        mapping = mapping[~((mapping['class'] == cls) & (mapping['id'].isin(ids)))]

    return mapping


if __name__ == '__main__':
    from time import perf_counter
    tic = perf_counter()

    # a, b, c = BUSI_dataloader(seed=1, batch_size=1, transforms=None)
    a, b, c = BUSI_dataloader_CV(seed=1, batch_size=1, transforms=None)

    toc = perf_counter()
    print(toc-tic)