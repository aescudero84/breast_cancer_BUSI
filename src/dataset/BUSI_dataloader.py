import logging as log
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


def BUSI_dataloader(seed, batch_size, transforms, augmentations=None, normalization=None, train_size=0.8,
                    classes=None, path_images="./Datasets/Dataset_BUSI_with_GT_postprocessed/"):

    # classes to use by default
    if classes is None:
        classes = ['benign', 'malignant']

    # Checking if the path, where the images are, exists
    path_images = Path(path_images).resolve()
    assert path_images.exists(), f"Path '{path_images}' it doesn't exist"
    log.info(f"Images are contained in the following path: {path_images}")

    # loading mapping file
    mapping = pd.read_csv(f"{path_images}/mapping_128.csv")

    # filtering specific classes
    mapping = mapping[mapping['class'].isin(classes)]

    # Splitting the mapping dataset into train_mapping, val_mapping and test_mapping
    train_mapping, val_mapping_ = train_test_split(mapping, train_size=train_size, random_state=int(seed), shuffle=True,
                                                   stratify=mapping['class'])
    val_mapping, test_mapping = train_test_split(val_mapping_, test_size=0.5, random_state=int(seed), shuffle=True,
                                                 stratify=val_mapping_['class'])

    print(train_mapping)
    print(val_mapping)
    print(test_mapping)
    # Creating the train-validation-test datasets
    train_dataset = BUSI(mapping_file=train_mapping, transforms=transforms, augmentations=augmentations, normalization=normalization)
    val_dataset = BUSI(mapping_file=val_mapping, transforms=None, augmentations=augmentations, normalization=normalization)
    test_dataset = BUSI(mapping_file=test_mapping, transforms=None, augmentations=augmentations, normalization=normalization)

    print(f"Size of train dataset: {train_dataset.__len__()}")
    print(f"Shape of images used for training: {train_dataset.__getitem__(0)['image'].shape}")
    print(f"Size of validation dataset: {val_dataset.__len__()}")
    print(f"Shape of images used for validating: {val_dataset.__getitem__(0)['image'].shape}")
    print(f"Size of test dataset: {test_dataset.__len__()}")
    print(f"Shape of images used for testing: {test_dataset.__getitem__(0)['image'].shape}")

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def BUSI_dataloader_CV(seed, batch_size, transforms, augmentations=None, normalization=None, train_size=0.8,
                       classes=None, n_folds=5, path_images="./Datasets/Dataset_BUSI_with_GT_postprocessed/"):

    # classes to use by default
    if classes is None:
        classes = ['benign', 'malignant']

    # Checking if the path, where the images are, exists
    path_images = Path(path_images).resolve()
    assert path_images.exists(), f"Path '{path_images}' it doesn't exist"
    log.info(f"Images are contained in the following path: {path_images}")

    # loading mapping file
    mapping = pd.read_csv(f"{path_images}/mapping_128.csv")

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

        print(train_mapping)
        print(val_mapping)
        print(test_mapping)

        # append the corresponding subset to train-val-test sets for each CV
        fold_trainset.append(BUSI(mapping_file=train_mapping, transforms=transforms, augmentations=augmentations, normalization=normalization))
        fold_valset.append(BUSI(mapping_file=val_mapping, transforms=transforms, augmentations=augmentations, normalization=normalization))
        fold_testset.append(BUSI(mapping_file=test_mapping, transforms=transforms, augmentations=augmentations, normalization=normalization))

    # Creating a list of dataloaders. Each component of the list corresponds to a CV fold
    train_loader = [DataLoader(fold, batch_size=batch_size, shuffle=True) for fold in fold_trainset]
    val_loader = [DataLoader(fold, batch_size=1, shuffle=True) for fold in fold_valset]
    test_loader = [DataLoader(fold, batch_size=1) for fold in fold_testset]

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    from time import perf_counter
    tic = perf_counter()
    # transforms = torch.nn.Sequential(
    #     # transforms.RandomCrop(500, pad_if_needed=True),
    #     transforms.Resize((256, 256)),
    # )

    # a, b, c = BUSI_dataloader(seed=1, batch_size=1, transforms=None)
    a, b, c = BUSI_dataloader_CV(seed=1, batch_size=1, transforms=None)

    toc = perf_counter()
    print(toc-tic)