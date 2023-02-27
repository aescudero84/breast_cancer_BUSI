from __future__ import print_function, division

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

def count_pixels(segmentation):
    import numpy as np
    unique, counts = np.unique(segmentation, return_counts=True)
    pixels_dict = dict(zip(unique, counts))

    return pixels_dict


def min_max_scaler(image: np.array) -> np.array:
    """ Min max scaler function."""

    min_, max_ = torch.min(image), torch.max(image)
    image = (image - min_) / (max_ - min_)

    return image


class BUSI(Dataset):
    """INSTANCE dataset."""

    def __init__(self, mapping_file: pd.DataFrame, transforms=None, augmentations={}, normalization=None):
        super(BUSI, self).__init__()
        """
        Args:
            mapping_file (string): Path to the mapping file
        """

        self.mapping_file = mapping_file
        self.transforms = transforms
        self.augmentations = True if sum([v for k, v in augmentations.items()]) else False
        if augmentations:
            self.CLAHE = augmentations.get("CLAHE", False)
            self.brightness_brighter = augmentations.get("brightness_brighter", False)
            self.brightness_darker = augmentations.get("brightness_darker", False)
            self.contrast_high = augmentations.get("contrast_high", False)
            self.contrast_low = augmentations.get("contrast_low", False)

        self.normalization = normalization

        self.data = []
        for index, row in self.mapping_file.iterrows():
            # loading image and mask
            image = cv2.imread(row['img_path'], 0)
            mask = cv2.imread(row['mask_path'], 0)
            mask[mask == 255] = 1

            # loading other features
            label = row['class']
            patient_id = row['id']
            dim1 = row['dim1']
            dim2 = row['dim2']
            tumor_pixels = row['tumor_pixels']

            # appending information in a list
            self.data.append({
                'patient_id': patient_id,
                'label': label,
                'image': image,
                'mask': mask,
                'dim1': dim1,
                'dim2': dim2,
                'tumor_pixels': tumor_pixels
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        patient_info = self.data[idx]

        # adding channel component
        image = torch.unsqueeze(torch.as_tensor(patient_info['image'], dtype=torch.float32), 0)
        mask = torch.unsqueeze(torch.as_tensor(patient_info['mask'], dtype=torch.float32), 0)

        if self.normalization is not None:
            image = min_max_scaler(image)

        # Augmentations
        if self.augmentations:
            aumengs = []

            if self.CLAHE:
                clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(4, 4))
                aumengs.append(torch.unsqueeze(torch.as_tensor(clahe.apply(patient_info['image']), dtype=torch.float32), 0))

            if self.brightness_brighter:  # brightness
                brightness_matrix = np.ones(patient_info['image'].shape, dtype='uint8') * 80
                img_brighter = cv2.add(patient_info['image'], brightness_matrix)
                aumengs.append(torch.unsqueeze(torch.as_tensor(img_brighter, dtype=torch.float32), 0))
            if self.brightness_darker:  # brightness
                brightness_matrix = np.ones(patient_info['image'].shape, dtype='uint8') * 80
                img_darker = cv2.subtract(patient_info['image'], brightness_matrix)
                aumengs.append(torch.unsqueeze(torch.as_tensor(img_darker, dtype=torch.float32), 0))

            if self.contrast_low:  # contrast
                matrix1 = np.ones(patient_info['image'].shape) * .02
                img_low_contrast = np.uint8(cv2.multiply(np.float64(patient_info['image']), matrix1))
                aumengs.append(torch.unsqueeze(torch.as_tensor(img_low_contrast, dtype=torch.float32), 0))
            if self.contrast_high:  # contrast
                matrix2 = np.ones(patient_info['image'].shape) * 1.5
                img_high_contrast = np.uint8(np.clip(cv2.multiply(np.float64(patient_info['image']), matrix2), 0, 255))
                aumengs.append(torch.unsqueeze(torch.as_tensor(img_high_contrast, dtype=torch.float32), 0))

        # apply transformations without augmentations
        if self.transforms is not None and not self.augmentations:
            joined = self.transforms(torch.cat([mask, image], dim=0))
            mask = torch.unsqueeze(joined[0, :, :], 0)
            image = torch.unsqueeze(joined[1, :, :], 0)

        # apply transformations with augmentations
        if self.transforms is not None and self.augmentations:
            joined = torch.cat([mask, image] + aumengs, dim=0)
            joined = self.transforms(joined)
            mask = torch.unsqueeze(joined[0, :, :], 0)
            image = joined[1:, :, :]

        if self.transforms is None and self.augmentations:
            image = torch.cat([image] + aumengs, dim=0)

        # if self.normalization is not None:
        #     # image = torch.cat([image, aug1], dim=0)
        #     image = min_max_scaler(image)

        return {
            'patient_id': patient_info['patient_id'],
            'label': patient_info['label'],
            'image': image,
            'mask': mask,
            'dim1': patient_info['dim1'],
            'dim2': patient_info['dim2'],
            'tumor_pixels': patient_info['tumor_pixels']
        }

# if '__main__' == __name__:
#
#     mapping = pd.read_csv("./Datasets/Dataset_BUSI_with_GT_postprocessed_128/mapping.csv")
#     transforms = torch.nn.Sequential(
#         transforms.RandomCrop(500, pad_if_needed=True),
#     )
#     # dataset = BUSI(mapping_file="./Datasets/Dataset_BUSI_with_GT_postprocessed_128/mapping.csv", transform=transforms)
#     dataset = BUSI(mapping_file=mapping, transform=None)
#
#     # for i in range(dataset.__len__()):
#     #     print(dataset.__getitem__(i)['mask'].max(), dataset.__getitem__(i)['mask'].min())
#
#     patient = dataset.__getitem__(601)
#
#     print(patient['image'].shape)
#
#     plt.imshow(patient['image'][0, :, :], cmap='gray')
#     plt.show()
#     plt.imshow(patient['mask'][0, :, :], cmap='gray')
#     plt.show()
