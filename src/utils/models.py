import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.utils.metrics import calculate_metrics


def load_pretrained_model(model: nn.Module, ckpt_path: str):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

def save_segmentation(seg, path):
    seg = seg[0, 0, :, :].astype(int)
    seg[seg > 0] = 255
    cv2.imwrite(path, seg)
