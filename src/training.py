
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceLoss
from torch import logical_and as l_and, logical_not as l_not
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.dataset.BUSI_dataloader import BUSI_dataloaders
from src.models.segmentation.BTS_UNet import BTSUNet
from src.utils.miscellany import init_log
from src.utils.miscellany import seed_everything
from src.utils.metrics import dice_score_from_tensor
from src.utils.models import inference
from src.utils.models import load_pretrained_model


def train_one_epoch(epoch_index, tb_writer):
    running_training_loss = 0.
    running_dice = 0.
    last_loss = 0.

    # Iterating over training loader
    for k, data in enumerate(training_loader):
        inputs, masks = data['image'].to(dev), data['mask'].to(dev)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        if type(outputs) == list:
            # if deep supervision
            loss = torch.sum(torch.stack([loss_fn(o, masks) / (n + 1) for n, o in enumerate(reversed(outputs))]))
        else:
            loss = loss_fn(outputs, masks)
        if not torch.isnan(loss):
            running_training_loss += loss.item()
            loss.backward()
        else:
            logging.info("NaN in model loss!!")

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        # if i % 10 == 0:
        #     last_loss = running_loss / 10  # loss per batch
        #     # logging.info('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(training_loader) + i + 1
        #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.

        # measuring DICE
        if type(outputs) == list:
            outputs = outputs[-1]
        # masks = masks.detach().cpu().numpy()
        # outputs = torch.sigmoid(outputs).detach().cpu().numpy() > .5
        outputs = torch.sigmoid(outputs) > .5
        dice = dice_score_from_tensor(masks, outputs)
        running_dice += dice

    return running_training_loss / (k + 1), running_dice / (k + 1)


seed_everything(1993)
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

transforms = torch.nn.Sequential(
    # transforms.RandomCrop(128, pad_if_needed=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=random.choice([30, 60, 90, 120]))
)

training_loader, validation_loader, test_loader = BUSI_dataloaders(seed=1993, batch_size=1, transforms=transforms,
                                                                   train_size=0.8, augmentations=True,
                                                                   normalization=None)
# model = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=(16, 32, 64, 128), strides=(2, 2, 2)).to(dev)
model = BTSUNet(sequences=2, regions=1, width=24, deep_supervision=True).to(dev)
# model = SegResNet(spatial_dims=2, init_filters=16, in_channels=2, out_channels=1).to(dev)
# model = VNet(spatial_dims=2, in_channels=2, out_channels=1).to(dev)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = DiceLoss(include_background=True, sigmoid=True, smooth_dr=1, smooth_nr=1, squared_pred=True)

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
Path(f"runs/{timestamp}/segs/").mkdir(parents=True, exist_ok=True)
# init log
# init_time = time.perf_counter()
init_log(log_name=f"./runs/{timestamp}/execution.log")
# logging.info(args)


writer = SummaryWriter('runs/{}'.format(timestamp))
epoch_number = 0
EPOCHS = 5000

best_validation_loss = 1_000_000.
patience = 0
max_patience = 100
for epoch in range(EPOCHS):

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_train_loss, avg_dice = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_validation_loss = 0.0
    running_validation_dice = 0.0
    for i, validation_data in enumerate(validation_loader):

        validation_images, validation_masks = validation_data['image'].to(dev), validation_data['mask'].to(dev)
        validation_outputs = model(validation_images)
        if type(validation_outputs) == list:
            validation_loss = torch.sum(torch.stack(
                [loss_fn(vo, validation_masks) / (n + 1) for n, vo in enumerate(reversed(validation_outputs))]))
        else:
            validation_loss = loss_fn(validation_outputs, validation_masks)
        running_validation_loss += validation_loss

        # measuring DICE
        if type(validation_outputs) == list:
            validation_outputs = validation_outputs[-1]
        validation_outputs = torch.sigmoid(validation_outputs) > .5
        dice = dice_score_from_tensor(validation_masks, validation_outputs)
        running_validation_dice += dice

    avg_validation_loss = running_validation_loss / (i+1)
    avg_validation_dice = running_validation_dice / (i+1)
    logging.info(f'EPOCH {epoch} --> '
                 f'|| Training loss {avg_train_loss:.4f} '
                 f'|| Validation loss {avg_validation_loss:.4f} '
                 f'|| Training DICE {avg_dice:.4f} '
                 f'|| Validation DICE  {avg_validation_dice:.4f} ||')
    # Log the running loss averaged per batch for both training and validation
    # writer.add_scalars('Training vs. Validation Loss',
    #                    { 'Training' : avg_train_loss, 'Validation' : avg_vloss},
    #                    epoch_number + 1)
    # writer.flush()

    # Track best performance, and save the model's state
    if avg_validation_loss < best_validation_loss:
        patience = 0
        best_validation_loss = avg_validation_loss
        # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_validation_loss
        }, f'runs/{timestamp}/model_{timestamp}')
    else:
        patience += 1

    # early stopping
    if patience > max_patience:
        logging.info(f"\nValidation loss did not improve over the last {patience} epochs. Stopping training")
        break

    epoch_number += 1

logging.info(f"\nTesting phase")

# model = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=(16, 32, 64, 128), strides=(2, 2, 2)).to(dev)
model = BTSUNet(sequences=2, regions=1, width=24, deep_supervision=True).to(dev)
model = load_pretrained_model(model, f'runs/{timestamp}/model_{timestamp}')
# model = pretrained_model(model, 'model_20230131_142233')
results = inference(model=model, test_loader=test_loader, path=f"runs/{timestamp}", device=dev)

logging.info(results)

logging.info(results.DICE.mean())
logging.info(results.DICE.median())
logging.info(results.DICE.max())
#
# logging.info(results[results['class'] != 'normal'].DICE.mean())
# logging.info(results[results['class'] != 'normal'].DICE.median())
# logging.info(results[results['class'] != 'normal'].DICE.max())

# logging.info(results[(results['class'] != 'normal') & (results.dice > 0)].dice.mean())
# logging.info(results[(results['class'] != 'normal') & (results.dice > 0)].dice.median())
# logging.info(results[(results['class'] != 'normal') & (results.dice > 0)].dice.max())
