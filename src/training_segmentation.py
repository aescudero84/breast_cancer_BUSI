import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from src.dataset.BUSI_dataloader import BUSI_dataloader
from src.utils.experiment_init import device_setup
from src.utils.experiment_init import load_experiments_artefacts
from src.utils.metrics import dice_score_from_tensor
from src.utils.miscellany import init_log
from src.utils.miscellany import load_config_file
from src.utils.miscellany import seed_everything
from src.utils.models import inference_binary_segmentation
from src.utils.models import load_pretrained_model


def train_one_epoch():
    running_training_loss = 0.
    running_dice = 0.

    # Iterating over training loader
    for k, data in enumerate(training_loader):
        inputs, masks = data['image'].to(dev), data['mask'].to(dev)

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        if type(outputs) == list:
            # if deep supervision
            loss = torch.sum(torch.stack([criterion(o, masks) / (n + 1) for n, o in enumerate(reversed(outputs))]))
        else:
            loss = criterion(outputs, masks)
        if not torch.isnan(loss):
            running_training_loss += loss.item()
        else:
            logging.info("NaN in model loss!!")

        # Performing backward step through scaler methodology
        loss.backward()
        optimizer.step()

        # measuring DICE
        if type(outputs) == list:
            outputs = outputs[-1]
        outputs = torch.sigmoid(outputs) > .5
        dice = dice_score_from_tensor(masks, outputs)
        running_dice += dice

        del loss
        del outputs

    return running_training_loss / training_loader.__len__(), running_dice / training_loader.__len__()


@torch.no_grad()
def validate_one_epoch():
    running_validation_loss = 0.0
    running_validation_dice = 0.0
    for i, validation_data in enumerate(validation_loader):

        validation_images, validation_masks = validation_data['image'].to(dev), validation_data['mask'].to(dev)
        validation_outputs = model(validation_images)
        if type(validation_outputs) == list:
            validation_loss = torch.sum(torch.stack(
                [criterion(vo, validation_masks) / (n + 1) for n, vo in enumerate(reversed(validation_outputs))]))
        else:
            validation_loss = criterion(validation_outputs, validation_masks)
        running_validation_loss += validation_loss

        # measuring DICE
        if type(validation_outputs) == list:
            validation_outputs = validation_outputs[-1]
        validation_outputs = torch.sigmoid(validation_outputs) > .5
        dice = dice_score_from_tensor(validation_masks, validation_outputs)
        running_validation_dice += dice

    avg_validation_loss = running_validation_loss / validation_loader.__len__()
    avg_validation_dice = running_validation_dice / validation_loader.__len__()

    return avg_validation_loss, avg_validation_dice


# initializing times
init_time = time.perf_counter()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# loading config file
config_model, config_opt, config_loss, config_training, config_data = load_config_file(path='./src/config.yaml')
if config_training['CV'] < 2:
    sys.exit("This code is prepared for receiving a CV greater than 1")

# initializing seed and gpu if possible
seed_everything(config_training['seed'], cuda_benchmark=config_training['cuda_benchmark'])
dev = device_setup()

# initializing folder structures and log
run_path = (f"runs/{timestamp}_{config_model['architecture']}_{config_model['width']}_batch_"
            f"{config_data['batch_size']}_{'_'.join(config_data['classes'])}")
Path(f"{run_path}").mkdir(parents=True, exist_ok=True)
init_log(log_name=f"./{run_path}/execution.log")
shutil.copyfile('./src/config.yaml', f'./{run_path}/config.yaml')

# initializing experiment's objects
n_augments = sum([v for k, v in config_data['augmentation'].items()])
transforms = torch.nn.Sequential(
    # transforms.RandomCrop(128, pad_if_needed=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomRotation(degrees=random.choice([30, 60, 90, 120]))
    transforms.RandomRotation(degrees=np.random.choice(range(0, 360))),
    # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.5, 1.5))

)

training_loader, validation_loader, test_loader = BUSI_dataloader(seed=config_training['seed'],
                                                                  batch_size=config_data['batch_size'],
                                                                  # transforms=config_data['transforms'],
                                                                  transforms=transforms,
                                                                  remove_outliers=config_data['remove_outliers'],
                                                                  train_size=config_data['train_size'],
                                                                  augmentations=config_data['augmentation'],
                                                                  normalization=None,
                                                                  classes=config_data['classes'],
                                                                  path_images=config_data['input_img'],
                                                                  oversampling=config_data['oversampling'])
model, optimizer, criterion, scheduler = load_experiments_artefacts(config_model, config_opt, config_loss,
                                                                    n_augments, run_path)
model = model.to(dev)

best_validation_loss = 1_000_000.
patience = 0
for epoch in range(config_training['epochs']):
    start_epoch_time = time.perf_counter()

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_train_loss, avg_dice = train_one_epoch()

    # We don't need gradients on to do reporting
    model.train(False)
    avg_validation_loss, avg_validation_dice = validate_one_epoch()

    # # Update the learning rate at the end of each epoch
    scheduler.step(avg_validation_loss)

    # Track best performance, and save the model's state
    if avg_validation_loss < best_validation_loss:
        patience = 0  # restarting patience
        best_validation_loss = avg_validation_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': 'scheduler',
            'val_loss': best_validation_loss
        }, f'runs/{timestamp}/model_{timestamp}')
    else:
        patience += 1

    # logging results of current epoch
    end_epoch_time = time.perf_counter()
    logging.info(f'EPOCH {epoch} --> '
                 f'|| Training loss {avg_train_loss:.4f} '
                 f'|| Validation loss {avg_validation_loss:.4f} '
                 f'|| Training DICE {avg_dice:.4f} '
                 f'|| Validation DICE  {avg_validation_dice:.4f} '
                 f'|| Patience: {patience} '
                 f'|| Epoch time: {end_epoch_time-start_epoch_time:.4f}')

    # early stopping
    if patience > config_training['max_patience']:
        logging.info(f"\nValidation loss did not improve over the last {patience} epochs. Stopping training")
        break

logging.info(f"\nTesting phase")
model = load_pretrained_model(model, f'runs/{timestamp}/model_{timestamp}')
results = inference_binary_segmentation(model=model, test_loader=test_loader, path=f"runs/{timestamp}", device=dev)

logging.info(results)
logging.info(results.mean())

end_time = time.perf_counter()
logging.info(f"Total time: {end_time - init_time:.2f}")
