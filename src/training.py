import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat
import random
import numpy as np
import torch
import yaml
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from src.dataset.BUSI_dataloader import BUSI_dataloader
from src.utils.metrics import dice_score_from_tensor
from src.utils.miscellany import init_log
from src.utils.miscellany import seed_everything
from src.utils.models import inference
from src.utils.models import init_loss_function
from src.utils.models import init_optimizer
from src.utils.models import init_segmentation_model
from src.utils.models import load_pretrained_model


def train_one_epoch():
    running_training_loss = 0.
    running_dice = 0.

    # Iterating over training loader
    for k, data in enumerate(training_loader):
        inputs, masks = data['image'].to(dev), data['mask'].to(dev)

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)

        # enable auto-casting
        # with autocast(enabled=False):
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
        else:
            logging.info("NaN in model loss!!")

        # Performing backward step through scaler methodology
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
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


def validate_one_epoch():
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

    avg_validation_loss = running_validation_loss / validation_loader.__len__()
    avg_validation_dice = running_validation_dice / validation_loader.__len__()

    return avg_validation_loss, avg_validation_dice


# start time
init_time = time.perf_counter()
# initializing folder structures and log
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
Path(f"runs/{timestamp}/segs/").mkdir(parents=True, exist_ok=True)
init_log(log_name=f"./runs/{timestamp}/execution.log")

# loading config file
with open('./src/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    logging.info(pformat(config))
shutil.copyfile('./src/config.yaml', f'./runs/{timestamp}/config.yaml')

config_model = config['model']
config_opt = config['optimizer']
config_loss = config['loss']
config_training = config['training']
config_data = config['data']

# initializing seed and gpu if possible
seed_everything(config_training['seed'])
if torch.cuda.is_available():
    dev = "cuda:0"
    logging.info("GPU will be used to train the model")
else:
    dev = "cpu"
    logging.info("CPU will be used to train the model")

# initializing experiment's objects
n_augments = sum([v for k, v in config_data['augmentation'].items()])
model = init_segmentation_model(architecture=config_model['architecture'],
                                sequences=config_model['sequences'] + n_augments,
                                width=config_model['width'],
                                deep_supervision=config_model['deep_supervision'],
                                save_folder=Path(f'./runs/{timestamp}/')).to(dev)
optimizer = init_optimizer(model=model, optimizer=config_opt['opt'], learning_rate=config_opt['lr'])
loss_fn = init_loss_function(loss_function=config_loss['function'])
# scaler = GradScaler()  # create a GradScaler object for scaling the gradients
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

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
                                                                  transforms=transforms,
                                                                  train_size=config_data['train_size'],
                                                                  augmentations=config_data['augmentation'],
                                                                  normalization=None,
                                                                  classes=config_data['classes'],
                                                                  path_images=config_data['input_img'])

best_validation_loss = 1_000_000.
patience = 0
for epoch in range(config_training['epochs']):
    start_epoch_time = time.perf_counter()

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_train_loss, avg_dice = train_one_epoch()

    # We don't need gradients on to do reporting
    model.train(False)
    with torch.no_grad():
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
results = inference(model=model, test_loader=test_loader, path=f"runs/{timestamp}", device=dev)

logging.info(results)

logging.info(results.mean())
logging.info(results.median())
logging.info(results.max())

end_time = time.perf_counter()
logging.info(f"Total time: {end_time - init_time:.2f}")
