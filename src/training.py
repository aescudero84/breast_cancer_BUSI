import logging
import random
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat
import shutil

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.dataset.BUSI_dataloader import BUSI_dataloaders
from src.models.segmentation.BTS_UNet import BTSUNet
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

        # measuring DICE
        if type(outputs) == list:
            outputs = outputs[-1]
        outputs = torch.sigmoid(outputs) > .5
        dice = dice_score_from_tensor(masks, outputs)
        running_dice += dice

    return running_training_loss / (k + 1), running_dice / (k + 1)

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

    avg_validation_loss = running_validation_loss / (i + 1)
    avg_validation_dice = running_validation_dice / (i + 1)

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
model = init_segmentation_model(architecture=config_model['architecture'], sequences=config_model['sequences'],
                                width=config_model['width'], deep_supervision=config_model['deep_supervision'],
                                save_folder=Path(f'./runs/{timestamp}/')).to(dev)
optimizer = init_optimizer(model=model, optimizer=config_opt['opt'], learning_rate=config_opt['lr'])
loss_fn = init_loss_function(loss_function=config_loss['function'])

transforms = torch.nn.Sequential(
    # transforms.RandomCrop(128, pad_if_needed=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomRotation(degrees=random.choice([30, 60, 90, 120]))
    transforms.RandomRotation(degrees=random.choice(range(0, 360)))
)

training_loader, validation_loader, test_loader = BUSI_dataloaders(seed=config_training['seed'],
                                                                   batch_size=config_data['batch_size'],
                                                                   transforms=transforms,
                                                                   train_size=config_data['train_size'],
                                                                   augmentations=True,
                                                                   normalization=None)


best_validation_loss = 1_000_000.
patience = 0
for epoch in range(config_training['epochs']):

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_train_loss, avg_dice = train_one_epoch()

    # We don't need gradients on to do reporting
    model.train(False)
    avg_validation_loss, avg_validation_dice = validate_one_epoch()

    # logging results of current epoch
    logging.info(f'EPOCH {epoch} --> '
                 f'|| Training loss {avg_train_loss:.4f} '
                 f'|| Validation loss {avg_validation_loss:.4f} '
                 f'|| Training DICE {avg_dice:.4f} '
                 f'|| Validation DICE  {avg_validation_dice:.4f} ||')

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

    # early stopping
    if patience > config_training['max_patience']:
        logging.info(f"\nValidation loss did not improve over the last {patience} epochs. Stopping training")
        break


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


end_time = time.perf_counter()
logging.info(f"Total time: {end_time-init_time:.2f}")
#
# logging.info(results[results['class'] != 'normal'].DICE.mean())
# logging.info(results[results['class'] != 'normal'].DICE.median())
# logging.info(results[results['class'] != 'normal'].DICE.max())

# logging.info(results[(results['class'] != 'normal') & (results.dice > 0)].dice.mean())
# logging.info(results[(results['class'] != 'normal') & (results.dice > 0)].dice.median())
# logging.info(results[(results['class'] != 'normal') & (results.dice > 0)].dice.max())
