import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from src.dataset.BUSI_dataloader import BUSI_dataloader_CV_prod
from src.utils.metrics import dice_score_from_tensor
from src.utils.miscellany import init_log
from src.utils.miscellany import seed_everything
from src.utils.models import inference_binary_segmentation
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

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        if type(outputs) == list:
            # if deep supervision
            loss = torch.sum(torch.stack([loss_fn(o, masks) / (j + 1) for j, o in enumerate(reversed(outputs))]))
        else:
            loss = loss_fn(outputs, masks)
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


# initializing folder structures and log
start_time = time.perf_counter()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# loading config file
with open('./src/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config_model = config['model']
config_opt = config['optimizer']
config_loss = config['loss']
config_training = config['training']
config_data = config['data']

run_path = f"{timestamp}_{config_model['architecture']}_{config_model['width']}_batch_{config_data['batch_size']}_" \
           f"{'_'.join(config_data['classes'])}_prod"
Path(f"runs/{run_path}/").mkdir(parents=True, exist_ok=True)
logging.info(pformat(config))
shutil.copyfile('./src/config.yaml', f'./runs/{run_path}/config.yaml')


# initializing seed and gpu if possible
seed_everything(config_training['seed'], cuda_benchmark=config_training['cuda_benchmark'])
if torch.cuda.is_available():
    dev = "cuda:0"
    logging.info("GPU will be used to train the model")
else:
    dev = "cpu"
    logging.info("CPU will be used to train the model")


n_augments = sum([v for k, v in config_data['augmentation'].items()])
transforms = torch.nn.Sequential(
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=np.random.choice(range(0, 360)))
)

if config_training['CV'] > 1:
    train_loaders, test_loaders = BUSI_dataloader_CV_prod(seed=config_training['seed'],
                                                          batch_size=config_data['batch_size'],
                                                          transforms=transforms,
                                                          remove_outliers=config_data['remove_outliers'],
                                                          train_size=config_data['train_size'],
                                                          n_folds=config_training['CV'],
                                                          augmentations=config_data['augmentation'],
                                                          normalization=None,
                                                          classes=config_data['classes'],
                                                          oversampling=config_data['oversampling'],
                                                          path_images=config_data['input_img'])
else:
    sys.exit("This code is prepared for receiving a CV greater than 1")

for n, (training_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):

    # creating specific paths and experiment's objects for each fold
    init_time = time.perf_counter()
    Path(f"runs/{run_path}/fold_{n}/segs/").mkdir(parents=True, exist_ok=True)
    init_log(log_name=f"./runs/{run_path}/fold_{n}/execution_fold_{n}.log")
    model = init_segmentation_model(architecture=config_model['architecture'],
                                    sequences=config_model['sequences'] + n_augments,
                                    width=config_model['width'], deep_supervision=config_model['deep_supervision'],
                                    save_folder=Path(f'./runs/{run_path}/')).to(dev)
    optimizer = init_optimizer(model=model, optimizer=config_opt['opt'], learning_rate=config_opt['lr'])
    loss_fn = init_loss_function(loss_function=config_loss['function'])
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    logging.info(f"\n\n *********************  FOLD {n}  ********************* \n\n")

    for epoch in range(config_training['epochs']):
        start_epoch_time = time.perf_counter()

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_train_loss, avg_dice = train_one_epoch()

        # Update the learning rate at the end of each epoch
        scheduler.step(avg_train_loss)
        # best_validation_loss = avg_train_loss

        # # Iterating over test loader
        # running_test_dice = 0.
        # for k, data in enumerate(test_loader):
        #     inputs, masks = data['image'].to(dev), data['mask'].to(dev)
        #     outputs = model(inputs)
        #     # measuring DICE
        #     if type(outputs) == list:
        #         outputs = outputs[-1]
        #     outputs = torch.sigmoid(outputs) > .5
        #     dice = dice_score_from_tensor(masks, outputs)
        #     running_test_dice += dice
        # avg_test_dice = running_test_dice / test_loader.__len__()

        # logging results of current epoch
        end_epoch_time = time.perf_counter()
        logging.info(f'EPOCH {epoch} --> '
                     f'|| Training loss {avg_train_loss:.4f} '
                     f'|| Training DICE {avg_dice:.4f} '
                     # f'|| Test DICE {avg_test_dice:.4f} '
                     f'|| Epoch time: {end_epoch_time - start_epoch_time:.4f}')

    # Saving model
    torch.save({
        'epoch': config_training['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': 'scheduler'
    }, f'runs/{run_path}/fold_{n}/model_{timestamp}_fold_{n}')

    logging.info(f"\nTesting phase for fold {n}")
    model = load_pretrained_model(model, f'runs/{run_path}/fold_{n}/model_{timestamp}_fold_{n}')
    results = inference_binary_segmentation(model=model, test_loader=test_loader, path=f"runs/{run_path}/fold_{n}/",
                                            device=dev)
    logging.info(results)
    logging.info(results.mean())

    end_time = time.perf_counter()
    logging.info(f"Total time for fold {n}: {end_time - init_time:.2f}")

    # Clear the GPU memory after evaluating on the test data for this fold
    torch.cuda.empty_cache()

# Measuring total time
end_time = time.perf_counter()
logging.info(f"Total time for all of the folds: {end_time - start_time:.2f}")
