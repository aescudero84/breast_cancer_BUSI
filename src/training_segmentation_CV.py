import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torchvision import transforms
from torchvision.transforms.v2 import RandomResizedCrop, ElasticTransform

from src.dataset.BUSI_dataloader import BUSI_dataloader_CV
from src.utils.metrics import dice_score_from_tensor
from src.utils.miscellany import init_log
from src.utils.miscellany import seed_everything
from src.utils.models import inference_binary_segmentation
from src.utils.models import init_loss_function
from src.utils.models import init_optimizer
from src.utils.models import init_segmentation_model
from src.utils.models import load_pretrained_model
from src.utils.visualization import plot_evolution
from src.utils.models import init_lr_scheduler


def load_config_file(path):
    with open(path) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)
        logging.info(pformat(config))
    return config['model'], config['optimizer'], config['loss'], config['training'], config['data']


def write_metrics_file(path_file, text_to_write, close=True):
    with open(path_file, 'a') as fm:
        fm.write(text_to_write)
        fm.write("\n")
        if close:
            fm.close()


def apply_criterion_binary_segmentation(criterion, ground_truth, segmentation, inversely_weighted=False):
    if type(segmentation) == list:
        if inversely_weighted:
            loss = torch.sum(torch.stack(
                [criterion(s, ground_truth) / (j+1) for j, s in enumerate(reversed(segmentation))]
            ))
        else:
            loss = torch.sum(torch.stack(
                [criterion(s, ground_truth) for j, s in enumerate(reversed(segmentation))]
            ))
    else:
        loss = loss_fn(segmentation, ground_truth)

    if not torch.isnan(loss):
        return loss
    else:
        logging.info("NaN in model loss!!")
        sys.exit(1)


def train_one_epoch():
    training_loss = 0.
    running_dice = 0.

    # Iterating over training loader
    for k, data in enumerate(training_loader):
        inputs, masks = data['image'].to(dev), data['mask'].to(dev)

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        batch_loss = apply_criterion_binary_segmentation(criterion=loss_fn, ground_truth=masks, segmentation=outputs,
                                                         inversely_weighted=config_loss['inversely_weighted'])
        training_loss += batch_loss.item()

        # Performing backward step through scaler methodology
        batch_loss.backward()
        optimizer.step()

        # measuring DICE
        if type(outputs) == list:
            outputs = outputs[-1]
        outputs = torch.sigmoid(outputs) > .5
        dice = dice_score_from_tensor(masks, outputs)
        running_dice += dice

        del batch_loss
        del outputs

    return training_loss / training_loader.__len__(), running_dice / training_loader.__len__()


@torch.no_grad()
def validate_one_epoch():
    validation_loss = 0.0
    validation_dice = 0.0
    for i, validation_data in enumerate(validation_loader):

        validation_images, validation_masks = validation_data['image'].to(dev), validation_data['mask'].to(dev)
        validation_outputs = model(validation_images)

        # Compute the validation loss
        batch_validation_loss = apply_criterion_binary_segmentation(criterion=loss_fn, ground_truth=validation_masks,
                                                                    segmentation=validation_outputs,
                                                                    inversely_weighted=config_loss['inversely_weighted'])
        validation_loss += batch_validation_loss.item()

        # measuring DICE
        if type(validation_outputs) == list:
            validation_outputs = validation_outputs[-1]
        validation_outputs = torch.sigmoid(validation_outputs) > .5
        dice = dice_score_from_tensor(validation_masks, validation_outputs)
        validation_dice += dice

    return validation_loss / validation_loader.__len__(), validation_dice / validation_loader.__len__()


# initializing folder structures and log
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# loading config file
config_model, config_opt, config_loss, config_training, config_data = load_config_file(path='./src/config.yaml')

# initializing seed and gpu if possible
seed_everything(config_training['seed'], cuda_benchmark=config_training['cuda_benchmark'])
if torch.cuda.is_available():
    dev = "cuda:0"
    logging.info("GPU will be used to train the model")
else:
    dev = "cpu"
    logging.info("CPU will be used to train the model")

# initializing experiment's objects
n_augments = sum([v for k, v in config_data['augmentation'].items()])
transforms = torch.nn.Sequential(
    # transforms.RandomCrop(128, pad_if_needed=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # RandomResizedCrop(size=(128, 128)),
    ElasticTransform(alpha=25.),
    transforms.RandomRotation(degrees=np.random.choice(range(0, 360))),
    # transforms.RandomCrop(64)
)

if config_training['CV'] > 1:
    train_loaders, val_loaders, test_loaders = BUSI_dataloader_CV(seed=config_training['seed'],
                                                                  batch_size=config_data['batch_size'],
                                                                  transforms=transforms,
                                                                  remove_outliers=config_data['remove_outliers'],
                                                                  train_size=config_data['train_size'],
                                                                  n_folds=config_training['CV'],
                                                                  augmentations=config_data['augmentation'],
                                                                  normalization=None,
                                                                  classes=config_data['classes'],
                                                                  oversampling=config_data['oversampling'],
                                                                  use_duplicated_to_train=config_data['use_duplicated_to_train'],
                                                                  path_images=config_data['input_img'])
else:
    sys.exit("This code is prepared for receiving a CV greater than 1")

for n, (training_loader, validation_loader, test_loader) in enumerate(zip(train_loaders, val_loaders, test_loaders)):

    # creating specific paths and experiment's objects for each fold
    init_time = time.perf_counter()
    run_path = (f"runs/{timestamp}_{config_model['architecture']}_{config_model['width']}_batch_"
                f"{config_data['batch_size']}_{'_'.join(config_data['classes'])}")
    Path(f"{run_path}/fold_{n}/segs/").mkdir(parents=True, exist_ok=True)
    Path(f"{run_path}/fold_{n}/plots/").mkdir(parents=True, exist_ok=True)
    shutil.copyfile('./src/config.yaml', f'./{run_path}/config.yaml')
    init_log(log_name=f"./{run_path}/fold_{n}/execution_fold_{n}.log")
    model = init_segmentation_model(architecture=config_model['architecture'],
                                    sequences=config_model['sequences'] + n_augments,
                                    width=config_model['width'], deep_supervision=config_model['deep_supervision'],
                                    save_folder=Path(f'./{run_path}/')).to(dev)
    optimizer = init_optimizer(model=model, optimizer=config_opt['opt'], learning_rate=config_opt['lr'])
    loss_fn = init_loss_function(loss_function=config_loss['function'])
    scheduler = init_lr_scheduler(optimizer=optimizer, scheduler=config_opt['scheduler'], t_max=config_opt['t_max'],
                                  patience=config_opt['patience'], min_lr=config_opt['min_lr'],
                                  factor=config_opt['decrease_factor'])

    logging.info(f"\n\n *********************  FOLD {n}  ********************* \n\n")

    # init metrics file
    write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                       text_to_write=f'epoch,LR,Train,Validation,Test,Train_loss,Val_loss')

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
        if config_opt['scheduler'] == 'cosine':
            scheduler.step()
        else:
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
            }, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}.tar')
        else:
            patience += 1

        # logging results of current epoch
        end_epoch_time = time.perf_counter()
        results = inference_binary_segmentation(model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/", device=dev)
        logging.info(f'EPOCH {epoch} --> '
                     f'|| Training loss {avg_train_loss:.4f} '
                     f'|| Validation loss {avg_validation_loss:.4f} '
                     f'|| Training DICE {avg_dice:.4f} '
                     f'|| Validation DICE  {avg_validation_dice:.4f} '
                     # f'|| Test DICE  {results["DICE"].mean():.4f} '
                     f'|| Patience: {patience} '
                     f'|| Epoch time: {end_epoch_time - start_epoch_time:.4f} '
                     f'|| LR: {optimizer.param_groups[0]["lr"]:.8f}')

        # write metrics
        write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                           text_to_write=f'{epoch},{optimizer.param_groups[0]["lr"]:.8f},'
                                         f'{avg_dice:.4f}, {avg_validation_dice:.4f},{results["DICE"].mean():.4f},'
                                         f'{avg_train_loss:.4f},{avg_validation_loss:.4f}',
                           close=True)

        # early stopping
        if patience > config_training['max_patience']:
            logging.info(f"\nValidation loss did not improve over the last {patience} epochs. Stopping training")
            break

    # store metrics
    metrics = pd.read_csv(f'{run_path}/fold_{n}/metrics.csv')
    plot_evolution(metrics, columns=['Train', 'Validation', 'Test'],
                   path=f'{run_path}/fold_{n}/plots/metrics_evolution.png',
                   title='Evolucion de la metrica DICE', ylabel='DICE',)
    plot_evolution(metrics, columns=['Train_loss', 'Val_loss'],
                   path=f'{run_path}/fold_{n}/plots/loss_evolution.png',
                   title='Evolucion de la funcion de perdida DICE', ylabel='Loss DICE',)

    # validation results
    logging.info(f"\nValidation phase for fold {n}")
    model = load_pretrained_model(model, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}.tar')
    val_results = inference_binary_segmentation(model=model, test_loader=validation_loader,
                                                path=f"{run_path}/fold_{n}/", device=dev)
    logging.info(val_results.mean())

    logging.info(f"\nTesting phase for fold {n}")
    results = inference_binary_segmentation(model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/",
                                            device=dev)

    logging.info(results)
    logging.info(results.mean())

    end_time = time.perf_counter()
    logging.info(f"Total time for fold {n}: {end_time - init_time:.2f}")

    # Clear the GPU memory after evaluating on the test data for this fold
    torch.cuda.empty_cache()

# Measuring total time
end_time = time.perf_counter()
logging.info(f"Total time for all of the folds: {end_time - init_time:.2f}")
