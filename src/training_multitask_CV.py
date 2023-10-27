import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, ElasticTransform
from src.utils.metrics import accuracy_from_tensor, f1_score_from_tensor
from src.dataset.BUSI_dataloader import load_datasets
from src.utils.metrics import dice_score_from_tensor
from src.utils.miscellany import init_log
from src.utils.miscellany import seed_everything
from src.utils.models import inference_multitask_binary_classification_segmentation
from src.utils.models import inference_multitask_multiclass_classification_segmentation
from src.utils.experiment_init import load_multitask_experiment_artefacts
from src.utils.criterions import apply_criterion_multitask_segmentation_classification
from src.utils.models import load_pretrained_model
from src.utils.metrics import precision, sentitivity, specificity, accuracy, f1_score
from src.utils.visualization import plot_evolution
from sklearn.metrics import confusion_matrix
from src.utils.miscellany import load_config_file
from src.utils.miscellany import write_metrics_file
from src.utils.experiment_init import device_setup
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score


def train_one_epoch():
    running_training_loss = 0.
    running_dice = 0.
    ground_truth_label = []
    predicted_label = []

    # Iterating over training loader
    for k, data in enumerate(training_loader):
        inputs, masks, label = data['image'].to(dev), data['mask'].to(dev), data['label'].to(dev)
        if len(config_data['classes']) > 2:
            label = torch.nn.functional.one_hot(label.flatten().to(torch.int64), num_classes=3).to(torch.float)

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)

        # Make predictions for this batch
        pred_class, outputs = model(inputs)

        # Compute the loss and its gradients
        segmentation_loss, classification_loss = apply_criterion_multitask_segmentation_classification(
            segmentation_criterion, masks, outputs, classification_criterion, label, pred_class,
            config_loss['inversely_weighted']
        )

        # weighting each of the loss functions
        total_loss = alpha * segmentation_loss + (1 - alpha) * classification_loss
        running_training_loss += total_loss.item()

        # Performing backward step through scaler methodology
        total_loss.backward()
        optimizer.step()

        # measuring DICE
        if type(outputs) == list:
            outputs = outputs[-1]
        outputs = torch.sigmoid(outputs) > .5
        dice = dice_score_from_tensor(masks, outputs)
        running_dice += dice

        # averaging prediction if deep supervision
        if type(pred_class) == list:
            pred_class = torch.mean(torch.stack(pred_class, dim=0), dim=0)

        # this if-else differentiates between multiclass and binary class predictions
        if len(config_data['classes']) > 2:
            label = [torch.argmax(l, keepdim=True).to(torch.float) for l in label]
            pred_class = [torch.argmax(pl, keepdim=True).to(torch.float) for pl in pred_class]
            # print(f"label: {label}, pred_class: {pred_class}")
            if type(pred_class) == list:
                for l, p in zip(label, pred_class):
                    ground_truth_label.append(l)
                    predicted_label.append(p)
            else:
                ground_truth_label.append(label)
                predicted_label.append(pred_class)
        else:
            # adding ground truth label and predicted label
            if pred_class.shape[0] > 1:  # when batch size > 1, each element is added individually
                for i in range(pred_class.shape[0]):
                    predicted_label.append((torch.sigmoid(pred_class[i, :]) > .5).double())
                    ground_truth_label.append(label[i, :])
            else:
                predicted_label.append((torch.sigmoid(pred_class) > .5).double())
                ground_truth_label.append(label)

        del segmentation_loss
        del pred_class, outputs

    if len(config_data['classes']) > 2:
        ground_truth_label = [tensor.item() for tensor in ground_truth_label]
        predicted_label = [tensor.item() for tensor in predicted_label]
        running_acc = accuracy_score(ground_truth_label, predicted_label)
        running_f1_score = f1(y_true=ground_truth_label, y_pred=predicted_label, labels=[0, 1, 2], average='micro')
        # macro_f1 = f1_score(y_true=ground_truth_label, y_pred=predicted_label, labels=[0, 1, 2], average='macro')
    else:
        running_acc = accuracy_from_tensor(torch.Tensor(ground_truth_label), torch.Tensor(predicted_label))
        running_f1_score = f1_score_from_tensor(torch.Tensor(ground_truth_label), torch.Tensor(predicted_label))

    return running_training_loss / training_loader.__len__(), running_dice / training_loader.__len__(), running_acc, running_f1_score


@torch.inference_mode()
def validate_one_epoch():
    running_validation_loss = 0.0
    running_validation_dice = 0.0
    val_ground_truth_label = []
    val_predicted_label = []

    for i, validation_data in enumerate(validation_loader):

        validation_images, validation_masks = validation_data['image'].to(dev), validation_data['mask'].to(dev)
        validation_label = validation_data['label'].to(dev)
        if len(config_data['classes']) > 2:
            validation_label = torch.nn.functional.one_hot(validation_label.flatten().to(torch.int64), num_classes=3).to(torch.float)

        pred_val_class, validation_outputs = model(validation_images)

        segmentation_val_loss, classification_val_loss = apply_criterion_multitask_segmentation_classification(
            segmentation_criterion, validation_masks, validation_outputs, classification_criterion, validation_label, pred_val_class,
            config_loss['inversely_weighted']
        )

        # weighting each of the loss functions
        validation_loss = alpha * segmentation_val_loss + (1-alpha) * classification_val_loss
        running_validation_loss += validation_loss

        # measuring DICE
        if type(validation_outputs) == list:
            validation_outputs = validation_outputs[-1]
        validation_outputs = torch.sigmoid(validation_outputs) > .5
        dice = dice_score_from_tensor(validation_masks, validation_outputs)
        running_validation_dice += dice

        # averaging prediction if deep supervision
        if type(pred_val_class) == list:
            pred_val_class = torch.mean(torch.stack(pred_val_class, dim=0), dim=0)


        if len(config_data['classes']) > 2:
            validation_label = [l.argmax() for l in validation_label]
            pred_val_class = [pl.argmax() for pl in pred_val_class]
            if len(pred_val_class) > 1:
                for l, p in zip(validation_label, pred_val_class):
                    val_ground_truth_label.append(l)
                    val_predicted_label.append(p)
            else:
                val_ground_truth_label.append(validation_label)
                val_predicted_label.append(pred_val_class)
        else:
            # adding ground truth label and predicted label
            if pred_val_class.shape[0] > 1:  # when batch size > 1, each element is added individually
                for i in range(pred_val_class.shape[0]):
                    val_predicted_label.append((torch.sigmoid(pred_val_class[i, :]) > .5).double())
                    val_ground_truth_label.append(validation_label[i, :])
            else:
                val_predicted_label.append((torch.sigmoid(pred_val_class) > .5).double())
                val_ground_truth_label.append(validation_label)

        del validation_loss
        del pred_val_class, validation_outputs

    avg_validation_loss = running_validation_loss / validation_loader.__len__()
    avg_validation_dice = running_validation_dice / validation_loader.__len__()

    if len(config_data['classes']) > 2:
        val_ground_truth_label = [tensor.item() for tensor in val_ground_truth_label]
        val_predicted_label = [tensor.item() for tensor in val_predicted_label]
        running_acc = accuracy_score(val_ground_truth_label, val_predicted_label)
        running_f1_score = f1(y_true=val_ground_truth_label, y_pred=val_predicted_label, labels=[0, 1, 2], average='micro')
        # macro_f1 = f1_score(y_true=ground_truth_label, y_pred=predicted_label, labels=[0, 1, 2], average='macro')
    else:
        running_acc = accuracy_from_tensor(torch.Tensor(val_ground_truth_label), torch.Tensor(val_predicted_label))
        running_f1_score = f1_score_from_tensor(torch.Tensor(val_ground_truth_label), torch.Tensor(val_predicted_label))

    return avg_validation_loss, avg_validation_dice, running_acc, running_f1_score


# alphas = [1, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]

# for alpha in alphas:

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
# config_training['alpha'] = alpha
alpha = config_training['alpha']
run_path = f"runs/{timestamp}_{config_model['architecture']}_{config_model['width']}_alpha_{config_training['alpha']}" \
           f"_batch_{config_data['batch_size']}"
Path(f"{run_path}").mkdir(parents=True, exist_ok=True)
init_log(log_name=f"./{run_path}/execution.log")
shutil.copyfile('./src/config.yaml', f'./{run_path}/config.yaml')

# initializing experiment's objects
n_augments = sum([v for k, v in config_data['augmentation'].items()])
transforms = torch.nn.Sequential(
    # transforms.RandomCrop(128, pad_if_needed=True),
    # ElasticTransform(alpha=25.),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation(degrees=np.random.choice(range(0, 360)))
)
train_loaders, val_loaders, test_loaders = load_datasets(config_training, config_data, transforms, mode='CV')


for n, (training_loader, validation_loader, test_loader) in enumerate(zip(train_loaders, val_loaders, test_loaders)):
    logging.info(f"\n\n *********************  FOLD {n}  ********************* \n\n")
    logging.info(f"\n\n ###############  TRAINING PHASE  ###############  \n\n")

    # creating specific paths and experiment's objects for each fold
    fold_time = time.perf_counter()
    Path(f"{run_path}/fold_{n}/segs/").mkdir(parents=True, exist_ok=True)
    Path(f"{run_path}/fold_{n}/plots/").mkdir(parents=True, exist_ok=True)
    Path(f"{run_path}/fold_{n}/features_map/").mkdir(parents=True, exist_ok=True)

    # artefacts initialization
    model, optimizer, segmentation_criterion, classification_criterion, scheduler = load_multitask_experiment_artefacts(config_data, config_model, config_opt, config_loss, n_augments, run_path)
    model = model.to(dev)

    # init metrics file
    write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                       text_to_write=f'epoch,LR,Train_loss,Validation_loss,Train_dice,Validation_dice,Train_acc,Train_F1,Validation_acc,Validation_F1')

    best_validation_loss = 1_000_000.
    patience = 0
    for epoch in range(config_training['epochs']):
        current_lr = optimizer.param_groups[0]["lr"]
        start_epoch_time = time.perf_counter()

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_train_loss, avg_dice, train_acc, train_f1_score = train_one_epoch()

        # We don't need gradients on to do reporting
        model.train(False)
        avg_validation_loss, avg_validation_dice, val_acc, val_f1_score = validate_one_epoch()

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
            }, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}')
        else:
            patience += 1

        # logging results of current epoch
        end_epoch_time = time.perf_counter()
        logging.info(f'EPOCH {epoch} --> '
                     f'|| Training loss {avg_train_loss:.4f} '
                     f'|| Validation loss {avg_validation_loss:.4f} '
                     f'|| Training DICE {avg_dice:.4f} '
                     f'|| Validation DICE  {avg_validation_dice:.4f} '
                     f'|| Training ACC {train_acc:.4f} '
                     f'|| Training F1 {train_f1_score:.4f} '
                     f'|| Validation ACC {val_acc:.4f} '
                     f'|| Validation F1 {val_f1_score:.4f} '
                     f'|| Patience: {patience} '
                     f'|| Epoch time: {end_epoch_time - start_epoch_time:.4f}')

        write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                           text_to_write=f'{epoch},{current_lr:.8f},{avg_train_loss:.4f},{avg_validation_loss:.4f},'
                                         f'{avg_dice:.4f}, {avg_validation_dice:.4f},{train_acc:.4f},'
                                         f'{train_f1_score:.4f},{val_acc:.4f},{val_f1_score:.4f}',
                           close=True)

        # early stopping
        if patience > config_training['max_patience']:
            logging.info(f"\nValidation loss did not improve over the last {patience} epochs. Stopping training")
            break

    # store metrics
    # f.close()
    metrics = pd.read_csv(f'{run_path}/fold_{n}/metrics.csv')
    plot_evolution(metrics, columns=['Train_loss', 'Validation_loss'], path=f'{run_path}/fold_{n}/loss_evolution.png')
    plot_evolution(metrics, columns=['Train_dice', 'Validation_dice'],
                   path=f'{run_path}/fold_{n}/segmentation_metrics_evolution.png')
    plot_evolution(metrics, columns=['Train_acc', 'Train_F1', 'Validation_acc', 'Validation_F1'],
                   path=f'{run_path}/fold_{n}/classification_metrics_evolution.png')

    logging.info(f"\nTesting phase for fold {n}")
    model = load_pretrained_model(model, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}')
    if len(config_data['classes']) <= 2:
        results, metrics = inference_multitask_binary_classification_segmentation(model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/", device=dev)
    else:
        results, metrics = inference_multitask_multiclass_classification_segmentation(model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/", device=dev)

    logging.info(results)

    logging.info(results.mean())
    logging.info(results.median())
    logging.info(results.max())

    # classification metrics
    if len(config_data['classes']) <= 2:
        metrics[metrics.ground_truth > 0] = 1
        cm = confusion_matrix(y_true=metrics.ground_truth, y_pred=metrics.predicted_label).ravel()
        logging.info(cm)
        tn, fp, fn, tp = cm.ravel()

        logging.info(f"Precision: {precision(tp, fp)}")
        logging.info(f"Sensitivity: {sentitivity(tp, fn)}")
        logging.info(f"Specificity: {specificity(tn, fp)}")
        logging.info(f"Accuracy: {accuracy(tp, tn, fp, fn)}")
        logging.info(f"F1 score: {f1_score(tp, fp, fn)}")

        end_time = time.perf_counter()
        logging.info(f"Total time for fold {n}: {end_time - init_time:.2f}")
    else:
        accuracy = accuracy_score(metrics.ground_truth, metrics.predicted_label)
        micro_f1 = f1(y_true=metrics.ground_truth, y_pred=metrics.predicted_label, labels=[0, 1, 2], average='micro')
        macro_f1 = f1(y_true=metrics.ground_truth, y_pred=metrics.predicted_label, labels=[0, 1, 2], average='macro')
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Macro F1 score: {macro_f1}")
        logging.info(f"Micro F1 score: {micro_f1}")

    # Clear the GPU memory after evaluating on the test data for this fold
    torch.cuda.empty_cache()

    del model

# Measuring total time
end_time = time.perf_counter()
logging.info(f"Total time for all of the folds: {end_time - init_time:.2f}")
