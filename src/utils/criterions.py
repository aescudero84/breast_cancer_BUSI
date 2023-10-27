import torch
import logging
import sys


def apply_criterion_binary_segmentation(
        criterion: torch.nn.Module,
        ground_truth: torch.Tensor,
        segmentation: torch.Tensor,
        inversely_weighted: bool = False
):
    if isinstance(segmentation, list):
        if inversely_weighted:
            loss = torch.sum(torch.stack(
                [criterion(s, ground_truth) / (j+1) for j, s in enumerate(reversed(segmentation))]
            ))
        else:
            loss = torch.sum(torch.stack(
                [criterion(s, ground_truth) for j, s in enumerate(reversed(segmentation))]
            ))
    else:
        loss = criterion(segmentation, ground_truth)

    if not torch.isnan(loss):
        return loss
    else:
        logging.info("NaN in model loss!!")
        sys.exit(1)


def apply_criterion_multitask_segmentation_classification(
        criterion_seg: torch.nn.Module,
        ground_truth: torch.Tensor,
        segmentation: torch.Tensor,
        criterion_class: torch.nn.Module,
        label: torch.Tensor,
        predicted_class: torch.Tensor,
        inversely_weighted=False
):
    if isinstance(segmentation, list):
        if inversely_weighted:
            segmentation_loss = torch.sum(torch.stack([criterion_seg(s, ground_truth) / (n + 1) for n, s in enumerate(reversed(segmentation))]))
            classification_loss = torch.sum(torch.stack([criterion_class(c, label) for n, c in enumerate(reversed(predicted_class))]))
        else:
            segmentation_loss = torch.sum(torch.stack([criterion_seg(s, ground_truth) for n, s in enumerate(reversed(segmentation))]))
            classification_loss = torch.sum(torch.stack([criterion_class(c, label) for n, c in enumerate(reversed(predicted_class))]))
    else:
        segmentation_loss = criterion_seg(segmentation, ground_truth)
        classification_loss = criterion_class(predicted_class, label)

    if not torch.isnan(segmentation_loss) and not torch.isnan(classification_loss):
        return segmentation_loss, classification_loss
    else:
        logging.info("NaN in model loss!!")
        sys.exit(1)
