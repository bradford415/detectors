import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as F
from tqdm import tqdm

from detectors.postprocessing.eval import (
    ap_per_class,
    get_batch_statistics,
    print_eval_stats,
)
from detectors.postprocessing.nms import non_max_suppression
from detectors.utils import misc
from detectors.utils.box_ops import val_preds_to_img_size, xywh2xyxy

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader_test: Iterable,
    class_names: List,
    # img_size: int = 416,
    criterion: Optional[nn.Module] = None,
    output_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tuple, List]:
    """A single forward pass to evluate the val set after training an epoch

    Args:
        model: Model to train
        criterion: TODO
        dataloader_val: Dataloader for the validation set
        device: Device to run the model on

    Returns:
        A Tuple containing
            1. A Tuple of the (prec, rec, ap, f1, and class) per class
            2. A list of tuples containing the image_path and detections after postprocessing with nms

    """
    model.eval()

    labels = []
    sample_metrics = []  # List of tuples (true positives, cls_confs, cls_labels)
    image_paths = []
    final_preds = (
        []
    )  # holds a tensor of predictions for each image; (num_detections, 6)
    all_losses = torch.zeros(4, dtype=torch.float32)
    val_loss = torch.tensor([0.0], dtype=torch.float32)
    for steps, (samples, targets, target_meta) in enumerate(
        tqdm(dataloader_test, desc="Evaluating", ncols=100)
    ):
        img_size = samples.shape[2]

        # NOTE: I don't think we need to move targets to gpu during eval
        samples = samples.to(device)
        targets = targets.to(device)

        # Extract target labels and convert target boxes to xyxy; extract image paths for visualization
        labels += targets[:, 1].tolist()
        image_paths += [meta["image_path"] for meta in target_meta]

        # targets = targets.to(device)

        # # Extract object labels from all samples in the batch into a 1d python list
        # for target in targets:
        #     # extract labels (b*labels_per_img,) and image paths for visualization
        #     labels += target["labels"].tolist()

        #     # convert bbox yolo format to xyxy
        #     target["boxes"] = cxcywh_to_xyxy(target["boxes"])

        # (b, num_preds, 5 + num_classes) where 5 is (tl_x, tl_y, br_x, br_y, objectness)

        eval_outputs = model(samples)

        train_output = eval_outputs[0]
        predictions = eval_outputs[1]

        # Transfer preds to CPU for post processing
        # predictions = misc.to_cpu(predictions)
        if criterion is not None:
            _, loss_components = criterion(train_output, targets, model)
            all_losses += loss_components

        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        # TODO: define these thresholds in the config file under postprocessing maybe?
        # TODO: this is wrong I'm pretty sure; list (b,) of tensor predictions (max_nms_preds, 6)
        # where 6 = (tl_x, tl_y, br_x, br_y, conf, cls)

        # NOTE: yolo predicts (cx, cy, w, h, obj, cls_0, ..., num_classes)
        #       but nms() converts boxes to (x1, y1, x2, y2, obj, cls)
        # nms_preds = non_max_suppression(predictions, conf_thres=0.01, iou_thres=0.5)
        nms_preds = non_max_suppression(predictions, conf_thres=0.1, iou_thres=0.5)
        final_preds += nms_preds

        targets = targets.to("cpu")

        # [[TPs, predicted_scores, pred_labels], ..., num_val_images]
        sample_metrics += get_batch_statistics(nms_preds, targets, iou_threshold=0.5)

    if criterion is not None:
        log.info("\nValidation losses:")
        log.info(
            "val_loss: %-10.4f bbox_loss: %-10.4f obj_loss: %-10.4f class_loss: %-10.4f\n",
            all_losses[3] / steps,
            all_losses[0] / steps,
            all_losses[1] / steps,
            all_losses[2] / steps,
        )

    ## TODO plot val loss - can probably just use the function

    # No detections over whole validation set
    if len(sample_metrics) == 0:
        log.info("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, axis=0) for x in list(zip(*sample_metrics))
    ]

    assert (
        true_positives.ndim == 1
        and true_positives.shape == pred_scores.shape == pred_labels.shape
    )

    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose=True)

    # Pair the image paths with the final predictions to visualize
    image_detections = list(zip(image_paths, final_preds))

    val_loss = all_losses[3] / steps

    return metrics_output, image_detections, val_loss


@torch.no_grad()
def evaluate_detr(
    model: nn.Module,
    dataloader_test: Iterable,
    class_names: List,
    postprocessors: nn.Module,
    # img_size: int = 416,
    criterion: Optional[nn.Module] = None,
    output_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tuple, List]:
    """A single forward pass to evluate the val set after training an epoch

    Args:
        model: Model to train
        postprocessors: postprocessing that needs to be applied after validation/inference;
                        e.g., convert a models normalized outputs to the original image size
        criterion: TODO
        dataloader_val: Dataloader for the validation set
        device: Device to run the model on

    Returns:
        A Tuple containing
            1. A Tuple of the (prec, rec, ap, f1, and class) per class
            2. A list of tuples containing the image_path and detections after postprocessing with nms

    """
    model.eval()

    iou_type = "bbox"

    ### START HERE - pass in post processor and the other stuff the function needs

    labels = []
    sample_metrics = []  # List of tuples (true positives, cls_confs, cls_labels)
    image_paths = []
    final_preds = (
        []
    )  # holds a tensor of predictions for each image; (num_detections, 6)
    all_losses = torch.zeros(4, dtype=torch.float32)
    val_loss = torch.tensor([0.0], dtype=torch.float32)
    for steps, (samples, targets, target_meta) in enumerate(
        tqdm(dataloader_test, desc="Evaluating", ncols=100)
    ):
        img_size = samples.shape[2]

        # NOTE: I don't think we need to move targets to gpu during eval
        samples = samples.to(device)
        targets = targets.to(device)

        # Extract target labels and convert target boxes to xyxy; extract image paths for visualization
        labels += targets[:, 1].tolist()
        image_paths += [meta["image_path"] for meta in target_meta]

        # targets = targets.to(device)

        # # Extract object labels from all samples in the batch into a 1d python list
        # for target in targets:
        #     # extract labels (b*labels_per_img,) and image paths for visualization
        #     labels += target["labels"].tolist()

        #     # convert bbox yolo format to xyxy
        #     target["boxes"] = cxcywh_to_xyxy(target["boxes"])

        # (b, num_preds, 5 + num_classes) where 5 is (tl_x, tl_y, br_x, br_y, objectness)

        eval_outputs = model(samples)

        train_output = eval_outputs[0]
        predictions = eval_outputs[1]

        # Transfer preds to CPU for post processing
        # predictions = misc.to_cpu(predictions)
        if criterion is not None:
            _, loss_components = criterion(train_output, targets, model)
            all_losses += loss_components

        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        # TODO: define these thresholds in the config file under postprocessing maybe?
        # TODO: this is wrong I'm pretty sure; list (b,) of tensor predictions (max_nms_preds, 6)
        # where 6 = (tl_x, tl_y, br_x, br_y, conf, cls)

        # NOTE: yolo predicts (cx, cy, w, h, obj, cls_0, ..., num_classes)
        #       but nms() converts boxes to (x1, y1, x2, y2, obj, cls)
        # nms_preds = non_max_suppression(predictions, conf_thres=0.01, iou_thres=0.5)
        nms_preds = non_max_suppression(predictions, conf_thres=0.1, iou_thres=0.5)
        final_preds += nms_preds

        targets = targets.to("cpu")

        # [[TPs, predicted_scores, pred_labels], ..., num_val_images]
        sample_metrics += get_batch_statistics(nms_preds, targets, iou_threshold=0.5)

    if criterion is not None:
        log.info("\nValidation losses:")
        log.info(
            "val_loss: %-10.4f bbox_loss: %-10.4f obj_loss: %-10.4f class_loss: %-10.4f\n",
            all_losses[3] / steps,
            all_losses[0] / steps,
            all_losses[1] / steps,
            all_losses[2] / steps,
        )

    ## TODO plot val loss - can probably just use the function

    # No detections over whole validation set
    if len(sample_metrics) == 0:
        log.info("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, axis=0) for x in list(zip(*sample_metrics))
    ]

    assert (
        true_positives.ndim == 1
        and true_positives.shape == pred_scores.shape == pred_labels.shape
    )

    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose=True)

    # Pair the image paths with the final predictions to visualize
    image_detections = list(zip(image_paths, final_preds))

    val_loss = all_losses[3] / steps

    return metrics_output, image_detections, val_loss


def load_model_checkpoint(
    checkpoint_path: str,
    model: nn.Module = None,
    optimizer: nn.Module = None,
    device=torch.device("cpu"),
    lr_scheduler: Optional[nn.Module] = None,
):
    """Load the checkpoints of a trained or pretrained model from the state_dict file;
    this could be from a fully trained model or a partially trained model that you want
    to resume training from.

    Args:
        checkpoint_path: path to the weights file to resume training from
        model: the model being trained
        optimizer: the optimizer used during training
        device: the device to map the checkpoints to
        lr_scheduler: the learning rate scheduler used during training

    Returns:
        the epoch to start training on
    """
    # Load the torch weights
    weights = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # load the state dictionaries for the necessary training modules
    if model is not None:
        model.load_state_dict(weights["model"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(weights["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(weights["lr_scheduler"])
    start_epoch = weights["epoch"]

    return start_epoch
