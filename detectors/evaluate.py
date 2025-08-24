import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from pycocotools.coco import COCO
from torch import nn
from tqdm import tqdm

from detectors.data.coco_eval import CocoEvaluator
from detectors.postprocessing.eval import (
    ap_per_class,
    get_batch_statistics,
    print_eval_stats,
)
from detectors.postprocessing.nms import non_max_suppression
from detectors.utils import distributed, misc
from detectors.utils.box_ops import xywh2xyxy

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
    coco_api: Optional[COCO],
    postprocessors: nn.Module,
    criterion: Optional[nn.Module] = None,
    enable_amp: bool = False,
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

    # typically just "bbox" iou
    iou_types = tuple(postprocessors.keys())

    coco_evaluator = CocoEvaluator(coco_api, iou_types)

    epoch_loss = []
    running_loss_dict = {}  # TODO should make these default dicts
    running_total_loss_scaled = 0.0
    num_steps = 0
    for steps, (samples, targets) in enumerate(dataloader_test, 1):
        num_steps += 1
        samples = samples.to(device)

        # move label tensors to gpu
        targets = [
            {
                key: (val.to(device) if isinstance(val, torch.Tensor) else val)
                for key, val in t.items()
            }
            for t in targets
        ]

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=enable_amp,
        ):
            preds = model(samples, targets)

            loss_dict = criterion(preds, targets)

        weight_dict = criterion.weight_dict

        # compute the total loss by scaling each component of the loss by its weight value;
        # if the loss key is not a key in the weight_dict, then it is not used in the total loss;
        # dino sums a total of 39 losses w/ the default values;
        # see detectors/models/README.md for information on the losses that propagate gradients
        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        ## Remove this maybe? ###
        epoch_loss.append(loss.detach().cpu())

        # average the losses across all processes; represents the current step loss
        reduced_loss_dict = distributed.reduce_dict(loss_dict, average=True)

        # scale the loss components in the reduced dict just like the total loss computation;
        # this is the average loss across all processes
        reduced_loss_dict_scaled = {
            k: v * weight_dict[k]
            for k, v in reduced_loss_dict.items()
            if k in weight_dict
        }

        # sum the averaged (avg num boxes per node) loss components; at the end of validation
        # we'll divide by the numbber of steps to get an average loss
        for key, val in reduced_loss_dict_scaled.items():
            if key in running_loss_dict:
                running_loss_dict[key] += val.detach()
            else:
                running_loss_dict[key] = val.detach()

        running_total_loss_scaled += sum(reduced_loss_dict_scaled.values()).item()

        # extract the original image sizes (b, 2) where 2 = (h, w)
        orig_target_sizes = torch.stack([img["orig_size"] for img in targets], dim=0)

        # obtain `num_select` confidence `scores`, `labels`, and `bboxes` for each image;
        # bboxes are converted from relative [0, 1] cxcywh format to absolute
        # [0, orig_img_w/h] xyxy format
        results: list[dict] = postprocessors["bbox"](preds, orig_target_sizes)

        assert len(results) == samples.tensors.shape[0]

        # map the ground-truth image id to the predicted results
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }

        # NOTE: when calling CocoEvaluator.update() the predicted boxes will be converted to XYWH to match
        #       the ground-truth boxes; specifically , this is done here:
        #           https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/datasets/coco_eval.py#L89

        # compute the per image, per-category, per-area-range IoU metrics and store in coco_evaluator;
        # example: per-category will give us an IoU for each class (dog, cat, mouse)
        coco_evaluator.update(res)

    running_loss_dict["loss"] = running_total_loss_scaled
    stats = distributed.synchronize_loss_between_processes(num_steps, running_loss_dict)

    if coco_evaluator is not None:
        # Update the coco_eval object with the image ids and evaluations from all processes
        coco_evaluator.synchronize_between_processes()

        # accumulate() takes the per-image evaluation results (evalImgs) computed by evaluate().
        # for each category, area, IoU threshold, and max detections, it:
        #   1. collects all true positive / false positive matches (dtMatches, dtIgnore, gtIgnore).
        #   2. Computes cumulative TP and FP sums.
        #   3. Computes precision at fixed recall thresholds (p.recThrs) by sampling the cumulative data.
        #   4. Stores everything in the self.eval dictionary:
        #      - precision → the sampled precision values (used for plotting or computing AP)
        #      - recall → the maximum recall achieved per IoU/class/area/maxDet
        #      - scores → detection scores corresponding to the sampled precision
        coco_evaluator.accumulate()

        # computes the overall mAP (and other AP variants) through the precision array (T, R, K, A, M)
        # where T = iou thresholds (0.50:0.95), R = recall thresholds (101 points 0:0.01:1.0),
        #       K = categories (all classes), A = area ranges (all, small, medium, large),
        #       M = max detections per image (1, 10, 100)
        # overall mAP[0.5:0.95] (mean of all per-category precision values across all IoUs, recall thresholds,
        # and area ranges) slices the precision array like precision[:, :, :, 0, 2] which means to
        # average over all ious, all recalls, all classes, "all" area, and maxDet=100
        coco_evaluator.summarize()

        # extract the NumPy array of summary metrics computed by summarize() and convert to python list;
        # by default this contains the 12 values AP/AR values that are printed
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

    return stats


@torch.no_grad()
def test_detr(
    model: nn.Module,
    dataloader_test: Iterable,
    coco_api: Optional[COCO],
    postprocessors: nn.Module,
    criterion: Optional[nn.Module] = None,
    enable_amp: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tuple, List]:
    """A stripped down version of evaluate_detr() only for computing metrics on the test set

    Args:
        model: Model to train
        postprocessors: postprocessing that needs to be applied after validation/inference;
                        e.g., convert a models normalized outputs to the original image size
        dataloader_val: Dataloader for the validation set
        device: Device to run the model on

    Returns:
        A Tuple containing
            1. A Tuple of the (prec, rec, ap, f1, and class) per class
            2. A list of tuples containing the image_path and detections after postprocessing with nms

    """
    model.eval()

    # typically just "bbox" iou
    iou_types = tuple(postprocessors.keys())

    coco_evaluator = CocoEvaluator(coco_api, iou_types)

    for steps, (samples, targets) in tqdm(
        enumerate(dataloader_test, 1), total=len(dataloader_test), ncols=70
    ):
        samples = samples.to(device)

        # move label tensors to gpu
        targets = [
            {
                key: (val.to(device) if isinstance(val, torch.Tensor) else val)
                for key, val in t.items()
            }
            for t in targets
        ]

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=enable_amp,
        ):
            preds = model(samples, targets)

        # extract the original image sizes (b, 2) where 2 = (h, w)
        orig_target_sizes = torch.stack([img["orig_size"] for img in targets], dim=0)

        # obtain `num_select` confidence `scores`, `labels`, and `bboxes` for each image;
        # bboxes are converted from relative [0, 1] cxcywh format to absolute
        # [0, orig_img_w/h] xyxy format
        results: list[dict] = postprocessors["bbox"](preds, orig_target_sizes)

        assert len(results) == samples.tensors.shape[0]

        # map the ground-truth image id to the predicted results
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }

        # NOTE: when calling CocoEvaluator.update() the predicted boxes will be converted to XYWH to match
        #       the ground-truth boxes; specifically , this is done here:
        #           https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/datasets/coco_eval.py#L89

        # compute the per image, per-category, per-area-range IoU metrics and store in coco_evaluator;
        # example: per-category will give us an IoU for each class (dog, cat, mouse)
        coco_evaluator.update(res)

    if coco_evaluator is not None:
        # Update the coco_eval object with the image ids and evaluations from all processes
        coco_evaluator.synchronize_between_processes()

        # accumulate() takes the per-image evaluation results (evalImgs) computed by evaluate().
        # for each category, area, IoU threshold, and max detections, it:
        #   1. collects all true positive / false positive matches (dtMatches, dtIgnore, gtIgnore).
        #   2. Computes cumulative TP and FP sums.
        #   3. Computes precision at fixed recall thresholds (p.recThrs) by sampling the cumulative data.
        #   4. Stores everything in the self.eval dictionary:
        #      - precision → the sampled precision values (used for plotting or computing AP)
        #      - recall → the maximum recall achieved per IoU/class/area/maxDet
        #      - scores → detection scores corresponding to the sampled precision
        coco_evaluator.accumulate()

        # computes the overall mAP (and other AP variants) through the precision array (T, R, K, A, M)
        # where T = iou thresholds (0.50:0.95), R = recall thresholds (101 points 0:0.01:1.0),
        #       K = categories (all classes), A = area ranges (all, small, medium, large),
        #       M = max detections per image (1, 10, 100)
        # overall mAP[0.5:0.95] (mean of all per-category precision values across all IoUs, recall thresholds,
        # and area ranges) slices the precision array like precision[:, :, :, 0, 2] which means to
        # average over all ious, all recalls, all classes, "all" area, and maxDet=100
        coco_evaluator.summarize()

        # extract the NumPy array of summary metrics computed by summarize() and convert to python list;
        # by default this contains the 12 values AP/AR values that are printed
        stats = {}
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

    return stats


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
