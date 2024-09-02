import cProfile
import datetime
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import psutil
import torch
from pycocotools.coco import COCO
from torch import nn
from torch.utils import data
from torchvision.transforms import functional as F
from tqdm import tqdm

from detectors.data.coco_eval import CocoEvaluator
from detectors.data.coco_utils import convert_to_coco_api
from detectors.postprocessing.eval import (ap_per_class, get_batch_statistics,
                                           print_eval_stats)
from detectors.postprocessing.nms import non_max_suppression
from detectors.utils import misc, plots
from detectors.utils.box_ops import cxcywh_to_xyxy, val_preds_to_img_size

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    output_path: str,
    model: nn.Module,
    dataloader_test: Iterable,
    class_names: List,
    device: torch.device = torch.device("cpu"),
) -> None:
    """A single forward pass to evluate the val set after training an epoch

    Args:
        model: Model to train
        criterion: Loss function; only used to inspect the loss on the val set,
                    not used for backpropagation
        dataloader_val: Dataloader for the validation set
        device: Device to run the model on
    """
    model.eval()

    labels = []
    sample_metrics = []  # List of tuples (true positives, cls_confs, cls_labels)
    for steps, (samples, targets) in enumerate(dataloader_test):
        samples = samples.to(device)

        # Extract labels from all samples in the batch into a 1d list
        for target in targets:
            labels += target["labels"].tolist()

        for target in targets:
            target["boxes"] = cxcywh_to_xyxy(target["boxes"])

        # Predictions (B, num_preds, 5 + num_classes) where 5 is (tl_x, tl_y, br_x, br_y, objectness)
        predictions = model(samples, inference=True)

        # Transfer preds to CPU for post processing
        predictions = misc.to_cpu(predictions)

        # TODO: define these thresholds in the config file under postprocessing maybe?
        nms_preds = non_max_suppression(
            predictions, conf_thres=0.1, iou_thres=0.5  # nms thresh
        )

        sample_metrics += get_batch_statistics(nms_preds, targets, iou_threshold=0.5)

    # No detections over whole validation set
    if len(sample_metrics) == 0:
        log.info("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics (batch_size*num_preds,)
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))
    ]

    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose=True)

    return metrics_output


def load_model_state_dict(model: nn.Module, weights_path: str):
    """Load the weights of a trained or pretrained model from the state_dict file;
    this could be from a fully trained model or a partially trained model that you want
    to resume training from.

    Args:
        model: The torch model to load the weights into
        weights_path:
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Select device for inference

    state_dict  = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict["model"])

    return model
