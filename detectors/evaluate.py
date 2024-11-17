import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as F
from tqdm import tqdm

from detectors.postprocessing.eval import (ap_per_class, get_batch_statistics,
                                           print_eval_stats)
from detectors.postprocessing.nms import non_max_suppression
from detectors.utils import misc
from detectors.utils.box_ops import cxcywh_to_xyxy, val_preds_to_img_size

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader_test: Iterable,
    class_names: List,
    output_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tuple, List]:
    """A single forward pass to evluate the val set after training an epoch

    Args:
        model: Model to train
        criterion: Loss function; only used to inspect the loss on the val set,
                    not used for backpropagation
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
    final_preds = []
    for steps, (samples, targets) in enumerate(
        tqdm(dataloader_test, desc="Evaluating")
    ):
        samples = samples.to(device)

        # Extract object labels from all samples in the batch into a 1d python list
        for target in targets:
            # extract labels (b*labels_per_img,) and image paths for visualization
            labels += target["labels"].tolist()
            image_paths.append(target["image_path"])

            # convert bbox yolo format to xyxy
            target["boxes"] = cxcywh_to_xyxy(target["boxes"])

        # (b, num_preds, 5 + num_classes) where 5 is (tl_x, tl_y, br_x, br_y, objectness)
        predictions = model(samples, inference=True)

        # Transfer preds to CPU for post processing
        predictions = misc.to_cpu(predictions)

        # TODO: define these thresholds in the config file under postprocessing maybe?
        # list (b,) of tensor predictions (max_nms_preds, 6)
        # where 6 = (tl_x, tl_y, br_x, br_y, conf, cls)
        nms_preds = non_max_suppression(predictions, conf_thres=0.1, iou_thres=0.5)
        final_preds.extend(nms_preds)

        # [(num_true_)... batch_size]
        sample_metrics += get_batch_statistics(nms_preds, targets, iou_threshold=0.5)

    breakpoint()
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

    # Pair the image paths with the final predictions to visualize
    image_detections = list(zip(image_paths, final_preds))

    return metrics_output, image_detections


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

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict["model"])

    return model
