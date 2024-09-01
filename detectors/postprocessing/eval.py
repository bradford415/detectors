import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm

from detectors.utils.box_ops import bbox_iou

log = logging.getLogger(__name__)


def ap_per_class(
    tp, conf, pred_cls, target_cls
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.

    Args:
        tp: True positives list found from get_batch_statistics();
            binary list 1 for tp 0 for fp
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).

    Returns:
        A Tuple of the (prec, rec, ap, f1, and class) per class
    """
    # Sort by objectness in decreasing order (highest confidence first)
    best_conf_i = np.argsort(-conf)
    tp, conf, pred_cls = (
        tp[best_conf_i],
        conf[best_conf_i],
        pred_cls[best_conf_i],
    )  # (batch_size*num_preds,)

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, prec, rec = [], [], []
    for cls in tqdm.tqdm(unique_classes, desc="Computing AP"):
        # Select indices where pred classes equals the current target class (cls)
        cls_i = pred_cls == cls

        n_gt = (target_cls == cls).sum()  # Number of ground truth objects
        n_p = cls_i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            rec.append(0)
            prec.append(0)
        else:
            # Accumulate FPs and TPs
            fp_count = (
                1 - tp[cls_i]
            ).cumsum()  # select the true positives for the current cls
            tp_count = (tp[cls_i]).cumsum()

            # Recall; tp / (tp + fn) => n_gt = (tp + fn)

            recall_curve = tp_count / (n_gt + 1e-16)
            rec.append(recall_curve[-1])

            # Precision; tp / (tp + fp)
            precision_curve = tp_count / (tp_count + fp_count)
            prec.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    prec, rec, ap = np.array(prec), np.array(rec), np.array(ap)
    f1 = 2 * prec * rec / (prec + rec + 1e-16)

    return prec, rec, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """Compute the average precision, for one class, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args:
        recall:    The recall curve (list).
        precision: The precision curve (list).

    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end;
    # these sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope;
    # I believe this smoothes out the zigzag pattern of the PR curve and turns
    # it into rectangles; this post explains it very well:
    # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value;
    # the area under the PR curve is only sampled where the precision of the smoothed curve
    # (no zigzags) changees
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def print_eval_stats(
    metrics_output: Optional[Tuple[np.ndarray, ...]],
    class_names: List[int],
    verbose: bool = True,
):
    """TODO

    Args:
        metrics_output: A tuple containing the precision, recall, AP, f1, and class index;
                        each element is a numpy array of length (num_unique_targets,) for the batch
                        of images; ap_class is used to index these numpy arrays thus
                        they all have the same length
        class_names: List of class names of the dataset; list order should match the dataset index label
        verbose: Whether to print the AP for each class

    """
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        # breakpoint()
        if verbose:
            # Prints class AP and mean AP
            for index, cls_num in enumerate(ap_class):
                log.info(
                    "%s: %-15.2d %s =  %-15s %s = %-15.4f",
                    "index",
                    index,
                    "class",
                    class_names[cls_num],
                    "AP",
                    AP[index],
                )
                # ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            # print(ap_table)#.table)
        log.info("---- mAP %.5f} ----", AP.mean())
    else:
        print("---- mAP not measured (no detections found by model) ----")


def get_batch_statistics(
    outputs: List[torch.Tensor], targets: List[Dict], iou_threshold
) -> List:
    """Compute true positives, predicted scores and predicted labels per sample.
    This function must be used after non_max_suppression

    Args:
        outputs: Model predictions after non_max_suppression has been applied
                 shape of each list element (max_preds, 6) where 6 = (tl_x, tl_y, br_x, br_y, conf, cls)
        targets: Ground truth labels for the image; at minimum this must contain bbox coords [tl_x, tl_y, br_x, br_y] for
                 each object and a corresponding class label
        iou_threshold: IoU threshold required to qualify as detected; IoU must be greater than or equal
                       to this value

    Returns:
        A list of len(batch_size) where each element has a list containing:
            1. a numpy array of true_postives (num_preds,); 1 for true positive and 0 for false positive; num_preds is limited to 300 in nms
            2. a tensor of class confidences (num_preds,)
            3. a tensor of the pred labels (num_preds,)
    """
    batch_metrics = []

    # Loop through each batch
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue

        # Extract a single sample prediction
        output = outputs[sample_i]

        # Extract the labels for the sample
        target = targets[sample_i]

        # Extract box preds (xyxy, objectness, class score)
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        # (max_nms_preds,); this is defined by max_nms in non_max_suppression()
        true_positives = np.zeros(pred_boxes.shape[0])

        # annotations = targets[targets[:, 0] == sample_i][:, 1:] # I think the code has its targets as [image_index, class_index, cx, cy, w, h], need to figure out if this is changed to xyxy and if i need to == sample
        annotations = torch.cat(
            [torch.unsqueeze(target["labels"], 1), target["boxes"]], dim=1
        )

        # target_labels = torch.unsqueeze(target["labels"], 1) if len(annotations) else []
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            # Loop through each box prediction
            for pred_i, (pred_box, pred_label) in enumerate(
                zip(pred_boxes, pred_labels)
            ):
                # If all the targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                # Explanation: I think x is a tuple of (index, tensor), where tensor is a single box coord
                #              from target_boxes; x[0] is the index; when target_labels at that index equals
                #              the pred_label then it will return the index and the target_box coord
                filtered_target_position, filtered_targets = zip(
                    *filter(
                        lambda x: target_labels[x[0]] == pred_label,
                        enumerate(target_boxes),
                    )
                )
                # filtered_targets contains only the target boxes where the label matches the predicted label;
                # filtered_target_positions is the indices of the rows in target_boxes for the pred_label

                # iou, box_filtered_index = box_iou_modified(
                #     pred_box.unsqueeze(0), torch.stack(filtered_targets), return_union=False
                # ).max(0)
                # Find the best matching target for our predicted box;
                # predicted box is a single box prediction and filtered targets is 1 or more box labels
                # which allows us to try and match the predicted box with the best overlapping target
                iou, box_filtered_index = bbox_iou(
                    pred_box.unsqueeze(0), torch.stack(filtered_targets)
                ).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min threshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    # if detected, set the true_positives tensor to 1 for that prediction
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])

    return batch_metrics
