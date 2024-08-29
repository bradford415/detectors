import logging
from typing import Tuple, List, Optional

import numpy as np
import tqdm

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
        TODO
        The average precision as computed in py-faster-rcnn.
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


def print_eval_stats(metrics_output: Optional[Tuple[np.ndarray, ...]], class_names: List[int], verbose: bool=True):
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
        #breakpoint()
        if verbose:
            # Prints class AP and mean AP
            for index, cls_num in enumerate(ap_class):
                log.info("%s: %-15.2d %s =  %-15s %s = %-15.4f", "index", index, "class", class_names[cls_num], "AP", AP[index])
                #ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            #print(ap_table)#.table)
        log.info("---- mAP %.5f} ----", AP.mean())
    else:
        print("---- mAP not measured (no detections found by model) ----")
        
        

            # log.info("\ntrain\t%-10s =  %-15.4f", "AP", bbox_stats[0])
            # log.info("train\t%-10s =  %-15.4f", "AP50", bbox_stats[1])
            # log.info("train\t%-10s =  %-15.4f", "AP75", bbox_stats[2])
            # log.info("train\t%-10s =  %-15.4f", "AP_small", bbox_stats[3])
            # log.info("train\t%-10s =  %-15.4f", "AP_medium", bbox_stats[4])
            # log.info("train\t%-10s =  %-15.4f", "AP_large", bbox_stats[5])
            # log.info("train\t%-10s =  %-15.4f", "AR1", bbox_stats[6])
            # log.info("train\t%-10s =  %-15.4f", "AR10", bbox_stats[7])
            # log.info("train\t%-10s =  %-15.4f", "AR100", bbox_stats[8])
            # log.info("train\t%-10s =  %-15.4f", "AR_small", bbox_stats[9])
            # log.info("train\t%-10s =  %-15.4f", "AR_medium", bbox_stats[10])
            # log.info("train\t%-10s =  %-15.4f", "AR_large", bbox_stats[11])