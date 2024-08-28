from typing import Tuple

import numpy as np
import tqdm


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


def print_eval_stats(metrics_output, class_names, verbose=True):
    """TODO

    Args:

    """
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print((ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")
