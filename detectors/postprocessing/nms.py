import time
from typing import List

import torch
import torchvision

from detectors.utils.box_ops import cxcywh_to_xyxy


def non_max_suppression(
    predictions, conf_thres=0.25, iou_thres=0.45, classes=None
) -> List[torch.tensor]:
    """Performs Non-Maximum Suppression (NMS) on inference results

    Args:
        predictions: Model output predictions (B, num_preds, num_class+5); num_preds is the number of
                     predictions across across all output scales; predictions from yolo are (cx, cy, w, h)
                     and are converted to (tl_x, tl_y, br_x, br_y) in this function

    Returns:
        A list of tensors where each element is the nms predictions for an image;
        the length of the output is the batch_size and each element has shape (max_nms, 6)
        where 6 = (tl_x, tl_y, br_x, br_y, conf, cls)
    """

    assert len(predictions.shape) == 3

    nc = predictions.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096  # 64*64
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device=predictions.device)] * predictions.shape[0]

    for image_index, box_pred in enumerate(predictions):
        # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height

        # This thresholds the prediction by their objectness; commenting this out for now
        box_pred = box_pred[box_pred[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not box_pred.shape[0]:
            continue

        # Compute confidences by multiplying the objectness score and the class score
        box_pred[:, 5:] *= box_pred[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = cxcywh_to_xyxy(box_pred[:, :4])

        # Filter out predictions by conf_threshold in the multi_label case and by the maximum class confidence in the single label case
        if multi_label:
            # Indices where the class_conf > conf_threshold (2, num_preds*(num_confs>0));
            # pred_index is the index of the pred corresponding to the class and class_index, each have shape (num_preds*(num_confs>0),)
            pred_index, class_index = (
                (box_pred[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            )
            box_pred = torch.cat(
                (
                    box[pred_index],
                    box_pred[pred_index, class_index + 5, None],
                    class_index[:, None].float(),
                ),
                1,
            )
        else:  # best class only
            # Extract the maximum class confidence and index
            max_conf, max_indices = box_pred[:, 5:].max(axis=1, keepdim=True)
            box_pred = torch.cat((box, max_conf, max_indices.float()), 1)[
                max_conf.view(-1) > conf_thres
            ]
        # box_pred shape if multi_label (num_preds*(num_confs>0), 6) where 6 is (box_coords, class_confidence, class_label)
        # box_pred shape if single label (num_preds, 6)

        # Filter by class
        if classes is not None:
            box_pred = box_pred[
                (box_pred[:, 5:6] == torch.tensor(classes, device=box_pred.device)).any(
                    1
                )
            ]

        # Check shape
        num_boxes = box_pred.shape[0]
        if not num_boxes:  # no boxes
            continue
        elif num_boxes > max_nms:  # excess boxes
            # sort by confidence
            box_pred = box_pred[box_pred[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = box_pred[:, 5:6] * max_wh  # classes

        # boxes (offset by class), scores; offset explained here https://github.com/ultralytics/yolov5/discussions/5825#discussioncomment-1720852
        boxes, scores = box_pred[:, :4] + c, box_pred[:, 4]

        # ops.nms returns indices of the elements that have been kept by NMS, sorted in decreasing order of scores
        nms_indices = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        # Limit detections if there are more than max_det detection
        if nms_indices.shape[0] > max_det:
            nms_indices = nms_indices[:max_det]

        ########### TODO: Should probably change the yolo layer to not chagne bbox format and return the full tensor, not objectness*cls_conf separately

        output[image_index] = box_pred[nms_indices]  # .detach().cpu()

        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output
