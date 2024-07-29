# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
from typing import List, Tuple

import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def get_region_boxes(boxes_and_confs):
    """Unpack bboxes and confidences and combine into 1 tensor of boxes
    and 1 tensor of confidences.

    Args:
        boxes_and_confs: bboxes and cls confidences from YoloLayer

    Returns:
        1. Tensor of bboxes (B, num_pred_1 + num_pred_2 + num_pred_3, 1, 4)
        2. Tensor of confidences (B, num_pred_1 + num_pred_2 + num_pred_3, num_classes)
    """

    boxes_list = []
    confs_list = []

    for bbox, confidence in boxes_and_confs:
        boxes_list.append(bbox)
        confs_list.append(confidence)

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)

    return (boxes, confs)


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = (cw <= 0) + (ch <= 0) > 0
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def get_region_boxes(boxes_and_confs: List[Tuple]):
    """Extracts the boxes and confidences from different scales and concatenates
    along the flattened cell dimension; since the out_h and out_w are flattened, we're
    able to concat them even though their predictions are at different dimensions.

    Args:
        boxes_and_confidences: list of bbox coordinate predictions and confidences for each output scale;
                               the bbox preds have been flattened so boxes shape (B, out_h*out_w, 1, 4);
                               this comes from the YoloLayer during inference;
                               YoloV4 has 3 output scales so len(boxes_and_confs) = 3
    """
    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, n_preds_scale1 + n_preds_scale2 + n_preds_scale3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)

    return [boxes, confs]


def val_preds_to_img_size(image: torch.Tensor, bbox_preds, targets):
    """Scale the bounding box predictions to the original image size
    
    Args:
        image: Input image to the model 
        bbox_preds: Bounding box outputs during evaluation
        targets: Ground truth labels
    """
    # TODO
    for img, target, boxes, confs in zip(image, targets, outputs[0], outputs[1]):
        img_height, img_width = img.shape[:2]
        # boxes = output[...,:4].copy()  # output boxes in yolo format
        boxes = boxes.squeeze(2).cpu().detach().numpy()
        boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
        boxes[...,0] = boxes[...,0]*img_width
        boxes[...,1] = boxes[...,1]*img_height
        boxes[...,2] = boxes[...,2]*img_width
        boxes[...,3] = boxes[...,3]*img_height
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # confs = output[...,4:].copy()
        confs = confs.cpu().detach().numpy()
        labels = np.argmax(confs, axis=1).flatten()
        labels = torch.as_tensor(labels, dtype=torch.int64)
        scores = np.max(confs, axis=1).flatten()
        scores = torch.as_tensor(scores, dtype=torch.float32)
        res[target["image_id"].item()] = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }