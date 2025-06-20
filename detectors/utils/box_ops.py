# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
from typing import List, Tuple

import numpy as np
import torch
from torchvision.ops.boxes import box_area


def xywh2xyxy(boxes: torch.Tensor):
    """Convert boxes in yolo format [cx, cy, w, h] to [tl_x, tl_y, br_x, br_y];
    returns a new tensor, does not modify in place

    Args:
        boxes: a tensor of boxes for each object in an image (num_objects, 4)
               where 4 represents the [cx, cy, w, h] of each object

    Returns:
        A new tensor of the same shape as the the input tensor
    """
    y = boxes.new(boxes.shape)
    y[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    y[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    y[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    y[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return y


def box_cxcywh_to_xyxy(x: torch.Tensor):
    """Converts bboxes from (cx, cy, w, h) (yolo format) to (tl_x, tl_y, br_x, br_y)
    
    Args:
        x: bounding boxes in (cx, cy, w, h) format; shape (num_boxes, 4)
    """
    # unpack the bounding boxes to each variable (num_boxes,)
    x_c, y_c, w, h = x.unbind(-1)

    # convert to (tl_x, tl_y, br_x, br_y)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def bbox_iou(box1, box2, x1y1x2y2=True):
    """Returns the IoUs of a single box (box1) with many candidate boxes (box2)

    For example, box1 can be a single predicted box coords and we want to know which
    ground-truth target box coord overlaps best with the predicted box coord;
    the number of bbox coords in box2 depends on how many target boxes there
    are for the predicted label

    Args:
        box1: (1, 4)
        box2: (b, 4)
        x1y1x2y2: whether bbox coords are in the form (tl_x, tl_y, br_x, br_y)

    Returns:
        TODO: figure this out better
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes (b, )
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle (b,); box-wise max comparison (i.e., does not flatten)
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def rescale_boxes(boxes, current_dim, original_size: tuple[int, int]):
    """TODO comment this
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_size

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_size))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_size))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


# modified from torchvision to also return the union
def box_iou_modified(boxes1, boxes2, return_union=False):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union

    if return_union:
        return iou, union

    return iou


def clip_boxes(boxes, img_shape, xyxy=True):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes the bounding boxes to clip
        shape (tuple): the shape of the image

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped boxes
    """
    if isinstance(
        boxes, torch.Tensor
    ):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


# modified from torchvision to also return the union
def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """Compute the intersection over union (IoU) for a single box in `boxes1`
    with every box in `boxes2`, and repeat this for all boxes in `boxes1`

    NOTE: the number of boxes in `boxes1` and `boxes2` does NOT have to be the same


    Args:
        boxes1: the first set of bboxes to compute the IoU between (num_boxes_1, 4); 
              the box format must be (tl_x, tl_y, br_x, br_y)
        boxes2: the second set of bboxes to compute the IoU between (num_boxes_2, 4); 
              the box format must be (tl_x, tl_y, br_x, br_y)

    Returns:
        the IoU between every box in `boxes1` with every box in `boxes2`;
        shape (num_boxes_1, num_boxes_2)
    """
    # compute the area of each individual box (num_boxes,)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # compute the top_left coordinate of the boxes intersection between a single box in
    # `boxes1` with every box in `boxes2`, and do this for all boxes in `boxes1` 
    # shape (num_boxes_1, num_boxes_2, 2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])

    # repeat the above step but for the bottom_right coordinate of the boxes intersection
    # (num_boxes_1, num_boxes_2, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    # compute the width and height among all box intersection coordinates 
    # (bottom_right - top_left); shape (num_boxes_1, num_boxes_2, 2)
    wh = (rb - lt).clamp(min=0)

    # compute the intersection area by multiplying the width*height of the box
    # intersection (num_boxes_1, num_boxes_2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,

    # compute the union by adding the area of two boxes and subtracting their intersection;
    # again, this is for a single area in `area1` for every area in `area2`
    # shape (num_boxes_1, num_boxes_2)
    union = area1[:, None] + area2 - inter


    # compute the IoU with element-wise division (num_boxes_1, num_boxes_2)
    iou = inter / (union + 1e-6)
    return iou, union




def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Args:
        boxes1: the first set of boxes to compute the giou over; 
                boxes should be in (x0, y0, x1, y1) format (i.e., top_left & bottom_right);
                shape (num_boxes, 4)
        boxes2: the second set of boxes to compute the giou over; 
                boxes should be in (x0, y0, x1, y1) format (i.e., top_left & bottom_right);
                shape (num_boxes, 4)

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    ######### START HERE #######
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


def val_preds_to_img_size(targets, bbox_preds: torch.Tensor, class_conf: torch.Tensor):
    """Scale the bounding box predictions to the original image size

    NOTE: The YoloV4 pytorch implementation resizes the validation/testing images to the size they were trained on (608,608) but
          does not resize the targets. I believe this is okay to do since it will predict the bboxes on the resized image, but
          then we can scale the predictions back to the original image size.

    Args:
        images: Images that were input to the model
        targets: Ground truth labels
        bbox_preds: Bounding box outputs during evaluation
                    (B, scale_1_w*h + scale_2_w*h + scale_3_w*h, 1, 4)
        class_conf: Class confidence predictions, calculated as (objectness * class_confidences)
                    (B, scale_1_w*h + scale_2_w*h + scale_3_w*h, 3)

    Return: TODO
    """
    # TODO
    result = {}
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    for target, boxes, confs in zip(targets, bbox_preds, class_conf):
        img_height, img_width = target["orig_size"].cpu().detach().numpy()
        # boxes = output[...,:4].copy()  # output boxes in yolo format

        boxes = (
            boxes.squeeze(2).cpu().detach().numpy()
        )  # There's 1 length dim in the 2nd dimension so I'm not sure what the squeeze is for

        # Convert from [tl_x, tl_y, br_x, br_y] to [tl_x, tl_y, w, h]
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]

        # Since box predictions are between [0, 1] we can multiply by the input image
        # w/h to scale the predictions back to the original size
        boxes[..., 0] = boxes[..., 0] * img_width
        boxes[..., 1] = boxes[..., 1] * img_height
        boxes[..., 2] = boxes[..., 2] * img_width
        boxes[..., 3] = boxes[..., 3] * img_height
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # confs = output[...,4:].copy()
        confs = confs.cpu().detach().numpy()
        labels = np.argmax(confs, axis=1).flatten()
        labels = torch.as_tensor(labels, dtype=torch.int64)
        scores = np.max(confs, axis=1).flatten()
        scores = torch.as_tensor(scores, dtype=torch.float32)
        result[target["image_id"].item()] = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }

        return result
