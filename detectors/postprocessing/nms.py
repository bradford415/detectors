import time

import torch
import torchvision


def non_max_suppression(bbox_preds, objectness, cls_confs, conf_thres=0.25, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = bbox_preds.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    breakpoint()
    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * bbox_preds.shape[0]

    for image_index, (box_pred, cls_conf) in enumerate(zip(bbox_preds, cls_confs)):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        
        # This thresholds the prediction by their objectness; commenting this out for now
        #x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        # if not x.shape[0]:
        #     continue

        # Compute conf; already have the confidences so commenting this out
        #x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2); boxes are already in xyxy so commenting this out
        #box = xywh2xyxy(x[:, :4])
        box = box_pred.clone()

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (box_pred[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            box_pred = torch.cat((box[i], box_pred[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            max_conf, max_indices = cls_conf.max(axis=1, keepdim=True)
            box_pred = torch.cat((box, max_conf, max_indices.float()), 1)[max_conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            box_pred = box_pred[(box_pred[:, 5:6] == torch.tensor(classes, device=box_pred.device)).any(1)]

        # Check shape
        num_boxes = box_pred.shape[0]
        if not num_boxes:  # no boxes
            continue
        elif num_boxes > max_nms:  # excess boxes
            # sort by confidence
            box_pred = box_pred[box_pred[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        breakpoint()
        c = cls_conf[:, 0:1] * max_wh  # classes
    
        # boxes (offset by class), scores; offset explained here https://github.com/ultralytics/yolov5/discussions/5825#discussioncomment-1720852
        boxes, scores = box_pred[:, :4] + c, box_pred[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        ########### TODO: Should probably change the yolo layer to not chagne bbox format and return the full tensor, not objectness*cls_conf separately

        output[image_index] = box_pred[i].detach().cpu()

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output