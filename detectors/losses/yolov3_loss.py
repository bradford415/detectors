import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Yolov3Loss(nn.Module):
    """TODO"""

    def __init__(self, num_anchors: int, device):
        """Initialize the yolov3 loss function

        Args:
            TODO
            num_anchors
        """
        super().__init__()
        self.device = device
        self.num_anchors = num_anchors

    def forward(self, preds, targets, model):
        """Compute the yolov3 loss

        Args:
            preds: list of output predictions at all 3 scales during training (b, num_anchors, ny, nx, 5 + num_classes)
                   where 5 = (cx, cy, w, h, objectness)
            targets: targets for the batch of images (num_objects*b, 6) where 6 = (batch_sample_index, obj_id, cx, cy, w, h)
        """
        # Add placeholder varables for the different losses;
        lbox, lobj, lcls = (
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
        )

        # build and preprocess yolo targets
        targ_cls, targ_box, indices, anchors = self._build_targets(
            preds, targets, model
        )

        #breakpoint()

        # We use binary cross-entropy and not regular cross-entropy because the classes are not mutually exclusive
        # e.g., some datasets may contains labels that are hierarchical or related, e.g., woman and person;
        #       so each output cell could have more than 1 class to be true; correspondingly, we also apply binary cross-entropy
        #       for each class one by one and sum them up because they are not mutually exclusive.
        bce_cls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0], device=self.device)
        )
        bce_obj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0], device=self.device)
        )

        # Calcluate losses for each yolo layer
        for layer_index, layer_predictions in enumerate(preds):
            # extract image_ids, ancors, grid index i/j for each target in the current prediction scale; (num_objects, )
            image_idx, anchor, grid_j, grid_i = indices[layer_index]

            # build empty object target tesnor with shape the same as object prediction
            t_obj = torch.zeros_like(
                layer_predictions[..., 0], device=self.device
            )  # (b, num_anchors, ny, nx, 5 + num_classes)

            # Number of objects in the batch of images
            num_targets = image_idx.shape[0]

            # NOTE: each target is a label box with some scaling and the association of an anchor box;
            #       label boxes may be associated to 0 or multiple anchors. So they are multiple times
            #       or not at all in the targets.

            # Check if there's targets for the batch
            if num_targets:
                # using the processed targets, load the corresponding values from the predictions
                # (num_objects, 5 + num_classes);
                # NOTE: this could extract the same prediction twice just with a different anchor box; this is normal
                ps = layer_predictions[image_idx, anchor, grid_j, grid_i]

                # apply sigmoid to bound cx,cy predctions to [0, 1] so they can be used as cell offsets
                p_xy = ps[:, :2].sigmoid()

                # apply e to wh predictions and multiply with the anchor box that matched best with the label
                # for each cell that has a target; NOTE: the anchors were scaled by the stride in _build_targets()
                p_wh = torch.exp(ps[:, 2:4]) * anchors[layer_index]

                # build box from scaled predictions; (num_objects, 4)
                pbox = torch.cat((p_xy, p_wh), 1)

                # Calculate CIoU or GIoU for each target with te predicted box for its cell + anchor
                iou = bbox_iou_loss(
                    pbox.T, targ_box[layer_index], x1y1x2y2=False, CIoU=True
                )

                # We want to minimize our loss and the best posshible IoU is 1 so we take 1 - IoU and reduce it with a mean
                lbox += (
                    1.0 - iou
                ).mean()  # iou loss # TODO put shape though it might be a scalar

                # Classification of the objectness
                # Fill our empty object target tensor with the IoU we just calculated for each target at the targets position
                t_obj[image_idx, anchor, grid_j, grid_i] = (
                    iou.detach().clamp(0).type(t_obj.dtype)
                )  # Use cells with iou > 0 as object targets

                # Classification of the class
                # Check if we need to do a classification (number of classes > 1)
                if ps.size(1) - 5 > 1:
                    # create one-hot class encoding (num_objects, num_class)
                    targ_cls_onehot = torch.zeros_like(
                        ps[:, 5:], device=self.device
                    )  # targets
                    targ_cls_onehot[range(num_targets), targ_cls[layer_index]] = 1

                    # Use the tensor to calculate the BCE loss
                    lcls += bce_cls(ps[:, 5:], targ_cls_onehot)  # BCE

            # Classification of the objectness the sequel
            # Calculate the BCE loss between the on the fly generated target and the network prediction
            # both params have shape (b, num_anchors, ny, nx); t_obj is 0 everywhere except the positions
            # defined by t_obj[image_idx, anchor, grid_j, grid_i]
            lobj += bce_obj(layer_predictions[..., 4], t_obj)  # obj loss

        # weight each of the loss components
        lbox *= 0.05
        lobj *= 1.0
        lcls *= 0.5

        # Merge losses
        loss = lbox + lobj + lcls

        return loss, torch.cat((lbox, lobj, lcls, loss)).detach().cpu()

    def _build_targets(
        self, preds, targets, model
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[tuple[torch.Tensor]],
        list[torch.Tensor],
    ]:
        """

        Args:
            preds:
            targets: (num_objs, 6) where 6 = (batch_sample_index, obj_id, cx, cy, w, h);
                     bbox coords are expected be relative to the img_size after augmentation;
                     i.e., they should not be scaled

        Returns:
            For each prediction scale in a list:
                1. class ids for the targets lower than the threshold (num_objects_thresholded,)
                2. scaled bbox coords (num_objects_thresholded, 4) where 4 = (cx,cy,w,h) where cx,cy are
                   relative to a single cell [0,1] and w,h are relative to the entire grid
                3. a tuple of indices for: the image_index in the batch, the anchor_index of the obj,
                                           the y coord of the cell, the x coord of the cell;
                                           each element of the tuple is (num_objects,)
                4. the scaled anchors (num_objects, 2)
        """
        num_targets = targets.shape[0]

        targ_cls, targ_box, indices, anch = [], [], [], []

        # TODO: also write the purpose of gain; a tensor used as a placeholder to store the targets (img id, class, x, y, w, h, anchor id)
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain

        # tensor that iterates 0-2 along rows and repeats along columns (num_anchors, num_targets)
        anchor_index = (
            torch.arange(self.num_anchors, device=targets.device)
            .float()
            .view(self.num_anchors, 1)
            .repeat(1, num_targets)
        )

        # Copy target boxes num_anchors times (num_anchors, batch_num_objects, 6) and append an anchor index
        # to each copy (num_anchors, batch_num_objects, 7)
        targets = torch.cat(
            (targets.repeat(self.num_anchors, 1, 1), anchor_index[:, :, None]), 2
        )

        # loop through each prediction scale
        for i, yolo_layer in enumerate(model.yolo_layers):
            # Scale anchors by the yolo grid cell size so that an anchor with the size of the cell would result in 1
            anchors = yolo_layer.anchors / yolo_layer.stride

            # extract grid h/w from the current prediction scale
            ny, nx = torch.tensor(preds[i].shape)[[2, 3]]

            # extract the yolo grid height and width (ny, nx) to the gain tensor for this layer; NOTE: training pred shape (b, num_anchors, ny, nx, (5+num_classes))
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            #gain[2:6] = torch.Tensor([yolo_layer.stride] * 4)

            # scale targets by the grid spatial dims; this will put them in the yolo grid coordinate system
            # NOTE: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/1d621c8489e22c76ceb93bb2397ac6c8dfb5ceb7/pytorchyolo/utils/transforms.py#L56
            #       creates relative labels by dividing the xywh labels by the img size during augmentation;
            #       therefore to scale the labels to the grid size you need to multiply by the number cells;
            #       since we keep the original label sizes in this implementation, we need to divide by the stride;
            #       this should be equivalent
            #scaled_targets = targets / gain
            scaled_targets = targets * gain

            # if targets exist
            if num_targets:
                # calculate the ratio between the scaled targets and anchors (wh)
                targ_anch_ratio = scaled_targets[:, :, 4:6] / anchors[:, None]

                # select the ratios that have the highest divergence in any axis and check if the ratio is less than 4;
                # inverting the ratio with 1/targ_anch_ratio considers the case where the target w/h is smaller than the anchors
                # by a factor of anchor_t so even if the ratio is less than the threshold it can still be
                # a factor of anchor_t smaller.
                anchor_t = 4.0  # TODO put this in the config

                ratio_thresh = (
                    torch.max(targ_anch_ratio, 1.0 / targ_anch_ratio).max(dim=2)[0]
                    < anchor_t
                )

                # Only use targets that have the correct ratios for their anchors
                # NOTE: even though this operation gets rid of the anchor_index dim,
                #       the anchor_index is still saved in the 7th position
                scaled_targets = scaled_targets[ratio_thresh]  # (num_under_thresh, 7)
            else:
                scaled_targets = targets[0]

            # Extract image id in the batch and class id
            image_index, class_id = scaled_targets[:, :2].long().T

            # extract cx, cy, w, h; these are already in the cell coordinate system meaning an x = 1.2 would be 1.2 times cellwidth
            grid_xy = scaled_targets[:, 2:4]  # grid cx, cy
            grid_wh = scaled_targets[:, 4:6]  # grid wh

            # Cast to int to get a cell index; e.g., 1.2 gets associated to cell 1
            # NOTE: yolov3 in ultralyitcs duplicates the targets by 5 and creates 4 offsets (-0.5 in every direction); this is not implemented here
            grid_ij = grid_xy.long()

            # Isolate x and y index dimensions
            grid_i, grid_j = grid_ij.T  # grid xy indices

            # Convert anchor indices to int
            anch_inds = scaled_targets[:, 6].long()

            # Store target tensors for this yolo layer to the output lists; clamp grid_inds to min/max range to prevent out of bounds
            indices.append(
                (
                    image_index,
                    anch_inds,
                    grid_j.clamp_(0, gain[3].long() - 1),
                    grid_i.clamp_(0, gain[2].long() - 1),
                )
            )

            # Store target box list and convert box coordinates from global grid coordinates to local offsets in the grid cell
            # e.g., [13.5, 8.2] - [13.0, 8.0] = [0.5, 0.2]; grid_wh is kept relative to grid size
            targ_box.append(torch.cat((grid_xy - grid_ij, grid_wh), 1))  # box ()

            # Store correct anchor for each target to the list (num_under_thresh, 4)
            anch.append(anchors[anch_inds])

            # Store class_id for each target to the list
            targ_cls.append(class_id)

        # The processed targets for all prediction scales
        return targ_cls, targ_box, indices, anch


def bbox_iou_loss(
    box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9
):
    """IoU between bounding boxes box1 and box2 used in the loss function

    # TODO: should go through this function and comment it

    NOTE: there's another bbox_iou in detectors.utils.box_ops that's used to get_batch_statistics after nms;
          I assume they're equivalent but have not stepped through the code yet

    Args:
        box1: typically the predicted boxes (4, num_objects)
        box2: typically the target boxes (num_objects, 4)
        x1y1x2y2: whether bounding boxes are in the format (tl_x, tl_y, br_x, br_y); if false
                  assume (cx, cy, w, h) form
    """
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    assert box1.shape[-1] == box2.shape[-1]

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU