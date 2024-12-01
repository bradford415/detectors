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
            TODO
            targets: targets for the batch of images (num_objects*b, 6) where 6 = (batch_sample_index, obj_id, cx, cy, w, h)
        """
        # Add placeholder varables for the different losses;
        lbox, lobj, lcls = (
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
        )

        # build yolo targets; TODO: clarify this
        self._build_targets(preds, targets, model)

    def _build_targets(self, preds, targets, model):
        """TODO

        Args:
            preds:
            targets: (num_objs, 6) where 6 = (sample_index, obj_id, cx, cy, w, h);
                     bbox coords are expected be relative to the img_size after augmentation;
                     i.e., they should not be scaled

        """
        num_targets = targets.shape[0]

        tcls, tbox, indices, anch = [], [], [], []

        # a tensor used as a placeholder to store the targets (img id, class, x, y, w, h, anchor id)
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

            # extract the yolo grid height and width (ny, nx) to the gain tensor for this layer; NOTE: training pred shape (b, num_anchors, ny, nx, (5+num_classes))
            # gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            gain[2:6] = torch.Tensor([yolo_layer.stride] * 4)

            # scale targets by the grid spatial dims; this will put them in the yolo grid coordinate system
            # NOTE: the github code creates relative labels by dividing the xywh labels by the img size during augmentation;
            #       therefore to scale the labels to the grid size you need to multiply by the number cells;
            #       since we keep the original label sizes in this implementation, we need to divide by the stride;
            #       this should be equivalent
            #
            scaled_targets = targets / gain
            breakpoint()
            # if targets existc
            if num_targets:
                # calculate the ratio between the scaled targets and anchors
                targ_anch_ratio = scaled_targets[:, :, 4:6] / anchors[:, None, :]

                # select the ratios that have the highest divergence in any axis and check if the ratio is less than 4;
                # inverting the ratio with 1/targ_anch_ratio considers the case where the target w/h is smaller than the anchors
                # by a factor of anchor_t so even if the ratio is less than the threshold it can still be
                # a factor of anchor_t smaller.

                anchor_t = 4 # TODO put this in the config
                ratio_thresh = (
                    torch.max(targ_anch_ratio, 1.0 / targ_anch_ratio).max(dim=2)[0] < anchor_t
                )  # compare #TODO

                # Only use targets that have the correct ratios for their anchors
                breakpoint()
                scaled_targets = scaled_targets[ratio_thresh] # (num_under_thresh, 7)
                # NOTE: even though this operation gets rid of the anchor_index dim, 
                #       the anchor_index is still saved in the 7th position
            else:
                scaled_targets = targets[0]
            print("GHer")
