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

        # build and preprocess yolo targets
        targ_cls, targ_box, indices, anchors = self._build_targets(preds, targets, model)

        # We use binary cross-entropy and not regular cross-entropy because the classes are not mutually exclusive
        # e.g., some datasets may contains labels that are hierarchical or related, e.g., woman and person;
        #       so each output cell could have more than 1 class to be true; correspondingly, we also apply binary cross-entropy
        #       for each class one by one and sum them up because they are not mutually exclusive.
        bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
        bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
        
        # Calcluate losses for each yolo layer
        for layer_index, layer_predictions in enumerate(preds):
            # extract image_ids, ancors, grid index i/j for each target in the current prediction scale
            image_idx, anchor, grid_j, grid_i = indices[layer_index]
            breakpoint()
            # build empty object target tesnor with shape the same as object prediction
            t_obj = torch.zeroslike(layer_predictions[..., 0], device=self.device) # (b, num_anchors, ny, nx, 5 + num_classes)
            
            # Number of objects in the batch of images
            num_targets = image_idx.shape[0]
            
            # NOTE: each target is a label box with some scaling and the association of an anchor box;
            #       label boxes may be associated to 0 or multiple anchors. So they are multiple times 
            #       or not at all in the targets.
            
            # Check if there's targets for the batch
            if num_targets:
                # load the corresponding
                ################## START HERE
            
            
        

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

            # extract grid h/w from the current prediction scale
            ny, nx = torch.tensor(preds[i].shape)[[2, 3]]

            # extract the yolo grid height and width (ny, nx) to the gain tensor for this layer; NOTE: training pred shape (b, num_anchors, ny, nx, (5+num_classes))
            # gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            gain[2:6] = torch.Tensor([yolo_layer.stride] * 4)

            # scale targets by the grid spatial dims; this will put them in the yolo grid coordinate system
            # NOTE: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/1d621c8489e22c76ceb93bb2397ac6c8dfb5ceb7/pytorchyolo/utils/transforms.py#L56
            #       creates relative labels by dividing the xywh labels by the img size during augmentation;
            #       therefore to scale the labels to the grid size you need to multiply by the number cells;
            #       since we keep the original label sizes in this implementation, we need to divide by the stride;
            #       this should be equivalent
            scaled_targets = targets / gain

            # if targets exist
            if num_targets:
                # calculate the ratio between the scaled targets and anchors
                targ_anch_ratio = scaled_targets[:, :, 4:6] / anchors[:, None, :]

                # select the ratios that have the highest divergence in any axis and check if the ratio is less than 4;
                # inverting the ratio with 1/targ_anch_ratio considers the case where the target w/h is smaller than the anchors
                # by a factor of anchor_t so even if the ratio is less than the threshold it can still be
                # a factor of anchor_t smaller.
                anchor_t = 4  # TODO put this in the config

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

            # Cast to int to get a cell index e.g., 1.2 gets associated to cell 1
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
                    grid_j.clamp_(0, ny.long() - 1),
                    grid_i.clamp_(0, nx.long() - 1),
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
