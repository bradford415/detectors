import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

## TODO: implement the yolov3 version for better readability


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 512  # 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [
            [12, 16],
            [19, 36],
            [40, 28],
            [36, 75],
            [76, 55],
            [72, 146],
            [142, 110],
            [192, 243],
            [459, 401],
        ]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        (
            self.masked_anchors,
            self.ref_anchors,
            self.grid_x,
            self.grid_y,
            self.anchor_w,
            self.anchor_h,
        ) = ([], [], [], [], [], [])

        for i in range(3):
            all_anchors_grid = [
                (w / self.strides[i], h / self.strides[i]) for w, h in self.anchors
            ]
            masked_anchors = np.array(
                [all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32
            )
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]

            grid_x = (
                torch.arange(fsize, dtype=torch.float)
                .repeat(batch, 3, fsize, 1)
                .to(device)
            )
            grid_y = (
                torch.arange(fsize, dtype=torch.float)
                .repeat(batch, 3, fsize, 1)
                .permute(0, 1, 3, 2)
                .to(device)
            )
            anchor_w = (
                torch.from_numpy(masked_anchors[:, 0])
                .repeat(batch, fsize, fsize, 1)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            anchor_h = (
                torch.from_numpy(masked_anchors[:, 1])
                .repeat(batch, fsize, fsize, 1)
                .permute(0, 3, 1, 2)
                .to(device)
            )

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(
            batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes
        ).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(
            device=self.device
        )
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(
            self.device
        )
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(
            self.device
        )

        # labels = labels.cpu().data
        # nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        nlabel = max([len(label["boxes"]) for label in labels])  # number of objects

        gt_labels = torch.zeros(
            (pred.shape[0], nlabel, 5)
        )  # (B, batch_max_objs, 5); 5 = (cx, cy, w, h, obj_id)

        # Extract the ground truth bbox and obj_id and store in gt_labels tensor
        for index, label in enumerate(labels):
            gt_labels[index, : len(label["boxes"]), :] = torch.cat(
                (label["boxes"], label["labels"].unsqueeze(1)), dim=1
            )

        labels = gt_labels

        n_objs = (gt_labels.sum(dim=2) > 0).sum(dim=1)

        # truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        # truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)

        truth_x_all = (labels[:, :, 0]) / (self.strides[output_id])
        truth_y_all = (labels[:, :, 1]) / (self.strides[output_id])

        # truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        # truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]

        truth_w_all = (labels[:, :, 2]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3]) / self.strides[output_id]

        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(n_objs[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(
                truth_box.cpu(), self.ref_anchors[output_id], CIoU=True
            )

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = (
                (best_n_all == self.anch_masks[output_id][0])
                | (best_n_all == self.anch_masks[output_id][1])
                | (best_n_all == self.anch_masks[output_id][2])
            )

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].reshape(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > self.ignore_thre
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(
                        torch.int16
                    ).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(
                        torch.int16
                    ).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti]
                        / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0]
                        + 1e-16
                    )
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti]
                        / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1]
                        + 1e-16
                    )
                    target[b, a, j, i, 4] = 1
                    target[
                        b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()
                    ] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize
                    )
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
                output[..., np.r_[:2, 4:n_ch]]
            )

            pred = output[..., :4].clone()

            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(
                pred, labels, batchsize, fsize, n_ch, output_id
            )

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(
                input=output[..., :2],
                target=target[..., :2],
                weight=tgt_scale * tgt_scale,
                reduction="sum",
            )
            loss_wh += (
                F.mse_loss(
                    input=output[..., 2:4], target=target[..., 2:4], reduction="sum"
                )
                / 2
            )
            loss_obj += F.binary_cross_entropy(
                input=output[..., 4], target=target[..., 4], reduction="sum"
            )
            loss_cls += F.binary_cross_entropy(
                input=output[..., 5:], target=target[..., 5:], reduction="sum"
            )
            loss_l2 += F.mse_loss(input=output, target=target, reduction="sum")

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


class YoloV4Loss(nn.Module):
    # Maybe TODO go back through and comment and change functions similar to https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/loss.py
    #      because it's commented very well
    """Loss function for YoloV4

    The loss function loops through all predictions for each grid cell.
    There will be num_anchors predictions per grid cell for a total of num_anchors*H*W predictions.
    The loss function is only calcuated for predictions which contain an object (objectness)
    and

    The original Yolo loss function is described in the original paper in section 2.2
    https://arxiv.org/pdf/1506.02640.pdf

    Addtional info about the Yolo Loss
    1. Yolo loss explained
        - https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation
    2. How the original Yolo loss differs from YoloV4 loss
        - https://stackoverflow.com/questions/68892124/whats-the-complete-loss-function-used-by-yolov4


    ########### NEED TO READ THROUGH THE CODE MORE AND UNDERSTAND THIS BETTER
    MY UNDERSTANDING OF J MAY BE INCORRECT BUT I THINK IT'S RIGHT
    ##################

    YoloV4 loss function is composed of 5 terms which are calculated
    for each grid cell and summed. The terms are explained below:
        Note and TODO: This is technically the Yolo loss, the YoloV4 loss slightly differs so I need to update this eventually
        1. Squared sum error of x, y center point predictions if an object exists
           AND the jth bounding box has the highest confidence score of all
           predictors in that grid cell (num_predictors=num_anchors);
           0 if no object exists or the predictor does not have the highest
           confidence in that grid cell (highest confidence is the highest class probablity);
           whether an object exists or not is determined by the ground truth object center,
           not the objectness prediction score
        2. Sqrt sum error of w, h predictions, following the same conditions as
           the first term
        3. Squared error of objectness score, following the same conditions as the first term
        4. Squared error of objectness score if NO object exists
           (penalizes for predicting objects when they don't exist)
        5. If any object is predicted in the cell (for this term we don't care about which predictor, j,
           says an object exists) find the squared error of the predicted classes confidence probability
           and the ground-truth label confidence probability
    """

    def __init__(self, anchors, batch_size, n_classes=80, n_anchors=3, device=None):
        """TODO

        Args:
            n_classes:
        """
        super().__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 512
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [
            [12, 16],
            [19, 36],
            [40, 28],
            [36, 75],
            [76, 55],
            [72, 146],
            [142, 110],
            [192, 243],
            [459, 401],
        ]
        self.anchors_temp = anchors
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        # If the IoU between the prediction and the ground truth is greater than this threshold,
        # exclude that calculation from the loss; the reason for this is because including too many
        # predictions in the loss  could cause over or underfitting; if the ignore threshold is too high, close to 1,
        # then there will be a lot of predictions used to calcuate the loss and could  overfitting;
        # if the ignore threshold is low, not many predictions will be used to calcuate the loss and the model might be underfit
        # This is also specified in section 2.1 of the YoloV2 paper
        self.iou_ignore_threshold = 0.5

        # masked_anchors:
        # ref_anchors: 2D list of all anchors divided by the output stride;
        #              need to scale the anchors by how much the output feature map was downsampled by
        (
            self.masked_anchors,
            self.ref_anchors,
            self.grid_x,
            self.grid_y,
            self.anchor_w,
            self.anchor_h,
        ) = ([], [], [], [], [], [])

        # Some code I started to write to see if I could clean up this for loop; it's probably not worth it because I might mess something up
        # Leaving it here in case I come back to it
        # _anchors, _strides = np.array(self.anchors_temp).reshape(-1, 2), np.array(self.strides).repeat(self.n_anchors)[:, np.newaxis]
        # scaled_anchors = _anchors / _strides
        # _ref_anchors = np.zeros((_anchors.shape[0], 4), dtype=np.float32)
        # _ref_anchors[:, 2:] = scaled_anchors
        # ref_anchors = torch.from_numpy(_ref_anchors)

        for i in range(self.n_anchors):
            all_anchors_grid = [
                (w / self.strides[i], h / self.strides[i]) for w, h in self.anchors
            ]
            masked_anchors = np.array(
                [all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32
            )
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            feature_map_sizes = image_size // self.strides[i]

            # Create x & y grid of feature map pixel indices (B, num_anchors, H/stride, W/stride)
            grid_x = (
                torch.arange(feature_map_sizes, dtype=torch.float)
                .repeat(self.batch_size, 3, feature_map_sizes, 1)
                .to(device)
            )
            grid_y = (
                torch.arange(feature_map_sizes, dtype=torch.float)
                .repeat(self.batch_size, 3, feature_map_sizes, 1)
                .permute(0, 1, 3, 2)
                .to(device)
            )

            # Create grid of the w & h of each anchor box w/ size of the feature maps (B, num_anchors, H/stride, W/stride)
            anchor_w = (
                torch.from_numpy(masked_anchors[:, 0])
                .repeat(self.batch_size, feature_map_sizes, feature_map_sizes, 1)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            anchor_h = (
                torch.from_numpy(masked_anchors[:, 1])
                .repeat(self.batch_size, feature_map_sizes, feature_map_sizes, 1)
                .permute(0, 3, 1, 2)
                .to(device)
            )

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_ground_truth_tensors(
        self,
        pred: list[torch.Tensor],
        labels: list[dict],
        f_map_size,
        num_pred_ch,
        output_id,
    ):
        """The main goal of the function is to build the intermediate ground truth tensor (target) for the loss function.

        The reason this is the intermediate gt tensor is because we will still need to multiply by binary masks to only compute the
        loss on the desired cells. This function also builds these aforementioned masks.
        This gt tensor will be similar shape to the output predictions of the network but for the ground truth labels.

        This function loops through a batch of predictions at each feature map scale. That is, the pred tensor
        height and width corresponds to an output scale and contains a batch of images.

        Args:
            pred: x, y, w, h predictions after adding the grid offsets to tx, ty and scaling the anchors' w, h;
                  (B, n_anchors, H, W, bbox_pred_coords)
            labels: Labels scaled labels for all images in the batch;
                    includes a lot of information but moset notably: bbox_coords, class labels, etc...
            f_map_size: Feature map dimensions at specific output; YoloV4 has 3 different outputs each at a different resolution
            num_pred_ch: Number of predictions per cell; 5 + num_classes
            output_id: The index of the output feature map scale; since YoloV4 has 3 outputs, the possible indices are [0, 1, 2]

        Return:
            TODO
        """
        # Note: tgt_mask is initalized to zeros and obj_mask is intialized to ones because
        #       only the objectness score is calculated in the loss for every prediction,
        #       while only the bbox and class label predictions that correspond with the best anchor are used

        # Binary target mask to 0 out predictions which do not correspond to a best anchor;
        # only 1 anchor corresponds to an object;
        # From the YoloV3 paper section 2.1 it mentions:
        #   "Unlike faster r-cnn our system only assigns one bounding box prior for each ground truth
        #    object. If a bounding box prior is not assigned to a ground
        #    truth object it incurs no loss for coordinate or class predictions, only objectness."

        tgt_mask = torch.zeros(
            self.batch_size, self.n_anchors, f_map_size, f_map_size, 4 + self.n_classes
        ).to(device=self.device)

        # Binary object mask used to set the predictions objectess score (index 4) to 0 if it surpasses the self.iou_ignore_threshold;
        # (B, num_cell_preds, out_H, out_W);
        obj_mask = torch.ones(
            self.batch_size, self.n_anchors, f_map_size, f_map_size
        ).to(device=self.device)

        # TODO: comment this once I know what it does (B, num_anchors,  out_h, out_w)
        tgt_scale = torch.zeros(
            self.batch_size, self.n_anchors, f_map_size, f_map_size, 2
        ).to(self.device)

        # Tensor to store the ground truth t_x, t_y, t_w, t_h, t_o and the class label (a 1 in the class position)
        target = torch.zeros(
            self.batch_size, self.n_anchors, f_map_size, f_map_size, num_pred_ch
        ).to(self.device)

        # Get the maximum number of objects in an image for the entire batch;
        # allows us to create a tensor for the ground truth bboxes and object ids for the batch
        batch_max_objects = max([len(label["boxes"]) for label in labels])
        gt_labels = torch.zeros(
            (pred.shape[0], batch_max_objects, 5)
        )  # (B, batch_max_objs, 5); 5 = (cx, cy, w, h, obj_id)

        # Extract the ground truth bbox and obj_id and store in gt_labels tensor
        for index, label in enumerate(labels):
            gt_labels[index, : len(label["boxes"]), :] = torch.cat(
                (label["boxes"], label["labels"].unsqueeze(1)), dim=1
            )

        # labels = labels.cpu().data

        # Note on how labels is formed (my interpretation from the repo im basing yolov4 off of):
        #   1. labels in github code is (B, max_gt_bboxes, 5), containing the augmented ground truth labels (data augmentation)
        #      (tl_x, tl_y, br_x, br_y, class_id)
        #   2. if max_gt_bboxes is 60, but an image only has 6 bounding boxes, only the first 6 rows will have values,
        #      the rest will be 0s
        #   3. The reason it is hardcoded at 60 is because each image will have a different number of bounding boxes,
        #      but to batch the labels together in a tensor they have to be the same shape
        #   4. Idk if this is the best way to do it but I think it only breaks if there are more than max_gt_bboxes in an image

        # Number of objects per image
        n_objs = (gt_labels.sum(dim=2) > 0).sum(dim=1)

        # Resize ground truth boxes to match the output dimensions by divding by the stride; stride = input_dims/final_output_dims
        # The starting bbox labels are in coco format [tl_x, tl_y, w, h], then data.coco_utisl.PreprocessCoco() converts
        # the bbox labels [tl_x, tl_y, br_x, br_y], and finally data.transforms.Normalize()
        # converts to bbox labels to yolo format ([cx, cy, w, h]), so we do not need to calcuate that here
        scaled_truth_cx_all = gt_labels[:, :, 0] / self.strides[output_id]
        scaled_truth_cy_all = gt_labels[:, :, 1] / self.strides[output_id]
        scaled_truth_w_all = gt_labels[:, :, 2] / self.strides[output_id]
        scaled_truth_h_all = gt_labels[:, :, 3] / self.strides[output_id]

        truth_cell_x_all = scaled_truth_cx_all.to(torch.int16).cpu().numpy()
        truth_cell_y_all = scaled_truth_cy_all.to(torch.int16).cpu().numpy()

        # TODO: This comment is might be wrong I think, need to relook into it; I think its grabbing the best 3 anchor boxes that match with the ground truth
        # Loop through each image in the batch
        for batch in range(self.batch_size):
            num_objs_img = int(n_objs[batch])

            # skip image if no labels in image
            if num_objs_img == 0:
                continue

            # Extract ground truth w, h for the current batch; this is intialized with size 4 because the x, y will be filled in later
            truth_box_cxcywh = torch.zeros(num_objs_img, 4).to(
                self.device
            )  # (num_gt_objects, 4)
            truth_box_cxcywh[:num_objs_img, 2] = scaled_truth_w_all[
                batch, :num_objs_img
            ]
            truth_box_cxcywh[:num_objs_img, 3] = scaled_truth_h_all[
                batch, :num_objs_img
            ]

            # Extract gt cx, cy for the current batch
            truth_cell_x = truth_cell_x_all[batch, :num_objs_img]
            truth_cell_y = truth_cell_y_all[batch, :num_objs_img]

            # Calculate IoU between ground truth and reference anchors; reference anchors are the anchors / output_stride;
            # Anchors are used ONLY for the width and height, there is no information about their location in the image i.e. their center coordinate
            # For this specific calcuation, the 0s are used as the top left coordinate, and the w/h are used as the bottom right coordinate;
            # therefore, we don't have to set the xyxy parameter in bboxes_iou() since it's already in that form
            # Example:
            #   If there are 2 objects in an image, truth_box_wh will be shape (2, 4) where [: , :2)] are 0s and [:, 2:] are the w and h of each object (scaled by the output_stride)
            #   If there are 9 ref_anchors, then anchors_ious_all be shape (2, 9) so each object has an IoU score for every anchor.
            #   Note: The 0s will be filled in later with x, y
            anchor_ious_all = bboxes_iou(
                truth_box_cxcywh.cpu(), self.ref_anchors[output_id], CIoU=True
            )  # (num_gt_boxes, num_anchor_boxes)
            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            # Get index of the highest IoUs per ground truth box;
            # the best fit anchor box for each object
            # shape: (num_gt_boxes)
            #
            best_anch_iou_all = anchor_ious_all.argmax(dim=1)

            # I don't know why this is here
            best_n = best_anch_iou_all % 3

            # Check if the best anchor IoUs are within the anchor boxes designated for that output scale; also I'm not sure why this is done
            # Example:
            #   self.anch_masks holds the index of the anchors for each output feature map
            #   so if there's 9 anchors, it wants to use 3 anchors for each outpu prediction scale
            #   self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]; again, this seems like a bad way to do this
            best_anch_iou_mask = (
                (best_anch_iou_all == self.anch_masks[output_id][0])
                | (best_anch_iou_all == self.anch_masks[output_id][1])
                | (best_anch_iou_all == self.anch_masks[output_id][2])
            )

            # If the none of the highest IoUs is not within the group of anchors corresponding to the anchor masks,
            # skip to the next image in the batch;
            # I think this check is to make sure you're on the proper output scale but I don't really understand it
            if sum(best_anch_iou_mask) == 0:
                continue

            # If the condition above is passed, we can now get the IoUs of the predictions and the ground truths bboxes

            # Grab the cx and cy coords of each object
            truth_box_cxcywh[:num_objs_img, 0] = scaled_truth_cx_all[
                batch, :num_objs_img
            ]
            truth_box_cxcywh[:num_objs_img, 1] = scaled_truth_cy_all[
                batch, :num_objs_img
            ]

            # Collapse all cell predictions into 2D (num_cell_preds*out_H*out_W, 4) and calculate the IoUs between all cell predictions and the ground truth boxes
            # num_cell_preds is the number of predictions per cell
            pred_ious = bboxes_iou(
                pred[batch].reshape(-1, 4), truth_box_cxcywh, xyxy=False
            )  # (num_cell_preds*out_H*out_W, num_objs_img)

            # Get the highest IoU for each prediction; this will be the gt object that has the highest IoU; (num_cell_preds*out_H*out_W)
            pred_best_iou, _ = pred_ious.max(dim=1)

            # Check if each predicted IoU is above the ignored threshold; see attribute definition fo rmore information
            pred_best_iou = pred_best_iou > self.iou_ignore_threshold

            # pred_best_iou is now a boolean tensor, convert it to (num_cell_preds, out_H, out_W);
            # we now have a boolean mask if the highest IoU between the prediction and the gt objects for
            # each cell prediction is higher than self.iou_threshold
            pred_best_iou = pred_best_iou.view(pred[batch].shape[:3])

            # Set mask elements to zero (ignore) if the prediction and ground truth IoU is greater than the threshold self.iou_ignore_threshold;
            # explanation here: https://stackoverflow.com/questions/56199478/what-is-the-purpose-of-ignore-thresh-and-truth-thresh-in-the-yolo-layers-in-yolo
            # another explanation here: https://www.programmersought.com/article/9049233456/ (I didn't look through this one yet)
            obj_mask[batch] = ~pred_best_iou

            # Loop through each object in the image
            for img_object in range(best_n.shape[0]):
                # If best object & anchor IoU is within the proper output scale
                if best_anch_iou_mask[img_object] == True:
                    # i is scaled_gt_cx and j is scaled_gt_cy
                    cell_x, cell_y = truth_cell_x[img_object], truth_cell_y[img_object]

                    # Extract best anchor from the iou between gt box and anchor box
                    best_anch = best_n[img_object]

                    # Set the obj_mask of the batch at the best anchor IoU (between gt and ref anchors) prediction of the cell location to 1;
                    # this is a special case where the prediction IoU is over the self.iou_ignore_threshold so it is set to 0, however,
                    # since this is the best anchor, we stil want to include it in the loss so we need to set it to 1... I think this is why...
                    # obj_mask (B, num_cell_preds, out_H, out_W)
                    obj_mask[batch, best_anch, cell_y, cell_x] = 1

                    # Set all elements in the last dimension of the same location 1 at the best anchor position;
                    # as described above and in the paper, gt objects are only assigned 1 anchor
                    # (B, num_cell_preds, out_H, out_W, 4 + num_classes);
                    # this will be used to 0 out the predictions which do not correspond to a best anchor
                    tgt_mask[batch, best_anch, cell_y, cell_x, :] = 1

                    # Store the x, y offsets from the grid cell top_left coordinate;
                    # Ex: grid_cell_x = 43 and x_gt = 43.35 the value stored will be 0.35
                    # Even though the variables are the same, this works becuase convert to an integer
                    # will truncate the floatW
                    target[batch, best_anch, cell_y, cell_x, 0] = scaled_truth_cx_all[
                        batch, img_object
                    ] - scaled_truth_cx_all[batch, img_object].to(torch.int16).to(
                        torch.float
                    )
                    target[batch, best_anch, cell_y, cell_x, 1] = scaled_truth_cy_all[
                        batch, img_object
                    ] - scaled_truth_cy_all[batch, img_object].to(torch.int16).to(
                        torch.float
                    )

                    # Calculate and store the ground truth t_w, t_h; this comes from the formula in the yolov2 paper b_w = (p_w)e^(t_w)
                    # where b_w = bbox_width, p_w = anchor box width, t_w = the width prediction from the NN; in this case, we're calculating gt t_w
                    # so we need to solve for it, or at least that's what I believe they're doing here;
                    # we can solve for t_w with: log(b_w/p_w) = t_w
                    # This is also expressed in the YoloV3 paper section 2.1: "This ground truth value can be easily computed by inverting the equations above."
                    target[batch, best_anch, cell_y, cell_x, 2] = torch.log(
                        scaled_truth_w_all[batch, img_object]
                        / torch.Tensor(self.masked_anchors[output_id])[
                            best_n[img_object], 0
                        ]
                        + 1e-16
                    )
                    target[batch, best_anch, cell_y, cell_x, 3] = torch.log(
                        scaled_truth_h_all[batch, img_object]
                        / torch.Tensor(self.masked_anchors[output_id])[
                            best_n[img_object], 1
                        ]
                        + 1e-16
                    )

                    # Set the objectness to 1 since an object does exist there
                    target[batch, best_anch, cell_y, cell_x, 4] = 1

                    # Set the ground truth object label to 1 in it's correct position in the last dimension, every other object label
                    # remains 0 because it is not that object;
                    # the class predictions are after the first 5 elements in the last dimension, so index 5 would be the first object label
                    # if gt_labels[batch, img_object, 4] >= 80:
                    target[
                        batch,
                        best_anch,
                        cell_y,
                        cell_x,
                        5
                        + gt_labels[batch, img_object, 4].to(torch.int16).cpu().numpy(),
                    ] = 1

                    # Not entirely sure what this does
                    tgt_scale[batch, best_anch, cell_y, cell_x, :] = torch.sqrt(
                        2
                        - scaled_truth_w_all[batch, img_object]
                        * scaled_truth_h_all[batch, img_object]
                        / f_map_size
                        / f_map_size
                    )

        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, bbox_predictions_scales: list[torch.Tensor], labels):
        """Calculate the loss loss

        Args:
            bbox_predictions:
            labels:
        """
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0

        # Loop through each prediction scale
        for bbox_id, bbox_predictions in enumerate(bbox_predictions_scales):
            batchsize = bbox_predictions.shape[0]
            feature_size = bbox_predictions.shape[2]
            num_pred_ch = 5 + self.n_classes

            assert batchsize == self.batch_size

            # (B, num_anchors, ch_per_anchor, H, W); num_anchors is basically the number of bounding box predictions per cell
            bbox_predictions = bbox_predictions.view(
                batchsize, self.n_anchors, num_pred_ch, feature_size, feature_size
            )

            # (B, num_Anchors, H, W, ch_per_anchor); allows us to access each grid cell prediction
            bbox_predictions = bbox_predictions.permute(0, 1, 3, 4, 2)  # .contiguous()

            # Apply sigmoid function to tx & ty, objectness, and cls predictions; this bounds all predictions between 0-1 except for tw, th (index 2 & 3);
            # tw, th not bound because they have to be able to predict a width and height that spans more than the grid cell
            bbox_predictions[..., np.r_[:2, 4:num_pred_ch]] = torch.sigmoid(
                bbox_predictions[..., np.r_[:2, 4:num_pred_ch]]
            )

            # Extract tx, ty, tw, th
            pred = bbox_predictions[..., :4].clone()

            # Add the grid coordinates to the prediction offsets
            pred[..., 0] += self.grid_x[bbox_id]
            pred[..., 1] += self.grid_y[bbox_id]

            # Calculate the bbox prediction w, h by scaling the anchors w, h for each cell prediction;
            # (B, num_bbox_pred, H, W, 4) * (B, num_anchors_per_scale, H, W)
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[bbox_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[bbox_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_ground_truth_tensors(
                pred, labels, feature_size, num_pred_ch, bbox_id
            )

            # Loss calculation
            # Set the objectness score to 0 if IoU is greater than the self.iou_ignore_threshold
            bbox_predictions[..., 4] *= obj_mask

            # Multiply the binary tgt_mask to every prediction (except for objectness) to 0 out the predictions which DO NOT
            # correspond to the best anchor;
            # np.r_ is used to temporarily concatenate the [0:4] and [5:num_pred_ch] so it can multiply tgt_mask by every element in the
            # last dimension except for index 4 since bbox_predictions last dimension has  85 elements and tgt_mask only has 84;
            # this leaves the 4th index unchanged in bbox_predictions
            bbox_predictions[..., np.r_[0:4, 5:num_pred_ch]] *= tgt_mask

            # Multiply the prediction w/h by the tgt_scale calculated above; I have no idea what the tgt_scale is used for
            bbox_predictions[..., 2:4] *= tgt_scale

            # Repeat the above 3 statements but for the target (labels)
            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:num_pred_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            # Calculate the loss of the cell offset predictions and ground truth offsets;
            # I'm not sure why they're using BCE loss instead of the MSE loss like the w/h use below
            # YoloV3 paper section 2.1 mentions: "During training we use sum of squared error loss"
            # whcih would be the mse_loss with reduction="sum";
            # breakpoint() # for testing use index [0,2,63,4]
            loss_xy += F.binary_cross_entropy(
                input=bbox_predictions[..., :2],
                target=target[..., :2],
                weight=tgt_scale * tgt_scale,
                reduction="sum",
            )

            # Calculate the loss of the bbox w/h scale predictions (tw, th);
            # the target tw, th has already been calculated by taking the inverse of the equation
            # in YoloV3 paper section 2.1
            loss_wh += (
                F.mse_loss(
                    input=bbox_predictions[..., 2:4],
                    target=target[..., 2:4],
                    reduction="sum",
                )
                / 2
            )

            # Calculate the objectness BCE loss
            loss_obj += F.binary_cross_entropy(
                input=bbox_predictions[..., 4], target=target[..., 4], reduction="sum"
            )

            # Calculate the class BCE loss; the reason we use BCE and not CE is because we're doing multilabel classifciation (not multiclass classification);
            # YoloV3 Paper section 2.2 mentions this;
            # this link explains it well: https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e#:~:text=Why%3F,they%20are%20not%20mutually%20exclusive.
            # the notable message is the following:
            #   "In YOLO v3, it’s changed to do multi-label classification instead of multi-class classification.
            #    Why? Because some dataset may contains labels that are hierarchical or related, eg woman and person.
            #    So each output cell could have more than 1 class to be true. Correspondingly, we also apply binary cross-entropy
            #    for each class one by one and sum them up because they are not mutually exclusive."
            # This link also explains the use of BCE for multi-label classification: https://discuss.pytorch.org/t/is-there-an-example-for-multi-class-multilabel-classification-in-pytorch/53579/7
            # Example of BCE loss for multi-label classification: https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/45?u=bradford415
            loss_cls += F.binary_cross_entropy(
                input=bbox_predictions[..., 5:], target=target[..., 5:], reduction="sum"
            )

            # This is calculated but not actually used in the loss functions
            loss_l2 += F.mse_loss(
                input=bbox_predictions, target=target, reduction="sum"
            )

        # Sum all the loss terms for the final loss value
        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
        GIoU: Generalized IoU
        DIoU: Distance IoU
        CIoU: Complete IoU
    Returns:
        A torch tensor of shape (N, K)
        where N is the num_gt_boxes and K is the boxes you're comparing
        The tensor contains the IoUs between both sets of bounding boxes

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # Compare every (1, 4) row in bboxes_a with every element in bboxes_b
        # intersection top left
        # tl (num_gt_objs, num_anchors, 2) bboxes_a[:, None, :2] (num_gt_objs, 1, 2) bboxes_b[:, :2] (num_anchors, 2)
        tl = torch.max(
            bboxes_a[:, None, :2], bboxes_b[:, :2]
        )  # indexing with None is the exact same as np.newaxis (https://stackoverflow.com/questions/1408311/numpy-array-slice-using-none)

        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])

        # centerpoint distance squared
        rho2 = (
            (bboxes_a[:, None, 0] + bboxes_a[:, None, 2])
            - (bboxes_b[:, 0] + bboxes_b[:, 2])
        ) ** 2 / 4 + (
            (bboxes_a[:, None, 1] + bboxes_a[:, None, 3])
            - (bboxes_b[:, 1] + bboxes_b[:, 3])
        ) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        # intersection bottom right
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        con_br = torch.max(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)

    # Create a truth matrix then convert the True/False to 0/1; tl/br (num_gt_objs, num_anchors, 2) -> en (num_gt_objs, 2)
    en = (tl < br).type(tl.type()).prod(dim=2)

    # Get the area of the intersection and union, then finally calculate the IoU; I did not spend the time going through this to fully understand it but it's on my TODO list
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2
                )
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou
