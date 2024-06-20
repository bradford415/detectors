import numpy as np
import torch
from torch import nn

class YoloV4Loss(nn.Module):
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

    ################# start here - UNDERSTAND THIS FUNCTION and document it#####################
    def __init__(self, anchors, n_classes=80, n_anchors=3, device=None, batch=2):
        """TODO
        
        Args:
            n_classes:
        """
        super().__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 512
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], 
                        [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anchors_temp = anchors
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        _anchors, _strides = np.array(self.anchors_temp).reshape(-1, 2), np.array(self.strides).repeat(self.n_anchors)[:, np.newaxis]
        scaled_anchors = _anchors / _strides
        _ref_anchors = np.zeros((_anchors.shape[0], 4), dtype=np.float32)
        _ref_anchors[:, 2:] = scaled_anchors
        ref_anchors = torch.from_numpy(_ref_anchors)

        for i in range(self.n_anchors):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            feature_map_sizes = image_size // self.strides[i]

            # Create x & y grid of feature map pixel indices (B, num_anchors, H/stride, W/stride)
            grid_x = torch.arange(feature_map_sizes, dtype=torch.float).repeat(batch, 3, feature_map_sizes, 1).to(device)
            grid_y = torch.arange(feature_map_sizes, dtype=torch.float).repeat(batch, 3, feature_map_sizes, 1).permute(0, 1, 3, 2).to(device)

            # Create grid of the w & h of each anchor box w/ size of the feature maps (B, num_anchors, H/stride, W/stride)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, feature_map_sizes, feature_map_sizes, 1).permute(0, 3, 1, 2).to(
                device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, feature_map_sizes, feature_map_sizes, 1).permute(0, 3, 1, 2).to(
                device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)


    # START HERE!!!
    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        """TODO
        """
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, bbox_predictions, labels):
        """Calculate the loss loss

        Args:
            bbox_predictions: 
            labels: 
        """
        breakpoint()
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0

        # Loop through each prediction scale
        for bbox_id, bbox_predictions in enumerate(bbox_predictions):
            batchsize = bbox_predictions.shape[0]
            feature_size = bbox_predictions.shape[2]
            anchor_num_ch = 5 + self.n_classes

            # (B, num_anchors, ch_per_anchor, H, W)
            bbox_predictions = bbox_predictions.view(batchsize, self.n_anchors, anchor_num_ch, feature_size, feature_size)

            # (B, num_Anchors, H, W, ch_per_anchor); allows us to access each grid cell prediction
            bbox_predictions = bbox_predictions.permute(0, 1, 3, 4, 2)  # .contiguous()

            # Apply sigmoid function to tx & ty, objectness, and cls predictions; this bounds all predictions between 0-1 except for tw, th (index 2 & 3); 
            # tw, th not bound because they have to be able to predict a width and height that spans more than the grid cell
            bbox_predictions[..., np.r_[:2, 4:anchor_num_ch]] = torch.sigmoid(bbox_predictions[..., np.r_[:2, 4:anchor_num_ch]])

            # Extract tx, ty, tw, th
            pred = bbox_predictions[..., :4].clone()
            
            # Add the grid coordinates to the prediction offsets
            pred[..., 0] += self.grid_x[bbox_id]
            pred[..., 1] += self.grid_y[bbox_id]
            
            # Calculate the bbox prediction w, h by scaling the anchors w, h for each cell prediction; 
            # (B, num_bbox_pred, H, W, 4) * (B, num_anchors_per_scale, H, W)
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[bbox_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[bbox_id]
            
            #START HERE
            breakpoint()
            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, feature_size, anchor_num_ch, bbox_id)

            # loss calculation
            bbox_predictions[..., 4] *= obj_mask
            bbox_predictions[..., np.r_[0:4, 5:anchor_num_ch]] *= tgt_mask
            bbox_predictions[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:anchor_num_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=bbox_predictions[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=bbox_predictions[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(input=bbox_predictions[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(input=bbox_predictions[..., 5:], target=target[..., 5:], reduction='sum')
            loss_l2 += F.mse_loss(input=bbox_predictions, target=target, reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2

