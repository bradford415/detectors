import math
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

        # masked_anchors: 
        # ref_anchors: 2D list of all anchors divided by the output stride; 
        #              need to scale the anchors by how much the output feature map was downsampled by
        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        # Some code I started to write to see if I could clean up this for loop; it's probably not worth it because I might mess something up
        # Leaving it here in case I come back to it
        # _anchors, _strides = np.array(self.anchors_temp).reshape(-1, 2), np.array(self.strides).repeat(self.n_anchors)[:, np.newaxis]
        # scaled_anchors = _anchors / _strides
        # _ref_anchors = np.zeros((_anchors.shape[0], 4), dtype=np.float32)
        # _ref_anchors[:, 2:] = scaled_anchors
        # ref_anchors = torch.from_numpy(_ref_anchors)


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


    def build_target(self, pred: list[torch.Tensor], labels: list[dict], batch_size, f_map_size, num_pred_ch, output_id):
        """Loops through a batch of predictions at each feature map scale TODO
        
        Args:
            pred: x, y, w, h predictions after scaling the anchors' w, h and adding the grid offsets to tx, ty (B, n_anchors, H, W, bbox_pred_coords)
            labels: Labels scaled labels for all images in the batch; includes bbox_coords, class labels, etc...
            batch_size:
            f_map_size: Feature map dimensions at specific output; YoloV4 has 3 different outputs each at a different resolution
            num_pred_ch: Number of predictions per cell; 5 + num_classes
            output_id: The index of the output feature map scale; since YoloV4 has 3 outputs, the possible indices are [0, 1, 2]
        """
        # Create mask of zeros and ones ##### START HERE
        tgt_mask = torch.zeros(batch_size, self.n_anchors, f_map_size, f_map_size, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batch_size, self.n_anchors, f_map_size, f_map_size).to(device=self.device)
        tgt_scale = torch.zeros(batch_size, self.n_anchors, f_map_size, f_map_size, 2).to(self.device)
        target = torch.zeros(batch_size, self.n_anchors, f_map_size, f_map_size, num_pred_ch).to(self.device)

        batch_max_objects = max([len(label["boxes"]) for label in labels])
        gt_labels = torch.zeros((pred.shape[0], batch_max_objects, 5))
        
        for index, label in enumerate(labels):
            gt_labels[index, :len(label["boxes"]), :] = torch.cat((label["boxes"], label["labels"].unsqueeze(1)), dim=1)

        # labels = labels.cpu().data

        # Note on how labels is formed (my interpretation from the repo im basing yolov4 off of):
        #   1. labels in github code is (B, max_gt_bboxes, 5), containing the augmented ground truth labels (tl_x, tl_y, br_x, br_y, class_id)
        #   2. if max_gt_bboxes is 60, but an image only has 6 bounding boxes, only the first 6 rows will have values,
        #      the rest will be 0s
        #   3. The reason it is hardcoded at 60 is because each image will have a different number of bounding boxes,
        #      but to batch the labels together in a tensor they have to be the same shape
        #   4. Idk if this is the best way to do it but I think it only breaks if there are more than max_gt_bboxes in an image
        
        # Number of objects per image
        n_objs = (gt_labels.sum(dim=2) > 0).sum(dim=1)

        # Resize ground truth boxes to match the output dimensions by divding by the stride; stride = input_dims/final_output_dims
        # transforms.Normalize already converted the gt boxes to yolo format ([cx, cy, w, h]), so we do not need to calcuate that here
        truth_x_all = gt_labels[:, :, 0] / self.strides[output_id]
        truth_y_all = gt_labels[:, :, 1] / self.strides[output_id]
        truth_w_all = gt_labels[:, :, 2] / self.strides[output_id]
        truth_h_all = gt_labels[:, :, 3] / self.strides[output_id]
        
        truth_i_all = truth_x_all.cpu().numpy()
        truth_j_all = truth_y_all.cpu().numpy()

        # Loop through each image
        for batch in range(batch_size):
            n = int(n_objs[batch])
            
            # skip image if no labels in image
            if n == 0:
                continue
            
            # Extract gt w, h for the current batch
            truth_box = torch.zeros(n, 4).to(self.device) # (num_gt_objects, 4)
            truth_box[:n, 2] = truth_w_all[batch, :n]
            truth_box[:n, 3] = truth_h_all[batch, :n]
            
            # Extract gt w, h for the current batch
            truth_i = truth_i_all[batch, :n]
            truth_j = truth_j_all[batch, :n]

            # Calculate IoU between truth and reference anchors; reference anchors are the anchors / output_stride
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True) # (num_gt_boxes, num_anchor_boxes)
            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            # Get index of the highest IoUs per ground truth box; shape: (num_gt_boxes)
            best_n_all = anchor_ious_all.argmax(dim=1)

            # I don't know why this is here
            best_n = best_n_all % 3
            
            # Check if the best anchor IoUs are within the first 3 - Also not sure why this is done
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            # If the highest IoU is not within the group of anchors corresponding to the anchor masks,skip to the next image in the batch
            if sum(best_n_mask) == 0:
                continue

            breakpoint()
            ## START HERE TODO need to understand this, so far it keeps skipping this part
            truth_box[:n, 0] = truth_x_all[batch, :n]
            truth_box[:n, 1] = truth_y_all[batch, :n]
            pred_ious = bboxes_iou(pred[batch].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[batch] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[batch, a, j, i] = 1
                    tgt_mask[batch, a, j, i, :] = 1
                    target[batch, a, j, i, 0] = truth_x_all[batch, ti] - truth_x_all[batch, ti].to(torch.int16).to(torch.float)
                    target[batch, a, j, i, 1] = truth_y_all[batch, ti] - truth_y_all[batch, ti].to(torch.int16).to(torch.float)
                    target[batch, a, j, i, 2] = torch.log(
                        truth_w_all[batch, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[batch, a, j, i, 3] = torch.log(
                        truth_h_all[batch, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[batch, a, j, i, 4] = 1
                    target[batch, a, j, i, 5 + gt_labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[batch, a, j, i, :] = torch.sqrt(2 - truth_w_all[batch, ti] * truth_h_all[batch, ti] / f_map_size / f_map_size)
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

            # (B, num_anchors, ch_per_anchor, H, W); num_anchors is basically the number of bounding box predictions per cell
            bbox_predictions = bbox_predictions.view(batchsize, self.n_anchors, num_pred_ch, feature_size, feature_size)

            # (B, num_Anchors, H, W, ch_per_anchor); allows us to access each grid cell prediction
            bbox_predictions = bbox_predictions.permute(0, 1, 3, 4, 2)  # .contiguous()

            # Apply sigmoid function to tx & ty, objectness, and cls predictions; this bounds all predictions between 0-1 except for tw, th (index 2 & 3); 
            # tw, th not bound because they have to be able to predict a width and height that spans more than the grid cell
            bbox_predictions[..., np.r_[:2, 4:num_pred_ch]] = torch.sigmoid(bbox_predictions[..., np.r_[:2, 4:num_pred_ch]])

            # Extract tx, ty, tw, th
            pred = bbox_predictions[..., :4].clone()
            
            # Add the grid coordinates to the prediction offsets
            pred[..., 0] += self.grid_x[bbox_id]
            pred[..., 1] += self.grid_y[bbox_id]
            
            # Calculate the bbox prediction w, h by scaling the anchors w, h for each cell prediction; 
            # (B, num_bbox_pred, H, W, 4) * (B, num_anchors_per_scale, H, W)
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[bbox_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[bbox_id]
            
            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, feature_size, num_pred_ch, bbox_id)

            ## START HERE!!!!!!!!!!!!!!!!

            # loss calculation
            bbox_predictions[..., 4] *= obj_mask
            bbox_predictions[..., np.r_[0:4, 5:num_pred_ch]] *= tgt_mask
            bbox_predictions[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:num_pred_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=bbox_predictions[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=bbox_predictions[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(input=bbox_predictions[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(input=bbox_predictions[..., 5:], target=target[..., 5:], reduction='sum')
            loss_l2 += F.mse_loss(input=bbox_predictions, target=target, reduction='sum')

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

    ############ START HERE, need to go on laptop and run repo to see what these values are 
    if xyxy:
        # Compare every (1, 4) row in bboxes_a with every element in bboxes_b
        # intersection top left
        # tl (num_gt_objs, num_anchors, 2) bboxes_a[:, None, :2] (num_gt_objs, 1, 2) bboxes_b[:, :2] (num_anchors, 2)
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2]) # indexing with None is the exact same as np.newaxis (https://stackoverflow.com/questions/1408311/numpy-array-slice-using-none)
        
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
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
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou
