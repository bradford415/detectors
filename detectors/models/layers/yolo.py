from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def yolo_forward_dynamic(
    output,
    conf_thresh,
    num_classes,
    anchors,
    num_anchors,
    scale_x_y,
):
    """Computes the normalized Yolo prediction bounding boxes (not input dim sized).
    This function is only called during inference, once for each prediction scale.

    Args:
        output: Output feature maps from the head
        conf_threshold:
        num_class: Number of classes in ontology
        anchors:
        num_anchors: Number of anchor box priors
        scale_x_y:

    Returns:
        1. boxes: (B, num_anchors * H * W, 1, 4) Normalized bounding box predictions, predicts num_anchors per grid cell (H, W);
                  grid cell -> size of final feature map ; 4 = normalized upper left and lower right bbox coordinates;
                  the coordinates are not the size of input dimensions, they are normalized
        2. confs: (B, num_anchors * H * W, num_classes) confidences of each class in the ontology;
                  these are NOT probabilities because they do not sum to 1
    """
    # Output would be invalid if it does not satisfy this assert
    assert output.shape[1] == (5 + num_classes) * num_anchors

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    # batch = output.size(0)
    # H = output.size(2)
    # W = output.size(3)

    # Lists to store prediction offsets (tx,ty), width/height, objectness, and class confidence 
    # by slicing the head output
    bxy_list = []
    bwh_list = []
    object_confs_list = []
    cls_confs_list = []

    # Each list will have length of num_anchors becuase each cell predicts num_anchors bboxes;
    # 1 bbox prediction has length (5 + num_classes), so if we let each cell predict 3 bboxes per cell,
    # the last dimension will have length 3 * (5 + num_classes), so each list alement will have (5 + num_classes)
    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)

        # (batch_size, 2, output_w, output_h)
        bxy_list.append(output[:, begin : begin + 2])

        # (batch_size, 2, output_w, output_h)
        bwh_list.append(output[:, begin + 2 : begin + 4])

        # (batch_size, 1, output_w, output_h)
        object_confs_list.append(output[:, begin + 4 : begin + 5])

        # (batch_size, num_classes, output_w, output_h)
        cls_confs_list.append(output[:, begin + 5 : end])

    # Combine the list of tensors along channel dimension
    bxy = torch.cat(bxy_list, dim=1)  # (batch_size, num_anchors * 2, H, W)
    bwh = torch.cat(bwh_list, dim=1)  # (batch_size, num_anchors * 2, H, W)
    object_confs = torch.cat(object_confs_list, dim=1)  # (batch, num_anchors, H, W)
    cls_confs = torch.cat(
        cls_confs_list, dim=1
    )  # (batch, num_anchors * num_classes, H, W)

    # Reshape to (batch, num_anchors * H * W)
    object_confs = object_confs.view(
        output.size(0), num_anchors * output.size(2) * output.size(3)
    )

    # Reshape to [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(
        output.size(0), num_anchors, num_classes, output.size(2) * output.size(3)
    )

    # NOTE: view/reshape only changes the shape and the way you access the tensor, it does not change the memory layout.
    #       permute/transpose DOES change the memory layout so this will affect how the data is processed

    # permute: (batch, num_anchors, num_classes, H * W]) -> (batch, num_anchors, H * W, num_classes)
    # reshape: (batch, num_anchors, H * W, num_classes) -> (batch, num_anchors * H * W, num_classes)
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(
        output.size(0), num_anchors * output.size(2) * output.size(3), num_classes
    )

    # These next few equations are described in the YoloV2 paper in figure 3.
    # Contrain bxy to [0, 1] with sigmoid; this represents the center of the bounding box relative to the grid cell
    # This is only computes the first part of bx and by because we still need to add CxCy
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (
        scale_x_y - 1
    )  # scale_x_y in this case is 1

    # Scale the w/h predictions by computing the first part of bw and bh, we will still need to multiply by anchor dimensions.
    # The e^bwh is explained in the link below; basically, this is how the authors decided to parametize the scaling because it has nice properties, it does not have to be done like this
    # https://stats.stackexchange.com/questions/345251/coordinate-prediction-parameterization-in-object-detection-networks/345267#345267
    bwh = torch.exp(bwh)

    # Similarly, scale the objectness and class confidences between [0,1] 
    object_confs = torch.sigmoid(object_confs)
    cls_confs = torch.sigmoid(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)

    # grid_x, and grid_y act as the grid cells and will be used to add Cx, Cy grid offsets to the predictions; each "grid cell" is a 1x1 unit
    # (B, 1, H, W)
    grid_x = np.expand_dims(
        np.expand_dims(
            np.expand_dims(
                np.linspace(0, output.size(3) - 1, output.size(3)), axis=0
            ).repeat(output.size(2), 0),
            axis=0,
        ),
        axis=0,
    )

    # (B, 1, H, W)
    grid_y = np.expand_dims(
        np.expand_dims(
            np.expand_dims(
                np.linspace(0, output.size(2) - 1, output.size(2)), axis=1
            ).repeat(output.size(3), 1),
            axis=0,
        ),
        axis=0,
    )

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    # Lists to store final computations for each anchor
    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h; add grid offsets and multiply anchor dimensions to predictions
    for i in range(num_anchors):
        ii = i * 2

        # Add x/y grid offsets to the x preds and store the result 
        # bx: (batch, 1, H, W), by: (batch, 1, H, W)
        bx = bxy[:, ii : ii + 1] + torch.tensor(
            grid_x, device=device, dtype=torch.float32
        )  # grid_x.to(device=device, dtype=torch.float32)
        by = bxy[:, ii + 1 : ii + 2] + torch.tensor(
            grid_y, device=device, dtype=torch.float32
        )  # grid_y.to(device=device, dtype=torch.float32)
        
        # Multiply w/h anchor by the predictions to scale them and store result (batch, 1, H, W)
        # bw: (batch, 1, H, W), bh: (batch, 1, H, W)
        bw = bwh[:, ii : ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1 : ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)

    # Concat list of final computed predictions along anchor dimension
    # bx, by, bw, bh: (batch_size, num_anchors, H, W)
    bx = torch.cat(bx_list, dim=1)
    by = torch.cat(by_list, dim=1)
    bw = torch.cat(bw_list, dim=1)
    bh = torch.cat(bh_list, dim=1)

    # Concat [final x coord, final width] and [final y coord, final height]
    # bx_bw, by_bh (B, 2 * num_anchors, H, W)
    bx_bw = torch.cat((bx, bw), dim=1)
    by_bh = torch.cat((by, bh), dim=1)

    # Normalize coordinates to [0, 1]; 
    # reminder output shape is (B, C, H, W) where H/W are downsampled by the stride
    bx_bw /= output.size(3)
    by_bh /= output.size(2)

    # Extract each component and reshape to (batch, num_anchors * H * W, 1)
    bx = bx_bw[:, :num_anchors].view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )
    by = by_bh[:, :num_anchors].view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )
    bw = bx_bw[:, num_anchors:].view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )
    bh = by_bh[:, num_anchors:].view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )

    # Get the x and y coordinates of the upper left and lower right of the bounding box prediction;
    # Currently, bx/by are the center coordinate predictions;
    # The coordinates are still normalized at this point and may contain negatives
    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Group upper left and low right coordinates to form num_anchors prediction for each grid_cell
    # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4] ** Not sure why the 1 dim is needed **
    boxes = torch.cat((bx1, by1, bx2, by2), dim=2).view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1, 4
    )
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # Final prediction shapes
    # boxes:     [batch, num_anchors * H * W, 1, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    # (B, num_anchors * H * W, 1)
    object_confs = object_confs.view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )

    # Get final confidence scores by multiplying objectness by the class confidences
    confs = cls_confs * object_confs

    # Final prediction shapes
    # boxes: [batch, num_anchors * H * W, 1, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    ################### START HERE AND SEE WHERE IT TAKES US ################
    return boxes, confs


class YoloLayer(nn.Module):
    """Yolo layer only used at inference time"""

    def __init__(
        self,
        num_classes=80,
        anchors=[],
        stride=32,
    ):
        """Initalize Yolo Inference Layer

        Args:
            num_classes: Number of classes in the ontology
            anchors: List of anchors in (w, h) format, relative to the original input dimensions
            stride: The ratio to which the image was downsampled; this will be used to normalize
                    the anchor boxes coordinates to the range of the downsampled image;
                    i.e. input_size: (512, 512), head_output: (64, 64) -> stride = 8
        """
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

    def forward(self, head_output) -> Tuple[torch.tensor, torch.tensor]:
        """TODO

        Args:
            head_output: Output predictions of a certain scale from the dector head;

        Returns:
            1. bounding box predictions
            2. bounding box confidences; object_confidence * class_confidence
        """

        # Divide each anchor point by stride; this normalizes the anchor coordinates from
        # the input size to the head_output size
        scaled_anchors = [anchor / self.stride for anchor in self.anchors]

        return yolo_forward_dynamic(
            head_output,
            self.thresh,
            self.num_classes,
            scaled_anchors,
            num_anchors=3,#int(len(scaled_anchors) / 2),
            scale_x_y=self.scale_x_y,
        )



################################ GITHUB CODE HERE ##########################
def yolo_forward_dynamic_pytorch(output, conf_thresh, num_classes, anchors, num_anchors, scale_x_y, only_objectness=1,
                              validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    # batch = output.size(0)
    # H = output.size(2)
    # W = output.size(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)
        
        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        det_confs_list.append(output[:, begin + 4 : begin + 5])
        cls_confs_list.append(output[:, begin + 5 : end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3))

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(output.size(0), num_anchors, num_classes, output.size(2) * output.size(3))
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(output.size(0), num_anchors * output.size(2) * output.size(3), num_classes)

    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(3) - 1, output.size(3)), axis=0).repeat(output.size(2), 0), axis=0), axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(2) - 1, output.size(2)), axis=1).repeat(output.size(3), 1), axis=0), axis=0)
    # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx = bxy[:, ii : ii + 1] + torch.tensor(grid_x, device=device, dtype=torch.float32) # grid_x.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1 : ii + 2] + torch.tensor(grid_y, device=device, dtype=torch.float32) # grid_y.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii : ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1 : ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)


    ########################################
    #   Figure out bboxes from slices     #
    ########################################
    
    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = torch.cat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = torch.cat((by, bh), dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= output.size(3)
    by_bh /= output.size(2)

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx_bw[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    by = by_bh[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    bw = bx_bw[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    bh = by_bh[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)

    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
    boxes = torch.cat((bx1, by1, bx2, by2), dim=2).view(output.size(0), num_anchors * output.size(2) * output.size(3), 1, 4)
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # boxes:     [batch, num_anchors * H * W, 1, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * H * W, 1, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return  boxes, confs

class YoloLayer_pytorch(nn.Module):
    ''' Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    '''
    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1, stride=32, model_out=False):
        super(YoloLayer_pytorch, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        self.model_out = model_out

    def forward(self, output, target=None):
        if self.training:
            return output
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        return yolo_forward_dynamic_pytorch(output, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask),scale_x_y=self.scale_x_y)