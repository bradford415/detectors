from itertools import chain
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def yolo_forward_dynamic_old(
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
        1. boxes: (B, num_anchors * H * W, 1, 4) Normalized bounding box predictions [0,1], predicts num_anchors per grid cell (H, W);
                  grid cell -> size of final feature map;
                  4 = normalized upper left and lower right bbox coordinates;
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

    # Each list will have length of num_anchors because each cell predicts num_anchors bboxes;
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
    # Contsrain bxy to [0, 1] with sigmoid; this represents the center of the bounding box relative to the grid cell
    # This is only computes the first part of bx and by because we still need to add CxCy
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (
        scale_x_y - 1
    )  # scale_x_y in this case is 1 which would 0 out the 2nd term

    # Scale the w/h predictions by computing the first part of bw and bh, we will still need to multiply by anchor dimensions.
    # Reminder, the equation is b_wh = p_wh*e^(t_wh).
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

    device = "cpu"
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

    return boxes, cls_confs, object_confs


def yolo_forward_dynamic(
    output,
    conf_thresh,
    num_classes,
    anchors,
    num_anchors,
    scale_x_y,
):
    return boxes, cls_confs, object_confs


class YoloLayer(nn.Module):
    """TODO: Flesh this out more; basically just a post processing step, not learned params
    Yolo layer only used at inference time"""

    def __init__(self, anchors: list[int], stride, num_classes=80, num_anchors=3):
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

        # The number of outputs for a single prediction
        self.num_output = num_classes + 5

        self.num_anchors = num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        # Flatten anchors to 1d python list then convert to tensor (anchor_pairs, 2)
        # anchors = [pair for anchor_pair in anchors for pair in anchor_pair] # commenting out for now since 1d lists are easiest to store in .yamls (should consider making the anchors a config.py)
        anchors = torch.tensor(anchors).float().view(-1, 2)

        # Assign parameters to be saved/restored by the state_dict but no trained by the optimizer;
        # these also become attributes of the class
        self.register_buffer("anchors", anchors)
        self.register_buffer(
            "anchor_grid", anchors.clone().view(1, -1, 1, 1, 2)
        )  # (1, anchor_pairs, 1, 1, 2)

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
        # anchors = [anchor / self.stride for anchor in self.anchors]

        # return yolo_forward_dynamic_old(
        #     head_output,
        #     self.thresh,
        #     self.num_classes,
        #     anchors,
        #     num_anchors=3,  # int(len(scaled_anchors) / 2),
        #     scale_x_y=self.scale_x_y,
        # )
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
        # TODO recomment this for the new code
            1. boxes: (B, num_anchors * H * W, 1, 4) Normalized bounding box predictions [0,1], predicts num_anchors per grid cell (H, W);
                    grid cell -> size of final feature map;
                    4 = normalized upper left and lower right bbox coordinates;
                    the coordinates are not the size of input dimensions, they are normalized
            2. confs: (B, num_anchors * H * W, num_classes) confidences of each class in the ontology;
                    these are NOT probabilities because they do not sum to 1
        """
        # Output would be invalid if it does not satisfy this assert
        assert head_output.shape[1] == (5 + self.num_classes) * self.num_anchors

        # Slice the second dimension (channel) of output into:
        # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
        # And then into
        # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
        # batch = output.size(0)
        # H = output.size(2)
        # W = output.size(3)

        batch_size, _, grid_h, grid_w = head_output.shape

        # (B, num_anchors*(num_classes+5), out_h, out_w) -> (B, num_anchors, (num_classes+5), out_h, out_w) -> (B, num_anchors, out_h, out_w, (num_classes+5))
        head_output = head_output.view(
            batch_size, self.num_anchors, self.num_output, grid_h, grid_w
        ).permute(0, 1, 3, 4, 2)

        # (1, 1, grid_h, grid_w, 2)
        self.grid = self._make_grid(grid_w, grid_h).to(head_output)

        # Scale cx, cy predictions to [0, 1] and offset by cell grid,
        # then multiply by stride to scale back to the input size range
        head_output[..., 0:2] = (
            head_output[..., 0:2].sigmoid() + self.grid
        ) * self.stride

        # Scale w, h predictions by e and multply by scaled anchor size; multiplying by the anchor size;

        # anchor_grid has 3 scaled anchors and each cell predicts 3 bboxes so multiplying by anchor_grid
        # applies the anchor sizes, element-wise, to the w/h predictions;
        # anchors are not divided by stride during inference (only divided by stride when computing the loss)
        head_output[..., 2:4] = torch.exp(head_output[..., 2:4]) * self.anchor_grid

        # Scale objectness and class confidence predictions to [0, 1]
        head_output[..., 4:] = head_output[..., 4:].sigmoid()  # conf, cls

        # Reshape to (B, out_w*out_h*num_anchors, num_classes+5);
        # this allows us to concatenate all the yolo layers along dim=1 since
        # each layer returns a different scale
        head_output = head_output.reshape(
            batch_size, -1, self.num_output
        )  # (B, out_w * out_h * num_anchors, 5 + num_classes)

        # Normalize coordinates to [0, 1];
        # reminder output shape is (B, C, H, W) where H/W are downsampled by the stride
        # bx_bw /= head_output.size(3)
        # by_bh /= head_output.size(2)

        return head_output

    @staticmethod
    def _make_grid(grid_w: int = 20, grid_h: int = 20):
        """Create grid of (x, y) coordinates used for the cell offsets

        Args:
            grid_w: Width of the grid
            grid_h: Height of the grid

        Returns:
            Tensor of x/y coords of the cell grid (1, 1, grid_h, grid_w, 2)
        """
        yv, xv = torch.meshgrid(
            [torch.arange(grid_h), torch.arange(grid_w)], indexing="ij"
        )

        # Stack so [:, :, 0] is the x coord grid and [:, :, 1] is the y coord grid
        return torch.stack((xv, yv), dim=2).view((1, 1, grid_h, grid_w, 2)).float()


class YoloLayerNew(nn.Module):
    """Detection layer
    TODO: Flesh this out more; basically just a post processing step, not learned params
    Yolo layer only used at inference time"""

    def __init__(self, anchors: list[list[int, int]], num_classes: int = 80):
        """Initalize Yolo Inference Layer

        Args:
            anchors: lsit of anchors in (w, h) format, relative to the original input dimensions
            num_classes: number of classes in the ontology
        """
        super().__init__()
        self.num_classes = num_classes

        # The number of outputs per anchor
        self.num_output = 5 + num_classes

        self.num_anchors = len(anchors)

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # Flatten anchors to 1d python list then convert to tensor (anchor_pairs, 2)
        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)

        assert anchors.shape[0] == self.num_anchors and anchors.ndim == 2

        self.grid = torch.zeros(1)

        # Assign as model parameters to be saved/restored by the state_dict;
        self.register_buffer("anchors", anchors)
        self.register_buffer(
            "anchor_grid", anchors.clone().view(1, -1, 1, 1, 2)
        )  # (1, num_anchors, 1, 1, 2)

        # will be used in the loss function to scale the anchors; this gets set in forward()
        self.stride = None

    def forward(
        self, head_output: torch.Tensor, img_size: int
    ) -> Tuple[torch.tensor, torch.tensor]:
        """TODO

        Args:
            head_output: Output predictions of a certain scale from the detector head (b, num_anchors*(5+num_classes), grid_h, grid_w)
            img_size:

        Returns:
            during inference:
                1. scales all predictions for each grid cell for every anchor (b, nx*ny*num_anchors, 5 + num_classes)
                   where 5 = (cx, cy, h, w, objectness)
            during training:
                1. reshapes x (b, num_anchors*(num_classes+5), ny, nx) -> (b, num_anchors, ny, nx, (5+num_classes))
        """

        assert head_output.shape[1] == (5 + self.num_classes) * self.num_anchors

        # the ratio the input image was downsample by
        stride = img_size // head_output.shape[2]
        self.stride = stride

        assert stride in {8, 16, 32}

        # ny & nx are the height and width of the grid i.e., the final downsample feature at a scale
        batch_size, _, ny, nx = head_output.shape

        # (b, num_anchors*(num_classes+5), ny, nx) -> (b, num_anchors, (num_classes+5), ny, nx) -> (B, num_anchors, ny, nx, (num_classes+5))
        head_output = (
            head_output.view(batch_size, self.num_anchors, self.num_output, ny, nx)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # During inference
        if not self.training:
            # breakpoint()
            if self.grid.shape[2:4] != head_output.shape[2:4]:
                # create grid of x, y coords (1, 1, ny, nx, 2) where 2 = (x, y) positions in the grid
                self.grid = self._make_grid(nx, ny).to(head_output)

            # bound cx, cy predictions to [0, 1] and offset by cell grid,
            # then multiply by stride to scale back to the input image size range
            head_output[..., 0:2] = (
                head_output[..., 0:2].sigmoid() + self.grid
            ) * stride  # x/y

            # Scale w, h predictions by e and multiply by anchor w/h;
            # multiplying by anchor_grid, applies the anchor sizes, element-wise, to the w/h predictions;
            # anchors are not divided by stride during inference (only divided by stride when computing the loss)
            head_output[..., 2:4] = (
                torch.exp(head_output[..., 2:4]) * self.anchor_grid
            )  # w/h

            # Scale objectness and class confidence predictions to [0, 1]
            head_output[..., 4:] = head_output[..., 4:].sigmoid()  # conf, cls

            # Reshape to (b, nx*ny*num_anchors, num_classes+5);
            # this allows us to concatenate all the yolo layers along dim=1 since
            # each layer returns a different scale
            head_output = head_output.reshape(batch_size, -1, self.num_output)

        return head_output

    @staticmethod
    def _make_grid(grid_w: int = 20, grid_h: int = 20):
        """Create grid of (x, y) coordinates used for the cell offsets

        Args:
            grid_w: Width of the grid
            grid_h: Height of the grid

        Returns:
            Tensor of x/y coords of the cell grid (1, 1, grid_h, grid_w, 2)
        """
        yv, xv = torch.meshgrid(
            [torch.arange(grid_h), torch.arange(grid_w)], indexing="ij"
        )

        # Stack so [:, :, 0] is the x coord grid and [:, :, 1] is the y coord grid
        return torch.stack((xv, yv), dim=2).view((1, 1, grid_h, grid_w, 2)).float()
