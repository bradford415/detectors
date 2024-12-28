import torch
from torch import nn

from detectors.models.layers.common import ConvNormLRelu, Upsample
from detectors.models.layers.yolo import YoloLayerNew


class ScalePrediction(nn.Module):
    """Predicts the the bounding boxes at a particular scale

    This is branches out from the main path by a few convolution layers to make the predictions;
    this module is used because after the prediction the main branch use has a differnt number of filters as an
    input, so we can use the output from the module before this one; this is why the yolo3.cfg file
    has [route] layers = -4, it grabs the output from 4 layers backwards.
    """

    def __init__(
        self,
        in_chs: int,
        pred_chs: int,
        scale_anchors: list[list[int, int]],
        num_classes: int,
    ):
        super().__init__()
        self.conv = ConvNormLRelu(in_channels=in_chs, out_channels=in_chs * 2)
        self.pred = nn.Conv2d(
            in_channels=in_chs * 2,
            out_channels=pred_chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.yolo = YoloLayerNew(scale_anchors, num_classes)

    def forward(self, x, img_size):
        """Create predictions for the particular scale

        Args:
            x: feature maps from previous CNN layers (b, c, h, w)
            img_size: height of the original input image to the model; this is used to calculate the output stride;

        Returns:
            during inference:
                1. scales all predictions for each grid cell for every anchor (b, nx*ny*num_anchors, 5 + num_classes)
            during training:
                1. reshapes x (b, num_anchors*(num_classes+5), ny, nx) -> (B, num_anchors, ny, nx, (num_classes+5))
        """
        x = self.conv(x)
        x = self.pred(
            x
        )  # (b, (num_classes+5)*num_anchors, ny, nx); ny = height of grid scale, nx = width of grid scale
        x = self.yolo(x, img_size)
        return x


class Yolov3Head(nn.Module):
    """TODO"""

    def __init__(self, anchors: list[list[int, int]], num_classes: int, input_ch: int):
        """TODO

        Args:
            anchors: list of all anchor box [w, h] across all 3 scales by increasing size
        """
        super().__init__()

        # darknet_pred_channels =

        assert len(anchors) % 3 == 0

        pred_chs = (5 + num_classes) * 3  # 3 = num_anchors

        # in_channels comes from the out_ch of DarkNet53
        self.layers = nn.ModuleList(
            [
                ConvNormLRelu(
                    in_channels=input_ch,
                    out_channels=512,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                ConvNormLRelu(in_channels=512, out_channels=1024),
                ConvNormLRelu(
                    in_channels=1024,
                    out_channels=512,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                ConvNormLRelu(in_channels=512, out_channels=1024),
                ConvNormLRelu(
                    in_channels=1024,
                    out_channels=512,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                ScalePrediction(
                    in_chs=512,
                    pred_chs=pred_chs,
                    scale_anchors=anchors[6:],
                    num_classes=num_classes,
                ),
                ConvNormLRelu(
                    in_channels=512,  # input comes from the output of the module before ScalePrediction; [route] layers=-4 in yolov3.cfg
                    out_channels=256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                Upsample(scale_factor=2),
                ConvNormLRelu(
                    in_channels=768,
                    out_channels=256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                ConvNormLRelu(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                ConvNormLRelu(
                    in_channels=512,
                    out_channels=256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                ConvNormLRelu(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                ConvNormLRelu(
                    in_channels=512,
                    out_channels=256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                ScalePrediction(
                    in_chs=256,
                    pred_chs=pred_chs,
                    scale_anchors=anchors[3:6],
                    num_classes=num_classes,
                ),
                ConvNormLRelu(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                Upsample(scale_factor=2),
                ConvNormLRelu(
                    in_channels=384,
                    out_channels=128,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                ConvNormLRelu(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                ConvNormLRelu(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                ConvNormLRelu(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                ConvNormLRelu(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                ScalePrediction(
                    in_chs=128,
                    pred_chs=pred_chs,
                    scale_anchors=anchors[:3],
                    num_classes=num_classes,
                ),
            ]
        )

    def forward(self, backbone_out, inter_fm_2, inter_fm_1, img_size: int):
        """Forward pass through Yolov3 head

        The sptial size of each feature map should be: backbone_out < inter_fm_2 < inter_fm_1

        Args:
            backbone_out: the final feature map output from the backbone
            inter_fm_2: an intermediate output of the backbone described by the yolov3.cfg file
            inter_fm_1: an intermediate output of the backbone described by the yolov3.cfg file;
            img_size: height of the original input image to the model; this is used to calculate the output stride;
        """
        x = backbone_out
        route_connection = [inter_fm_1, inter_fm_2]

        # fm_1 spatial dimensions should be greater than fm_2
        assert inter_fm_1.shape[2:] > inter_fm_2.shape[2:]

        yolo_outputs = []
        for layer in self.layers:

            if isinstance(layer, ScalePrediction):
                yolo_outputs.append(layer(x, img_size))
                continue

            x = layer(x)

            # Upsample then concat the intermediate feature maps from backbone
            if isinstance(layer, Upsample):
                # breakpoint()
                x = torch.cat([x, route_connection.pop()], dim=1)

        return yolo_outputs


class Yolov3(nn.Module):
    """Full YoloV3 detection model

    This conists of the backbone and head.
    """

    def __init__(
        self, backbone: nn.Module, anchors: list[list[int, int]], num_classes: int
    ):
        super().__init__()
        self.backbone = backbone
        self.head = Yolov3Head(anchors, num_classes, input_ch=backbone.final_num_chs)

        # keep track of initialized yolo layers to use their attributes in the loss function; e.g., scaling anchors
        self.yolo_layers = [
            layer.yolo
            for layer in self.head.layers
            if isinstance(layer, ScalePrediction)
        ]
        assert len(self.yolo_layers) == 3

    def forward(self, x):
        """Forward pass through Yolov3 model

        Args:
            x: batch of input images (b, c, h , w)
        """
        img_size = x.shape[2]  # used to calcuate the output stride

        out, inter2, inter1 = self.backbone(x)
        # breakpoint()
        yolo_outputs = self.head(out, inter2, inter1, img_size=img_size)

        # during inference, concatentate all predictions from every scale
        # then we can perform non-maximum suppression
        if not self.training:
            # (b,(grid_h*grid_w*num_anchors)*num_scales, 5 + num_classes)
            # TODO: verify this shape
            train_outputs, inference_outputs = zip(*yolo_outputs)
            inference_outputs = torch.cat(inference_outputs, dim=1)
            yolo_outputs = (train_outputs, inference_outputs)

        return yolo_outputs
