import torch
import torch.nn as nn
from torch.nn import functional as F

from detectors.models.layers.yolo import YoloLayer
from detectors.utils.box_ops import get_region_boxes


class Conv_Bn_Activation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        activation,
        bn=True,
        bias=False,
    ):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)
            )
        else:
            self.conv.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, pad, bias=False
                )
            )
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            raise ValueError(f"Activation function {activation} not supported.")

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert x.data.dim() == 4

        if inference:
            return (
                x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1)
                .expand(
                    x.size(0),
                    x.size(1),
                    x.size(2),
                    target_size[2] // x.size(2),
                    x.size(3),
                    target_size[3] // x.size(3),
                )
                .contiguous()
                .view(x.size(0), x.size(1), target_size[2], target_size[3])
            )
        else:
            return F.interpolate(
                x, size=(target_size[2], target_size[3]), mode="nearest"
            )


class Neck(nn.Module):
    """Neck for a ResNet18 backbone input. A few minor changes were made to work with ResNet

    Derived from YoloV4 paper section 3.4 and https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/models.py#L239
    """

    def __init__(self):
        super().__init__()

        self.conv1 = Conv_Bn_Activation(512, 512, 1, 1, "leaky")
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, "leaky")
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(256, 256, 1, 1, "leaky")
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, "leaky")
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(128, 128, 1, 1, "leaky")
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, "leaky")
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, "leaky")
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, "leaky")
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, "leaky")
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, "leaky")

    def forward(self, input, downsample3, downsample2, inference=False):
        """

        Args:
            input: Input to the neck module; final output from the backbone
        """
        # breakpoint()
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        x7_up = self.upsample1(x7, downsample3.shape, inference)
        # R 85
        x8 = self.conv8(downsample3)
        # R -1 -3
        x8 = torch.cat([x8, x7_up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        x14_up = self.upsample2(x14, downsample2.shape, inference)
        # R 54
        x15 = self.conv15(downsample2)
        # R -1 -3
        x15 = torch.cat([x15, x14_up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)

        # Output is the final neck output and each of the 2nd to last convolution before upsampling
        return x20, x13, x6


class Yolov4Head(nn.Module):
    """YoloV4 head (final prediction)

    Architecture described here: https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/a65d219f9066bae4e12003bd7cdc04531860c672/models.py#L323
    """

    def __init__(self, output_ch, n_classes, anchors):
        """


        Args:
            output_ch: Number of output channels for the predictions; (4 + 1 + num_classes) * num_bboxes
            n_classes: Number of classes in the ontology
            inference: Whether the model is inferencing
        """
        super().__init__()

        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, "leaky")
        self.conv2 = Conv_Bn_Activation(
            256, output_ch, 1, 1, "linear", bn=False, bias=True
        )

        # Largest head_output dimensions
        self.yolo1 = YoloLayer(
            num_classes=n_classes,
            anchors=anchors[:6],
            stride=8,
        )

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, "leaky")

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv10 = Conv_Bn_Activation(
            512, output_ch, 1, 1, "linear", bn=False, bias=True
        )

        # Medium head_output dimensions
        self.yolo2 = YoloLayer(
            num_classes=n_classes,
            anchors=anchors[6 : 6 * 2],
            stride=16,
        )

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, "leaky")

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv18 = Conv_Bn_Activation(
            1024, output_ch, 1, 1, "linear", bn=False, bias=True
        )

        # Smallest head_output dimensions
        self.yolo3 = YoloLayer(
            num_classes=n_classes,
            anchors=anchors[6 * 2 : 6 * 3],
            stride=32,  # 512 input_dim / 16 head_output = 32
        )

    def forward(self, input1, input2, input3, inference=False):
        x1 = self.conv1(input1)
        predictions_scale1 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        predictions_scale2 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        predictions_scale3 = self.conv18(x17)

        if inference:
            y1 = self.yolo1(predictions_scale1)
            y2 = self.yolo2(predictions_scale2)
            y3 = self.yolo3(predictions_scale3)

            return get_region_boxes([y1, y2, y3])

        else:
            # scale1 has the largest dimensions, scale2 medium, scale3 smallest dimensions (should verify this by viewing shape)
            return [predictions_scale1, predictions_scale2, predictions_scale3]


class YoloV4(nn.Module):
    """Yolov4 based on the architecture described here https://arxiv.org/pdf/2004.10934.pdf

    Yolov4 implementation details in paper section 3.4
    """

    def __init__(
        self,
        num_classes,
        backbone,
        anchors,
        neck=None,
        head=None,
        num_bboxes=3,
    ):
        """TODO

        Args:
            num_classes:

        """
        super().__init__()

        # 4 = (tx, ty, tw, th), 1 = objectness, num_classes = number of classes in the ontology, num_bboxes = number of bounding box predictions per grid cell (3 in yolov4)
        output_channels = (4 + 1 + num_classes) * num_bboxes

        # List of anchor points (w,h); alternates between w,h coordinates -> num_anchors is len(anchors)/2
        self.anchors = anchors
        self.backbone = backbone
        self.neck = Neck()
        self.head = Yolov4Head(output_channels, num_classes, anchors)

    def forward(self, x, inference=False):
        """Forward pass through the model

        Args:
            x: Batch of images to train the model

        Return:
            If training:
                list of predictions feature maps at each scale (B, (4 + 1 + num_classes) * num_bboxes, H, W);
                length of list should be 3 since there are threee outputs in the yolo head
            If inferencing:
                post-processed training output; see dectors.utils.box_ops.get_region_boxes() for more info
        """
        downsample1, downsample2, downsample3, backbone_out = self.backbone(x)
        neck_out, x13, x6 = self.neck(
            backbone_out, downsample3, downsample2, inference=inference
        )
        head_out = self.head(neck_out, x13, x6, inference=inference)

        return head_out
