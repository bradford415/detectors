import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

from detectors.models.layers.yolo import YoloLayer, YoloLayer_pytorch
from detectors.utils.box_ops import get_region_boxes


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Conv_Bn_Activation_old(nn.Module):
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
  
    
class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class Upsample(nn.Module):
    """Module to upsample tensors to the target size"""

    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size: torch.Size, inference=False):
        """Upsamples the H/W of the input tensor x to the target_size

        x: Input tensor to be upsampled
        target_size: shape of the input tensor; shape should be (B, C, H, W)
        """

        assert x.data.dim() == 4
        assert len(target_size) == 4

        if inference:
            # ATTENTION: This inference case seems to be the EXACT SAME as interpolation with nearest neighbors
            #            i.e., the `else` case in this if statement;
            #            I HAVE NO IDEA WHY THEY WROTE IT LIKE THIS IF INTERPOLATION RETURNS THE SAME THING

            # This code works in the following manner:
            # x: (B, C, out_h, out_w)
            # x.view: (B, C, out_h, 1, out_w, 1)
            # x.view.expand: (B, C, out_h, 1*h_upsample_factor, out_w, 1*width_upsample_factor); expand repeats the dimension along its axis axes;
            #                this will first repeat along the last two axes, then it will repeat on the 2nd and 3rd axes effectively quadrupling the values;
            #                here's a snippet I wrote to sort of understand this:
            #                   (Pdb) temp (2, 1, 1, 3)
            #                   tensor([[[[1, 2, 3]]],
            #
            #                          [[[6, 7, 8]]]])
            #                   (Pdb) temp.expand(2,2,2,3)
            #                    tensor([[[[1, 2, 3], <----------notice these rows sort of quadruple
            #                              [1, 2, 3]],
            #
            #                             [[1, 2, 3],
            #                              [1, 2, 3]]],
            #
            #
            #                            [[[6, 7, 8],
            #                              [6, 7, 8]],
            #
            #                             [[6, 7, 8],
            #                              [6, 7, 8]]]])
            #
            # x.view.expand.contiguous.view: (B, C, target_size_h, target_size_w); this merges the last 4 axes into a rectangle

            # upsampled_x = (
            #     x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1)
            #     .expand(
            #     x.size(0),
            #         x.size(1),
            #         x.size(2),
            #         target_size[2] // x.size(2),
            #         x.size(3),
            #         target_size[3] // x.size(3),
            #     )
            #     .contiguous()
            #     .view(x.size(0), x.size(1), target_size[2], target_size[3])
            # )
            # TODO: Changed this to interpolate because of incorrect dimension sizes during evaluation
            interp_x = F.interpolate(
                x, size=(target_size[2], target_size[3]), mode="nearest"
            )

            # torch.allclose(upsampled_x, interp_x) # THIS RETURNS TRUE, I HAVE NO IDEA WHY YOU NEED ALL THIS VIEW AND EXPAND LOGIC

            # return upsampled_x
            return interp_x
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

    def forward(
        self,
        input: torch.Tensor,
        downsample3: torch.Tensor,
        downsample2: torch.Tensor,
        inference: bool = False,
    ):
        """

        Args:
            input: Input to the neck module; final output from the backbone; smallest feature map passed to Neck
            downsample3: Feature map output from 3rd downsample block in backbone; 2nd largest feature map passed to Neck
            downsample2: Feature map output from 2nd downsample block in backbone; largest feature map passed to Neck
            inference: If inferencing or not
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

        # Upsample to size of downsample3.shape; if inferencing, TODO
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
            breakpoint()
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
    


############################################## github code starts here  ############################################################
 
class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Neck_pytorch(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3, inference=False):
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
        up = self.upsample1(x7, downsample4.size(), inference)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head_pytorch(nn.Module):
    def __init__(self, output_ch, n_classes, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer_pytorch(
                                anchor_mask=[0, 1, 2], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo2 = YoloLayer_pytorch(
                                anchor_mask=[3, 4, 5], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo3 = YoloLayer_pytorch(
                                anchor_mask=[6, 7, 8], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=32)

    def forward(self, input1, input2, input3, inference):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

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
        x18 = self.conv18(x17)
        
        if inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return get_region_boxes([y1, y2, y3])
        
        else:
            return [x2, x10, x18]


class Yolov4_pytorch(nn.Module):
    def __init__(self, yolov4conv137weight=None, n_classes=80, inference=False):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        # neck
        self.neck = Neck_pytorch(inference)
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neck)
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)
        
        # head
        self.head = Yolov4Head_pytorch(output_ch, n_classes, inference)


    def forward(self, input, inference=False):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neck(d5, d4, d3, inference)

        output = self.head(x20, x13, x6, inference)
        return output
