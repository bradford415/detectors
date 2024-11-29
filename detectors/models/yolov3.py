import torch
from torch import nn

from detectors.models.layers.common import ConvNormLRelu, Upsample
from detectors.models.layers.yolo import YoloLayerNew

class Yolov3Head(nn.Module):
    """TODO
    """
    
    def __init__(self, anchors: list[list[int, int]], num_classes: int):
        """TODO

        Args:
            anchors: list of all anchor box [w, h] across all 3 scales by increasing size 


        """
        super().__init__()

        assert len(anchors) % 3 == 0
        
        # in_channels comes from the out_ch of DarkNet53
        self.layers = nn.ModuleList([
        ConvNormLRelu(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
        ConvNormLRelu(in_channels=512, out_channels=1024),
        ConvNormLRelu(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
        ConvNormLRelu(in_channels=512, out_channels=1024),
        ConvNormLRelu(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
        ConvNormLRelu(in_channels=512, out_channels=1024),
        nn.Conv2d(in_channels=1024, out_channels=255, kernel_size=1, stride=1, padding=0),

        YoloLayerNew(anchors[6:], num_classes),

        ConvNormLRelu(in_channels=1024, in_channels=256, kernel_size=1, stride=1, padding=0),

        Upsample(stride=2),

        ConvNormLRelu(in_channels=256, in_channels=256, kernel_size=1, stride=1, padding=0),
        ConvNormLRelu(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        ConvNormLRelu(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
        ConvNormLRelu(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        ConvNormLRelu(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
        ConvNormLRelu(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=512, out_channels=255, kernel_size=1, stride=1, padding=0),

        YoloLayerNew(anchors[3:6], num_classes),

        ConvNormLRelu(in_channels=512, out_channels=128, keßrnel_size=1, stride=1, padding=0),
        
        Upsample(stride=2),
        
        ConvNormLRelu(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
        ConvNormLRelu(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        ConvNormLRelu(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
        ConvNormLRelu(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        ConvNormLRelu(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
        ConvNormLRelu(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=256, out_channels=255, kernel_size=1, stride=1, padding=0),

        YoloLayerNew(anchors[:3], num_classes),
        ])

    def foward(self, backbone_out, inter_fm_2, inter_fm_1):
        """Forward pass through Yolov3 neck or head TODO

        The sptial size of each feature map should be: backbone_out < inter_fm_2 < inter_fm_1

        Args:
            backbone_out: the final feature map output from the backbone
            inter_fm_2: an intermediate output of the backbone described by the yolov3.cfg file
            inter_fm_1: an intermediate output of the backbone described by the yolov3.cfg file;
        """
        x = backbone_out
        route_connection = [inter_fm_1, inter_fm_2]

        # fm_1 spatial dimensions should be greater than fm_2
        assert inter_fm_1.shape[2:] > inter_fm_2.shape[2:]

        yolo_outputs = []
        for layer in self.layers:
            if isinstance(layer, YoloLayerNew):
                yolo_outputs.append(layer(x))
                continue

            x = layer(x)

            # Upsample then concat the intermediate feature maps from DarkNet53
            if isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connection.pop()], dim=1)
        
        return yolo_outputs

    

class YoloV3(nn.Module):
    """Full YoloV3 detection model
    
    This conists of the backbone, neck, and head.
    """
    
    def __init__(self, backbone: nn.Module, anchors: list[list[int, int]], num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = Yolov3Head(anchors, num_classes)
    
    def forward(self, x):
        """Forward pass through YoloV3 model

        Args:
            x:
        """
        out, inter2, inter1 = self.backbone(x)
        yolo_outputs = self.head(out, inter2, inter1)

        # during inference, concatentate all predictions from every scale
        # then we can perform non-maximum suppression
        if not self.training:
            # (b,(grid_h*grid_w*num_anchors)*num_scales, 5 + num_classes)
            # TODO: verify this shape
            yolo_outputs = torch.cat(yolo_outputs, dim=1)

        return yolo_outputs