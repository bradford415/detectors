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
        self.conv1 = ConvNormLRelu(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvNormLRelu(in_channels=512, out_channels=1024)
        self.conv3 = ConvNormLRelu(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv4 = ConvNormLRelu(in_channels=512, out_channels=1024)
        self.conv5 = ConvNormLRelu(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv6 = ConvNormLRelu(in_channels=512, out_channels=1024)
        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=255, kernel_size=1, stride=1, padding=0)

        self.yolo1 = YoloLayerNew(anchors[6:], num_classes)

        self.conv8 = ConvNormLRelu(in_channels=1024, in_channels=256, kernel_size=1, stride=1, padding=0)

        self.upsample1 = Upsample(stride=2)

        self.conv9 = ConvNormLRelu(in_channels=256, in_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv10 = ConvNormLRelu(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv11 = ConvNormLRelu(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv12 = ConvNormLRelu(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv13 = ConvNormLRelu(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv14 = ConvNormLRelu(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=255, kernel_size=1, stride=1, padding=0)

        self.yolo2 = YoloLayerNew(anchors[3:6], num_classes)

        self.conv16 = ConvNormLRelu(in_channels=512, out_channels=128, keßrnel_size=1, stride=1, padding=0)
        
        self.upsample2 = Upsample(stride=2)
        
        self.conv17 = ConvNormLRelu(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv18 = ConvNormLRelu(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv19 = ConvNormLRelu(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv20 = ConvNormLRelu(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv21 = ConvNormLRelu(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv22 = ConvNormLRelu(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv23 = nn.Conv2d(in_channels=256, out_channels=255, kernel_size=1, stride=1, padding=0)

        self.yolo3 = YoloLayerNew(anchors[:3], num_classes)

    def foward():
        """Forward pass through Yolov3 neck or head TODO
        """
    

class YoloV3(nn.Module):
    """Full YoloV3 detection model
    
    This conists of the backbone, neck, and head.
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = 0
    