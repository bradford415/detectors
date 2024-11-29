from torch import nn

from detectors.models.layers.common import ConvNormLRelu
from detectors.models.layers.yolo import YoloLayerNew

class Yolov3Neck(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # in_channels comes from the out_ch of DarkNet53
        self.conv1 = ConvNormLRelu(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvNormLRelu(in_channels=512, out_channels=1024)
        self.conv3 = ConvNormLRelu(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv4 = ConvNormLRelu(in_channels=512, out_channels=1024)
        self.conv5 = ConvNormLRelu(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv6 = ConvNormLRelu(in_channels=512, out_channels=1024)
        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=255, kernel_size=1, stride=1, padding=0)

        ################## START HERE ################
        self.yolo1 = YoloLayerNew

class Yolov3Head(nn.Module):
    

class YoloV3(nn.Module):
    """Full YoloV3 detection model
    
    This conists of the backbone, neck, and head.
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = 0
    