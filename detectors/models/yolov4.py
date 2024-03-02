import torch
import torch.nn as nn
from torch.nn import functional as F


class Neck:
    def __init__(self):
        pass


class YoloV4(nn.Module):
    """Yolov4 based on the architecture described here https://arxiv.org/pdf/2004.10934.pdf

    Yolov4 implementation details in paper section 3.4
    """

    def __init__(self, backbone, neck=None):
        """
        Args:

        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck #################### START HERE, BUILD NECK FROM SCRATCH, REFEREANCE PAN PAPER, CAN BE FOUND FROM YOLOV4 SECTION 3.4#####################

    def forward(self, x):
        """Forward pass through the model

        Args:

        """
        out = self.backbone(x)

        return out
