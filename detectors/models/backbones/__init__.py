from .darknet import darknet53
from .presnet import PResNet
from .registry import BACKBONE_REGISTRY
from .resnet import resnet18, resnet50

backbone_map = {
    "darknet53": darknet53,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "PResNet": PResNet,
}
