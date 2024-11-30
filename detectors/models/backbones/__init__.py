from .darknet import darknet53
from .resnet import resnet18

backbone_map = {"darknet53": darknet53, "resnet18": resnet18}
