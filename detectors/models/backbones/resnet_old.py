from torch import nn
from torchvision import models


def build_resnet50(pretrain=False, remove_top=True):
    weights = "IMAGENET1K_V2" if pretrain else None

    model = models.resnet50(weights=weights)

    if remove_top:
        # Remove GlobalAveragePooling and FC layer
        model = nn.Sequential(*list(model.children())[:-2])

    return model
