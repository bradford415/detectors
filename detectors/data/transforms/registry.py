from torchvision.transforms.v2 import (
    RandomIoUCrop,
    RandomPhotometricDistort,
    RandomZoomOut,
    Resize,
    SanitizeBoundingBoxes,
)

import detectors.data.transforms as T
from detectors.utils.registry import Registry

from .transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomIoUCrop,
    RandomPad,
    RandomResize,
    RandomSelect,
    RandomSizeCrop,
    ToTensor,
    Unnormalize,
)

TRANSFORM_REGISTRY = Registry("TRANSFORMS")
TRANSFORM_REGISTRY.__doc__ = (
    "Registry for data transformations, which augment data at train/test time"
)
to_register = [
    CenterCrop,
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomIoUCrop,
    RandomPad,
    RandomPhotometricDistort,
    RandomResize,
    RandomSelect,
    RandomSizeCrop,
    RandomZoomOut,
    Resize,
    SanitizeBoundingBoxes,
    ToTensor,
    Unnormalize,
]

for name in to_register:
    TRANSFORM_REGISTRY.register(name)
