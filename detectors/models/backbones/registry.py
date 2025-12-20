from detectors.utils.registry import Registry

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = (
    "Registry for backbones, which extract feature maps from images"
)
