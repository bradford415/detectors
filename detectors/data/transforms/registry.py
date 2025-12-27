from detectors.utils.registry import Registry
from detectors.data.transforms import T

TRANSFORM_REGISTRY = Registry("TRANSFORMS")
TRANSFORM_REGISTRY.__doc__ = (
    "Registry for data transformations, which augment data at train/test time"
)

for name in getattr(T, "__all__", []):
    TRANSFORM_REGISTRY.register(getattr(T, name))
