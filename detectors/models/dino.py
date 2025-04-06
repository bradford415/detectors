import torch
from torch import nn


class DINO(nn.Module):
    """Cross-attention dector module that performs object detection"""

    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        num_classes: int,
        num_queries: int,
        num_heads: int,
        num_features_levels: int = 4,
        aux_loss: bool = True,
    ):
        """Initalize the DINO detector

        Args:
            backbone: backbone network to use for feature extraction; e.g., ResNet, ViT, Swin
            transformer: TODO
            num_classes: number of classes in the dataset ontology
            num_queries: number of object queries to use for the transformer; this is the number of maximum objects
                         conditional DETR can detect in a single image; for COCO, 100 queries is recommended
            num_heads: TODO
            num_features_levels: TODO: verify this is correct; number of feature levels outputs in the backbone
                                 to extract
            aux_loss: whether to use auxiliary decoding losses; i.e., a loss at each decoder layer

        """
        self.backbone = backbone
        self.transformer = transformer

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.aux_loss = aux_loss

        if num_features_levels > 1:
            

    def forward(self, x):
        pass
