#### START HER (: ######)
import torch
from torch import nn



class DINO(nn.Module):
    """Cross-attention dector module that performs object detection"""
    
    def __init__(self, backbone: nn.Module, transformer: nn.Module, num_classes: int, num_queries: int, num_heads: int, aux_loss: bool = False):
        """Initalize the DINO detector
        
        Args:
            backbone: backbone network to use for feature extraction; e.g., ResNet, ViT, Swin
            transformer: TODO
            num_classes: number of classes in the dataset ontology
            num_queries: number of object queries to use for the transformer; this is the number of maximum objects
                         conditional DETR can detect in a single image; for COCO, 100 queries is recommended
            aux_loss: whether to use auxiliary decoding losses; i.e., a loss at each decoder layer
            
        """
        pass
    
    def forward(self, x):
        pass