import torch
from torch import nn

from detectors.models.backbones.backbone import build_dino_backbone
from detectors.models.layers.deformable_transformer import \
    build_deformable_transformer


class DINO(nn.Module):
    """Cross-attention dector module that performs object detection"""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        transformer: nn.Module,
        num_classes: int,
        num_queries: int,
        num_heads: int = 8,
        num_feature_levels: int = 4,
        aux_loss: bool = True,
        query_dim: int = 4,
        two_stage_type: str = "standard",
        decoder_pred_class_embed_share: bool = True,
        decoder_pred_embed_share: bool = True,
        two_stage_bbox_embed_share: bool = True,
        two_stage_class_embed_share: bool = True,
        decoder_sa: str = "sa",
        num_patterns: int = 0,
        denoise_number: int = 100,
        denoise_box_noise_scale: int = 0.4,
        denoise_label_noise_rato: int = 0.5,
        denoise_labelboox_size: int = 100,  # maybe make this num_classes
    ):
        """Initalize the DINO detector

        Args:
            backbone: backbone network to use for feature extraction; e.g., ResNet, ViT, Swin
            transformer: TODO
            num_classes: number of classes in the dataset ontology
            num_queries: number of object queries to use for the transformer; this is the number of maximum objects
                         conditional DETR can detect in a single image; for COCO, 100 queries is recommended
            num_heads: TODO
            num_feature_levels: the number of multiscale feature maps to pass into the encoder; the
                                feature maps will come from the backbone defined by `return_levels` 
                                but if len(return_levels) < num_features_levels, new feature_maps
                                will be created in this module until num_feature_maps=num_feature_levels;
                                in the default case, 3 feature maps are extracted from the backbone and
                                1 is created in this module
            aux_loss: whether to use auxiliary decoding losses; i.e., a loss at each decoder layer

        """
        self.backbone = backbone
        self.transformer = transformer

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = transformer.d_model
        self.num_heads = num_heads
        
        self.bb_num_feature_levels = num_feature_levels
        
        self.aux_loss = aux_loss

        # Query dimensions TODO flesh out more
        self.query_dim = query_dim

        # Create the projection layers
        if num_feature_levels > 1:
            # Number of feature maps extracted from the backbone
            num_backbone_outs = len(backbone.bb_out_chs)
            
            # Create projection layers for each feature_map extracted from the backbone;
            # project the feature_maps output channels to hidden dim (b, bb_out_ch, h, w) -> (b, hidden_dim, h, w)
            input_proj_list = []
            for out_chs in backbone.bb_out_chs:
                in_channels = out_chs
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))

            # Create additional feature_maps through CNN layers until len(input_proj_list) = num_feature_levels 
            # (num_bb_feat_maps + additional_feat_maps); in the default case only 1 additional feat_map is created
            # NOTE: these layers have stride 2 so the additional feature maps will be downsampled by 2
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim

            assert len(input_proj_list) == num_feature_levels
            self.input_proj = nn.ModuleList(input_proj_list)
            
            ################ START HERE

            
    def forward(self, x):
        pass


def build_dino(
    *,
    backbone_args: dict[str, any],
    transformer_args: dict[str, any],
    dino_args: dict[str, any],
):
    """Build the DINO detector

    Args:
        backbone_args: parameters specifically for the build_backbone() function;
                       see models.backbones.backbone.build_backbone() for parameter descriptions
        transformer_args: TODO
        dino_args: TODO: see if I want to split the dino params up more

    """

    backbone = build_dino_backbone(**backbone_args)

    transformer = build_deformable_transformer(**transformer_args)

    model = DINO(**dino_args)
