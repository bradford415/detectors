import copy
import math

import torch
from torch import nn

from detectors.data.data import NestedTensor
from detectors.models.backbones.backbone import build_dino_backbone
from detectors.models.layers.common import MLP
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
        random_refpoints_xy: bool = False,
        fix_refpoints_hw: bool = False,  # why is hw and not wh?
        two_stage_type: str = "standard",
        two_stage_add_query_num: int = 0,
        decoder_pred_bbox_embed_share: bool = True,
        decoder_pred_class_embed_share: bool = True,
        two_stage_bbox_embed_share: bool = False,
        two_stage_class_embed_share: bool = False,
        decoder_self_attn_type: str = "sa",
        num_patterns: int = 0,
        denoise_number: int = 100,
        denoise_box_noise_scale: int = 0.4,
        denoise_label_noise_ratio: int = 0.5,
        denoise_labelbook_size: int = 100,  # TODO: maybe make this num_classes but verify first
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
            query_dim: TODO
            random_refpoints_xy: TODO
            fix_refpoints_hw: TODO
            two_stage_type: TODO
            two_stage_add_query_num: TODO
            decoder_pred_bbox_embed_share: TODO
            decoder_pred_class_embed_share: TODO
            two_stage_bbox_embed_share: TODO
            two_stage_class_embed_share: TODO
            decoder_self_attn_type: how to apply self-attention in the decoder;
                                    TODO verify these are accurate
                                    `sa` -> use standard self-attn among object queries

        """
        _hidden_dim = transformer.d_model

        self.backbone = backbone
        self.transformer = transformer

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = _hidden_dim
        self.num_heads = num_heads
        self.bb_num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss

        # TODO: figure out what this is for
        self.label_encoding = nn.Embedding(denoise_labelbook_size + 1, _hidden_dim)

        # Query dimensions TODO flesh out more
        self.query_dim = query_dim

        # Prediction layers' parameters
        self.decoder_pred_class_embed_share = decoder_pred_class_embed_share
        self.decoder_pred_bbox_embed_share = decoder_pred_bbox_embed_share

        # Denoise training params; TODO: maybe comment better
        self.num_patterns = num_patterns
        self.denoise_number = denoise_number
        self.denoise_box_noise_scale = denoise_box_noise_scale
        self.denoise_label_noise_ratio = denoise_label_noise_ratio
        self.denoise_labelbook_size = denoise_labelbook_size

        # Create the projection layers for feature map inputs to the encoder
        if num_feature_levels > 1:
            # Number of feature maps extracted from the backbone
            num_backbone_outs = len(backbone.bb_out_chs)

            # Create projection layers for each feature_map extracted from the backbone;
            # project the feature_maps output channels to hidden dim (b, bb_out_ch, h, w) -> (b, hidden_dim, h, w)
            input_proj_list = []
            for out_chs in backbone.bb_out_chs:
                in_channels = out_chs
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, _hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, _hidden_dim),
                    )
                )

            # Create additional feature_maps through CNN layers until len(input_proj_list) = num_feature_levels
            # (num_bb_feat_maps + additional_feat_maps); in the default case only 1 additional feat_map is created
            # NOTE: these layers have stride 2 so the additional feature maps will be downsampled by 2
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, _hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, _hidden_dim),
                    )
                )
                in_channels = _hidden_dim

            assert len(input_proj_list) == num_feature_levels

            self.input_proj = nn.ModuleList(input_proj_list)

        # Class embedding & bbox embedding; MLP will output dims of 4 (center_x, center_y, width, height) TODO: verify this
        _class_embed = nn.Linear(in_features=_hidden_dim, out_features=num_classes)
        _bbox_embed = MLP(
            input_dim=_hidden_dim, hidden_dim=_hidden_dim, out_dim=4, num_layers=3
        )

        # Initialize the class embedding to ~ -4.595;
        # intutition: intializing the bias of the classification layer encourages the model to
        # initially predict low confidence (near 0.01 bc sigmoid(-4.595) ~ 0.01) for  object presence,
        # this reflects the imbalance of object detection where most regions are background
        _prior_prob = 0.01
        bias_value = -math.log(
            (1 - _prior_prob) / _prior_prob  # inverse sigmoid of _prior_prob
        )
        _class_embed.bias.data = (
            torch.ones(self.num_classes)
            * bias_value  # In a linear layer the output_dim=num_bias values
        )

        # Initalize the weight & bias of the last layer in the bbox MLP to 0s
        # Intuition: setting last layer to 0s means the initial bbox prediction will always be
        # the same (typically center of the img with fixed size) regardless of input
        #   1. this "dummy" box gives the model a consistent starting point to learn from
        #   2. prevents wild, random box preds at the start
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        # Create a bbox MLP and class embed module for each decoder; if embed_share=True,
        # use the same MLP for each bbox and share the parameters
        if decoder_pred_bbox_embed_share:
            box_embed_layerlist = [
                _bbox_embed for _ in range(transformer.num_decoder_layers)
            ]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed)
                for _ in range(transformer.num_decoder_layers)
            ]
        if decoder_pred_class_embed_share:
            class_embed_layerlist = [
                _class_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(_class_embed)
                for i in range(transformer.num_decoder_layers)
            ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        if two_stage_type not in {"no", "standard"}:
            ValueError(f"Two stage type {two_stage_type} not supported")

        # If `standard` two_stage_type  (default is `standard`):
        #   1. create an additional MLP bbox and Linear layer class module for the transformer.enc_out
        #      TODO understand what this is and how it differs from self.transformer.decoder.bbox_embed
        #   2. if two_stage_embed_share=True (default is False) share the same MLP for the bboxes and
        #      the Linear layer for the class prediction, else create new modules (does not share parameters)
        #      TODO understand this two stage type more and the difference between the differnt bbox/cls modules created
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert decoder_pred_class_embed_share and decoder_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert decoder_pred_class_embed_share and decoder_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            # TODO: understand what this does and comment (default is 0 so this method not called)
            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:  # (default is 0)
                self.init_ref_points(two_stage_add_query_num)

        # default=sa; TODO: understand and comment
        self.decoder_self_attn_type = decoder_self_attn_type
        if decoder_self_attn_type == "ca_label":
            self.label_embedding = nn.Embedding(num_classes, _hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        elif decoder_self_attn_type == "sa" or "ca_content":
            # set label_embedding for each decoder layer to None; TODO explain a bit more
            self.label_embedding = None
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
        else:
            raise ValueError(
                f"decoder_sa_type `{decoder_self_attn_type}` not recognized"
            )

        self._init_projection_layers()

    def _init_projection_layers(self):
        """Initalize the parameters of conv2d for the feature_map projection layers;
        these layers project the feature_maps before passing to the encoder
        """
        for proj in self.input_proj:
            nn.init.xavier_uniform_(
                proj[0].weight, gain=1
            )  # Ensures stable forward/backward variance early in training
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        raise NotImplementedError(
            "The default DINO parameters does not call this method so I'm leaving it out for now to simplify the code; will add back if need"
        )

    def forward(self, samples: NestedTensor):
        """Forward pass through the DINO module

        Args:
            x: input images to the model
        """
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

    model = DINO(**dino_args)  # TODO pass args be keyword here probably
