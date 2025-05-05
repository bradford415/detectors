import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from detectors.data import NestedTensor
from detectors.models.backbones import backbone_map
from detectors.models.layers.positional import PositionEmbeddingSineHW
from detectors.utils.distributed import is_main_process

supported_backbones = ["resnet50"]


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.

    Copied exactly from DETR/DINO
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """Base class to prepare ResNet backbone networks, specifially for the DINO detector.

    In theory, this could be used for any backbone but there's a decent amount of hardcoded
    logic.
    """

    def __init__(
        self,
        backbone: nn.Module,
        bb_out_chs: list[int],
        bb_level_inds: list[int],
        train_backbone: bool = True,
    ):
        """Initialize the ResNet backbone base class

        Args:
            backbone: the backbone network to use as the feature extractor to the detector;
                      for dino this is typically a resnet or swin
            bb_out_chs: the number of output channels for each feature map level of the backbone
                        that was specified by bb_level_inds;
            bb_level_inds: the indices of backbone levels to fuse into the the detector;
                            i.e., heiarchicial features to extract from the backbone;
                            this is used to grab the number of output channels for each level;
                            len() must = 4 since resnet and swin only have 4 levels;
            train_backbone: whether to train the backbone network while training the detector;
                            if false, freeze the backbone and only train the detector

        """
        super().__init__()
        # Loop through all parameters in the backbone and freeze:
        #   1. all parameters if train_backbone is False (freezes the entire backbone)
        #   2. if train_backbone is True, only layer2, layer3, and layer4 (deep layers)
        #      are trainable and conv1, bn1, and layer1 (the early layers) of the resnet are frozen.
        #
        # The idea behind freezing the early layers is that they learn more generic features
        # of objects while the deeper layers are more task specific; i.e., fine-tuning only the
        # deep layers of ResNet keeps training stable, faster, and can often get better final
        # accuracy without over fitting; see models.backbones.resnet.ResNet
        # to see these layers; the default case in this code is to use #2
        for name, parameter in backbone.named_parameters():
            if not train_backbone or (
                "layer2" not in name and "layer3" not in name and "layer4" not in name
            ):
                parameter.requires_grad_(False)

        self.backbone = backbone
        self.bb_out_chs = bb_out_chs
        self.bb_level_inds = bb_level_inds

    def forward(self, tensor_list: NestedTensor) -> dict[str, NestedTensor]:
        """Extract features with the backbone and resize the input mask
        to match the spatial dimensions of each feature map

        Args:
            tensor_list: a NestedTensor of input images and masks

        Returns:
            a dictionary of nested tensors for each feature_map level with the padding
            mask interpolated to match the spatial resolution of the feature_map level;
            for example, if bb_level_inds=[1,2,3] the return dict will have:
                dict = {
                    "1": NestedTensor highest feature_map spatial resolution
                    "2": NestedTensor medium feature_map spatial resolution
                    "3": NestedTensor lowest feature_map spatial resolution
                }
        """
        # Extract features from the input images through the backbone network
        feature_maps = self.backbone(tensor_list.tensors)

        assert len(feature_maps) == 4

        # Extract only the desired feature maps
        feature_maps = np.array(feature_maps)[self.bb_level_inds]

        assert len(feature_maps) == len(self.bb_level_inds)

        # Resize the input mask to match the spatial size of the feature map;
        # the mask is currently the shape of the input tensor (tensor input to the back) and needs to be resized for each feature map
        nested_feature_maps: dict[str, NestedTensor] = {}
        # TODO: Ill most definitely have to modify this bc I didn't use intermediatlayergetter
        for name, feature_map in feature_maps.items():
            orig_mask = tensor_list.mask

            assert orig_mask is not None

            # resize padding mask to match the spatial dimensions of the feature map
            mask = F.interpolate(
                orig_mask[None].float(), size=feature_map.shape[-2:]
            ).to(torch.bool)[0]
            nested_feature_maps[name] = NestedTensor(feature_map, mask)

        return nested_feature_maps


class Backbone(BackboneBase):
    """Construct a backbone network for the DINO detector and use frozen batch norm layers

    In practice, this is only used for the resnet backbone; TODO: verify this.
    """

    def __init__(
        self,
        backbone_name: str,
        train_backbone: bool,
        bb_level_inds: list[int],
        batch_norm=FrozenBatchNorm2d,
    ):
        """Initialize the backbone network

        Args:
            backbone_name: the name of the backbone network to use; in practice this should only be
                  a resnet variant
            train_backbone: whether to train the backbone network while training the detector;
                            if false, freeze the backbone and only train the detector
            bb_level_inds: the indices of backbone levels to fuse into the the detector;
                            i.e., heiarchicial features to extract from the backbone;
                            this is used to grab the number of output channels for each level;
                            len() must = 4 since resnet and swin only have 4 levels
            batch_norm: the batch norm module to use; typically this is a frozen batch norm
                        while training the detector; i.e., only uses the bn statistics from
                        the pretrained backbone
        """
        if len(bb_level_inds) > 4 or max(bb_level_inds) >= 4:
            raise ValueError("resnet and swin only have 4 levels and are 0-indexed")

        if backbone_name in supported_backbones:
            backbone = backbone_map[backbone_name](
                # Only download the pretrained weights with the main process; the other
                # processes will load the weights in a later step
                pretrain=is_main_process(),
                remove_top=True,  # remove the classification head
                norm_layer=batch_norm,
            )
        else:
            raise NotImplementedError(f"currently only supports {supported_backbones}")

        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        assert backbone_name not in (
            "resnet18",
            "resnet34",
        ), "Only resnet50 and resnet101 are available."
        # assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]

        # Extract the number of output channels for each level of the backbone;
        # resnet has base_chs = [64, 128, 256, 512] and for resnet50 and greater these
        # base channels are multiplied by an expansion factor of 4;
        # see models.backbones.resnet.base_chs for more info
        bb_output_chs = backbone.base_chs

        if backbone_name in {"resnet50", "resnet101"}:
            expansion = 4
            bb_output_chs = [ch * expansion for ch in bb_output_chs]
        else:
            # expansion = 1 for rn 34 and rn 18
            raise NotImplementedError(f"Backbone {supported_backbones} not implemented")

        # Thefore for resnet50 and greater bb_output_chs = [256, 512, 1024, 2048]

        # Extract the number of output channels for each level of the backbone specified;
        bb_output_chs = np.array(bb_output_chs)[bb_level_inds]

        # NOTE: DETR only uses the final output of the backbone while dino fuses
        #       multiple level outputs (intermediate layers)

        # Initialize the backbone base class; NOTE: is this class really needed?
        super().__init__(backbone, bb_output_chs, bb_level_inds, train_backbone)


class Joiner(nn.Sequential):
    """Propagate the input images through the backbone, extract intermediate outputs,
    and create positional encodings for each feature map obtained by the backbone
    """

    def __init__(
        self, backbone: nn.Module, position_embedding: PositionEmbeddingSineHW
    ):
        """Initialize the Joiner module which inherits from Sequenital

        i.e., Sequential takes nn.Modules as inputs and calls them one after another

        Args:

        """
        # self[0] = backbone and self[1] position_embedding
        super().__init__(backbone, position_embedding)

    def forward(
        self, tensor_list: NestedTensor
    ) -> tuple[list[NestedTensor], list[Tensor]]:
        """Propagate the input images through the backbone, extract intermediate outputs,
        and create positional encodings for each feature map obtained by the backbone

        Args:
            tensor_list: a NestedTensor of the input tensor and padding mask

        Returns:
            a tuple of
                1. a list of intermediate feature_maps extracted from the backbone
                2. a list of positional encodings for each feature map
        """
        # self[0] = backbone; pass the nested tensor into the Backbone
        # and return the intermediate and final feature_maps specified by bb_level_inds
        feature_maps = self[0](tensor_list)

        # a list to store just the nested tensors; TODO: again, might not need to do this with my implementation
        feat_maps_list: list[NestedTensor] = []

        # stores the positional encodings for each feature map
        feat_maps_positionals = []

        for (
            name,
            feat_map,
        ) in (
            feature_maps.items()
        ):  # TODO since my backbone returns a list, I don't think I need items
            feat_maps_list.append(feat_map)

            # Calculate the positional encodings for each intermediate output;
            # self[1] = positional embedding module
            feat_maps_positionals.append(self[1](feat_map).to(feat_map.tensors.dtype))

        return feat_maps_list, feat_maps_positionals


def build_dino_backbone(
    backbone_name: str = "resnet50",
    hidden_dim: int = 256,
    temperature_h: int = 40,
    temperature_w: int = 40,
    normalize: bool = True,
    bb_level_inds: list[int] = [1, 2, 3],
):
    """Build the backbone class specfiically for the DINO detector.

    Args:
        hidden_dim: TODO
        temperature_h: The height temperature of the positional embedding equation (attention is all you need)
        temperature_w The width temperature of the positional embedding equation (attention is all you need)
        normalize: whether to normalize and scale positional coordinates from [0, 2pi)
    """
    # Initalize the positional embedding module to create positional embeddings
    # for the images patches (output of backbone) before passing into the transformer encoder
    positional_embedding = PositionEmbeddingSineHW(
        num_pos_feats=hidden_dim // 2,
        temperature_h=temperature_h,
        temperature_w=temperature_w,
        normalize=normalize,
    )

    # build the resnet backbone; NOTE: the Backbone class is very specific to resnet
    if "resnet" in backbone_name:
        backbone = Backbone(
            backbone_name=backbone_name,
            train_backbone=True,
            bb_level_inds=bb_level_inds,
            batch_norm=FrozenBatchNorm2d,  # do not update the bn statistics while training; only use the pretrained bn statistics
        )
    elif "swin" in backbone_name:
        # TODO: build the swin backbone at a later point
        raise NotImplementedError
    else:
        raise ValueError(f"Backbone {backbone_name} not implemented")

    # Module which calls the backbone, extracts the feature maps, and creates
    # positional embeddings for each feature map
    model = Joiner(backbone, positional_embedding)

    # Assign the number of output_channels per feature map to the model
    model.bb_out_chs: list[int] = backbone.bb_out_chs

    return model
