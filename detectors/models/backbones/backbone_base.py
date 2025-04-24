import torch
from torch import nn

from detectors.models.layers.positional import PositionEmbeddingSineHW


class BackboneBase(nn.Module):
    """Base class for backbone networks, specifially for the DINO detector"""

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_indices: list,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update(
                {
                    "layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(
                        layer_index
                    )
                }
            )

        # if len:
        #     if use_stage1_feature:
        #         return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        #     else:
        #         return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        # else:
        #     return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out


def build_backbone(
    hidden_dim: int,
    temperature_h: int = 40,
    temperature_w: int = 40,
    normalize: bool = True,
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
    
    ################# START HERE continue on with building the backbone ##############

    return 0
