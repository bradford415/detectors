from typing import Any

from torch import nn

from detectors.models.backbones import BACKBONE_REGISTRY
from detectors.models.components.rtdetrv2.hybrid_encoder import HybridEncoder
from detectors.models.components.rtdetrv2.rtdetrv2_decoder import RTDETRTransformerv2


class RTDETR(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


def build_rtdetrv2(detector_params: dict[str, Any]):
    """TODO"""

    detector_components = detector_params["RTDETRv2"]
    backbone_name = detector_components["backbone"]
    encoder_name = detector_components["encoder"]
    decoder_name = detector_components["decoder"]

    backbone = BACKBONE_REGISTRY.get(backbone_name, None)(
        **detector_params[backbone_name]
    )
    if backbone_name is None:
        raise ValueError("")

    if encoder_name == "HybridEncoder":
        encoder = HybridEncoder(**detector_params[encoder_name])
    else:
        raise ValueError(
            f"Error: only the HybridEncoder is supported; got {encoder_name}"
        )

    if decoder_name == "RTDETRTransformerv2":
        decoder = RTDETRTransformerv2(**detector_params[decoder_name])
    else:
        raise ValueError(
            f"Error: only the RTDETRTransformerv2 is supported for the decoder; got {decoder_name}"
        )

    model = RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)

    return model
