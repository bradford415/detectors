"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import math
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from detectors.utils import (
    distributed,
)


# TODO: do i need to add a parameter to initalize the number of steps if resuming?
class ModelEMA:
    """
    Brad's Notes: Exponential Moving Average (EMA) keeps a separate “shadow” copy of the model's weights.
    At each training step, this shadow copy is updated using a decay factor, so recent weights
    influence the EMA more than older ones.
    Over time, the contribution of earlier model weights decays exponentially, meaning older
    weights matter less while more recent weights dominate.

    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.

    Source: https://github.com/lyuwenyu/RT-DETR/blob/156ad827e55aba9d809942911f653f1559e8c5af/rtdetrv2_pytorch/src/optim/ema.py#L18
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmups: int = 2000,
    ):
        super().__init__()

        self.module = deepcopy(distributed.de_parallel(model)).eval()
        # if next(model.parameters()).device.type != 'cpu':
        #     self.module.half()  # FP16 EMA

        self.decay = decay
        self.warmups = warmups
        self.updates = 0  # number of EMA updates

        # decay exponential ramp (to help early epochs); this acts as a dynamic scaling factor such that
        # early on the decay factor is low (close to 0) and slowly ramps up to it's desired value;
        # if we started at the initial decay factor (0.9999) then the ema weights would be stuck
        # at the earlier model's main weights for a while
        # (early on in training model weights are randomly initalized and flucuate a lot while it's learning
        #  the initial epochs); once some timesteps have passed, e^-x/warmups will be very low so (1-e^-x/warmups)
        # will virutall be 1, this is the scaling factor that gets multiplied to the desired decay so that
        # the decay is the value we want it to be
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))

        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        """Update the EMA parameters according the formula:
                theta_{EMA} <-- d * theta_{EMA} + (1 - d) * theta_{online}

        If d is 0.999: The new weights only contribute 0.1% to the average. The model remembers 99.9% of its past state.Smoothing Effect: This filtering reduces the impact of outliers and prevents the model from "over-reacting" to the most recent training steps.
        """
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            msd = distributed.de_parallel(model).state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(
        self,
    ):
        return dict(module=self.module.state_dict(), updates=self.updates)

    def load_state_dict(self, state, strict=True):
        self.module.load_state_dict(state["module"], strict=strict)
        if "updates" in state:
            self.updates = state["updates"]

    def forwad(
        self,
    ):
        raise RuntimeError("ema...")

    def extra_repr(self) -> str:
        return f"decay={self.decay}, warmups={self.warmups}"


def create_ema_model(model, ema_params: dict[str, Any]):
    """Initializes the EMA Model

    Args:
        model (_type_): _description_
        ema_params (dict[str, Any]): _description_
    """
    ema_model = ModelEMA(model, ema_params)
    return ema_model
