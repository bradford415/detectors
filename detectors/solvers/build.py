import re
from asyncio.unix_events import BaseChildWatcher
from typing import Any, Optional, Sequence

import torch
from torch import nn

from detectors.losses import (
    Yolov3Loss,
    Yolov4Loss,
    create_dino_loss,
    create_rtdetrv2_loss,
)
from detectors.solvers.schedulers import create_multistep_lr_scheduler_w_warmup

scheduler_map = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "multistep_lr": create_multistep_lr_scheduler_w_warmup,
    "lambda_lr": torch.optim.lr_scheduler.LambdaLR,  # Multiply the initial lr by a factor determined by a user-defined function; it does NOT multiply the factor by the current lr, always the initial lr
}

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def get_backbone_params(model: torch.nn.Module):
    """Extracts the backbone parameters from the model for use in the optimizer

    Args:
        model: the model to extract the backbone parameters from

    Returns:
        a list of the backbone parameters
    """
    backbone_params = [
        param
        for name, param in model.named_parameters()
        if "backbone" in name and param.requires_grad
    ]
    return backbone_params


def get_non_backbone_params(
    model: torch.nn.Module, include_keywords: Optional[Sequence[str]] = None
):
    """Extracts the non-backbone parameters from the model for use in the optimizer

    Args:
        model: the model to extract the non-backbone parameters from

    Returns:
        a list of the non-backbone parameters
    """
    non_backbone_params = []
    for name, param in model.named_parameters():
        if "backbone" not in name and param.requires_grad:
            non_backbone_params.append(param)

    return non_backbone_params


def get_regex_params(model: torch.nn.Module, regex_pattern: str):
    regex_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad and re.search(regex_pattern, name):
            regex_params[name] = param
    return regex_params


def get_optimizer_params(
    model: torch.nn.Module, optmizer_params: dict[str, Any], strategy: str = "all"
):
    """Extract the traininable parameters from the model in different groupes; allows us to spceicfy
    different learning rates for different groups of parameters

    Strategies:
        all: extract all traininable parameters from the model and use the same learning rate
        separate_backbone: separate the backbone parameters from the rest of the model and u
                           a different learning rate for the backbone
    """
    if strategy == "all":
        parameters = [
            param for name, param in model.named_parameters() if param.requires_grad
        ]
        param_dicts = [{"params": parameters}]
    elif strategy == "separate_backbone":

        if backbone_lr is None:
            raise ValueError(
                "backbone_lr must be specified when using separate_backbone strategy"
            )

        # seperate the backbone and non-backbone parameters
        non_backbone_params = get_non_backbone_params(model)
        backbone_params = get_backbone_params(model)

        param_dicts = [
            {"params": non_backbone_params},  # lr used is set in the optimizer config
            {"params": backbone_params, "lr": backbone_lr},  # TODO fix this
        ]
    elif strategy == "rtdetrv2":
        # extract the decoder and decoder parameters which have norm or bn in their name
        encoder_decoder_norm_params = get_regex_params(
            model, regex_pattern=r"^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$"
        )

        # extract the backbone parameters that do not contain norm (bn is allowed though I don't think batch norm is used much anymore)
        backbone_non_norm_params = get_regex_params(
            model, regex_pattern=r"^(?=.*backbone)(?!.*norm).*$"
        )

        # determine the remaining trainable parameters that have not been assigned to an optmizer group yet
        visited_params = list(encoder_decoder_norm_params.keys()) + list(
            backbone_non_norm_params.keys()
        )
        all_param_names = [k for k, v in model.named_parameters() if v.requires_grad]
        unseen_params = set(all_param_names) - set(visited_params)

        remaining_params = {
            k: v
            for k, v in model.named_parameters()
            if v.requires_grad and k in unseen_params
        }

        # verify that all parameters have been assigned to a group
        visited_params += list(remaining_params.keys())
        assert len(visited_params) == len(
            all_param_names
        ), "Some parameters were not assigned to any optimizer parameter group"

        param_dicts = [
            {
                "params": encoder_decoder_norm_params.values(),
                "weight_decay": optmizer_params["encoder_decoder"]["weight_decay"],
            },  # lr used is set in the optimizer config
            {
                "params": backbone_non_norm_params.values(),
                "lr": optmizer_params["backbone"]["lr"],
            },
            {"params": remaining_params.values()},
        ]

    return param_dicts


def create_loss(
    model_name: str,
    num_classes: int,
    device: torch.device,
    base_config: dict,
):
    """Creates the loss function based on the model name

    Args:
        model_name: the name of the model
        num_classes: the number of classes in the dataset
        base_config: the base configuration dictionary containing all parameters
    """

    # initalize loss with specific args
    if model_name == "yolov3":
        # TODO: fix this to properly work with yolo (num_anchors)
        criterion = Yolov3Loss(num_anchors=base_config["num_anchors"], device=device)
    elif model_name == "dino":
        num_decoder_layers = base_config["detector"]["transformer"][
            "num_decoder_layers"
        ]
        criterion = create_dino_loss(
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
            aux_loss=base_config["aux_loss"],
            two_stage_type=base_config["detector"]["two_stage"]["type"],
            loss_args=base_config["params"]["loss_weights"],
            matcher_args=base_config["params"]["matcher"],
            device=device,
        )
    elif model_name == "rtdetrv2":

        criterion_params = base_config["criterion"]

        criterion = create_rtdetrv2_loss(
            matcher_params=criterion_params["matcher"]["params"],
            weight_dict=criterion_params["weight_dict"],
            losses=criterion_params["losses"],
            alpha=criterion_params["alpha"],
            gamma=criterion_params["gamma"],
        )
    else:
        ValueError(f"loss function for {model_name} not implemented")

    return criterion


def build_solvers(
    model: torch.nn.Module,
    optimizer_config: dict[str, any],
    scheduler_config: dict[str, any],
    parameter_strategy: str = "all",
    backbone_lr: Optional[float] = None,
):
    """Builds the optimizer and learning rate scheduler based on the provided parameters
    from solver.config

    Args:
        model: the model to train
        optimizer_params: the parameters used to build the optimizer
        scheduler_params: the parameters used to build the learning rate scheduler
        parameter_strategy: the strategy to use for extracting the parameters from the model
    """
    optimizer_name = optimizer_config["name"]
    scheduler_name = scheduler_config["name"]

    optimizer_params = optimizer_config["params"]
    scheduler_params = scheduler_config["params"]

    model_param_dict = get_optimizer_params(
        model, optmizer_params=optimizer_params, strategy=parameter_strategy
    )

    # Build optimizer
    if optimizer_name in optimizer_map:
        optimizer = optimizer_map[optimizer_name](
            model_param_dict, **optimizer_params["params"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if scheduler_name in scheduler_map:
        scheduler = scheduler_map[scheduler_name](optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

    return optimizer, scheduler
