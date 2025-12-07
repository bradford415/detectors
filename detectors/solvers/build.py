from typing import Optional, Sequence

import torch
from torch import nn

from detectors.losses import Yolov3Loss, Yolov4Loss, create_dino_loss

scheduler_map = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "lambda_lr": torch.optim.lr_scheduler.LambdaLR,  # Multiply the initial lr by a factor determined by a user-defined function; it does NOT multiply the factor by the current lr, always the initial lr
}

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

loss_map = {"yolov3": Yolov3Loss, "yolov4": Yolov4Loss, "dino": create_dino_loss}


def get_optimizer_params(
    model: torch.nn.Module, strategy: str = "all", backbone_lr: Optional[float] = None
):
    """Extract the traininable parameters from the model in different groupes; allows us to spceicfy
    different learning rates for different groups of parameters

    Strategies:
        all: extract all traininable parameters from the model and use the same learning rate
        separate_backbone: separate the backbone parameters from the rest of the model and use
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

        # sepeerate the backbone and non-backbone parameters
        non_backbone_params = [
            param
            for name, param in model.named_parameters()
            if "backbone" not in name and param.requires_grad
        ]
        backbone_params = [
            param
            for name, param in model.named_parameters()
            if "backbone" in name and param.requires_grad
        ]

        param_dicts = [
            {"params": non_backbone_params},  # lr used is set in the optimizer config
            {"params": backbone_params, "lr": backbone_lr},
        ]

    return param_dicts


def create_loss(
    model_name: str,
    model: nn.Module,
    num_classes: int,
    device: torch.device,
    base_config: dict,
):
    """Creates the loss function based on the model name

    Args:
        model_name: the name of the model
        model: the pytorch model being trained; if using ddp pass the pointer to the underlying model
        num_classes: the number of classes in the dataset
        base_config: the base configuration dictionary containing all parameters
    """

    # initalize loss with specific args
    if model_name == "yolov3":
        num_anchors = model.yolo_layers[0].num_anchors
        criterion = loss_map[model_name](num_anchors=num_anchors, device=device)
    elif model_name == "dino":
        num_decoder_layers = base_config["detector"]["transformer"][
            "num_decoder_layers"
        ]
        criterion = loss_map[model_name](
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
            aux_loss=base_config["aux_loss"],
            two_stage_type=base_config["detector"]["two_stage"]["type"],
            loss_args=base_config["params"]["loss_weights"],
            matcher_args=base_config["params"]["matcher"],
            device=device,
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
        backbone_lr: the learning rate to use for the backbone parameters if using the
                     separate_backbone strategy
    """
    optimizer_name = optimizer_config["name"]
    scheduler_name = scheduler_config["name"]

    optimizer_params = optimizer_config["params"]
    scheduler_params = scheduler_config["params"]

    model_params = get_optimizer_params(
        model, strategy=parameter_strategy, backbone_lr=backbone_lr
    )

    # Build optimizer
    if optimizer_name in optimizer_map:
        optimizer = optimizer_map[optimizer_name](model_params, **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if scheduler_name in scheduler_map:
        scheduler = scheduler_map[scheduler_name](optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

    return optimizer, scheduler
