from typing import Optional, Sequence

import torch

scheduler_map = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "lambda_lr": torch.optim.lr_scheduler.LambdaLR,  # Multiply the initial lr by a factor determined by a user-defined function; it does NOT multiply the factor by the current lr, always the initial lr
}

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


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
        breakpoint()
        optimizer = optimizer_map[optimizer_name](model_params, **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if scheduler_name in scheduler_map:
        scheduler = scheduler_map[scheduler_name](optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

    return optimizer, scheduler
