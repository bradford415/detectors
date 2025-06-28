# Utility functions to reproduce the results from experimentss
import json
import logging
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import yaml

from detectors.utils import distributed

log = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Set the seed for the sources of randomization; allows for more reproducible results"""

    # applying a slightly different seed for each process allows for diversity across
    # proccesses such that they do not produce the same random numbers; the same seed
    # could to poor training behavior
    seed = seed + distributed.get_global_rank()

    # sets the seed for PyTorch's CPU and CUDA random number generators.
    torch.manual_seed(seed)
    
    # sets numpy's RNG
    np.random.seed(seed)

    # sets pure python's RNG
    random.seed(seed)


def model_info(model):
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    log.info("Model params: %.2f M", (model_size / 1024 / 1024))


def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model params: %.2f M", (num_params / 1024 / 1024))


def save_configs(
    config_dicts: Sequence[tuple[dict, str]],
    output_path: Path,
    solver_dict: Optional[tuple[dict, str]] = None,
):
    """Save configuration dictionaries as yaml files in the output; this allows
    reproducibility of the model by saving the parameters used

    Args:
        config_dicts: Dictionaries containing the configuration parameters used to
                      to run the script (e.g., the base config and the model config)
        save_names: File names to save the reproducibility results as; must end with .yaml
        output_path: Output directory to save the configuration files; it's recommened to have the
                     final dir named "reproduce"
    """

    output_path.mkdir(parents=True, exist_ok=True)

    for config_dict, save_name in config_dicts:
        with open(output_path / save_name, "w") as f:
            yaml.dump(
                config_dict, f, indent=4, sort_keys=False, default_flow_style=False
            )

    if solver_dict is not None:
        # Save solver parameters (optimizer, lr_scheduler, etc.)
        param_dict, save_name = solver_dict
        with open(output_path / save_name, "w") as f:
            json.dump(param_dict, f, indent=4)
