# Utility functions to reproduce the results from experimentss
import json
import logging
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Dict

import numpy as np
import torch

log = logging.getLogger(__name__)


def reproducibility(seed: int) -> None:
    """Set the seed for the sources of randomization. This allows for more reproducible results"""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def model_info(model):
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    log.info("Model params: %.2f M", (model_size / 1024 / 1024))


def save_configs(
    config_dicts: Iterable[Dict], save_names: Iterable[str], output_path: Path
):
    """Save configuration dictionaries as json files in the output; this allows
    reproducibility of the model by saving the parameters used

    Args:
        config_dicts: Dictionaries containing the configuration parameters used to
                      to run the script (e.g., the base config and the model config)
        save_names: File names to save the reproducibility results as; must end with .json
        output_path: Output directory to save the configuration files; it's recommened to have the
                     final dir named "reproduce"
    """
    assert len(config_dicts) == len(save_names)

    output_path.mkdir(parents=True, exist_ok=True)

    for config_dict, save_name in zip(config_dicts, save_names):
        with open(output_path / save_name, "w") as f:
            json.dump(config_dict, f)
