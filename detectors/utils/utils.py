import random
from pathlib import Path

import numpy as np
import torch


def load_text_file(file_name: Path, mode: str = "r") -> str:
    """Load text file

    Args:
        file_name: path to file name
        mode: mode to open the file
    """
    with open(str(file_name), mode=mode, encoding="utf-8") as f:
        text = f.read()

    return text


def reproducibility(seed: int) -> None:
    """Set the seed for the sources of randomization. This allows for more reproducible results"""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
