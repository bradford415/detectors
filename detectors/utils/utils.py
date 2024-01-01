import random
from pathlib import Path

import numpy as np
import torch
import torchvision


def load_text_file(file_name: Path, mode: str = "r") -> str:
    """Load text file

    Args:
        file_name: path to file name
        mode: mode to open the file
    """
    with open(str(file_name), mode=mode, encoding="utf-8") as f:
        text = f.read()

    return text


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    # if version.parse(torchvision.__version__) < version.parse('0.7'):
    #     if input.numel() > 0:
    #         return torch.nn.functional.interpolate(
    #             input, size, scale_factor, mode, align_corners
    #         )

    #     output_shape = _output_size(2, input, size, scale_factor)
    #     output_shape = list(input.shape[:-2]) + list(output_shape)
    #     return torchvision.ops.misc._new_empty_tensor(input, output_shape)
    # else:
    return torchvision.ops.misc.interpolate(
        input, size, scale_factor, mode, align_corners
    )


def reproducibility(seed: int) -> None:
    """Set the seed for the sources of randomization. This allows for more reproducible results"""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
