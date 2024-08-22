from pathlib import Path

import matplotlib.pyplot as plt
import torch

from detectors.data.transforms import UnNormalize


def visualize_norm_img_tensors(img_tensors: torch.Tensor, output_dir: Path):
    """TODO"""

    un_norm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # assert torch.max(img_tensors) <= 1.0

    img_tensors = img_tensors.to("cpu")

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1)
    for index, image in enumerate(img_tensors):
        image = un_norm(image)
        image = image.permute(1, 2, 0)
        # ax.imshow(image.to(dtype=torch.uint8), vmin=0, vmax=255)
        ax.imshow(image)
        fig.savefig(f"{output_dir}/image_tensor_{index}.png")
