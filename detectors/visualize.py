import random
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullLocator
from PIL import Image

from detectors.data.transforms import Unnormalize
from detectors.utils.box_ops import rescale_boxes
from detectors.utils.misc import to_cpu

matplotlib.use("Agg")


def visualize_norm_img_tensors(
    img_tensors: torch.Tensor,
    targets: list[dict],
    classes: list[str],
    output_dir: Path,
    annotations,
):
    """Visualizes the boxes of augmented images just before the input of the model; this helps
    manually verify the data augmentation on the images and labels is accurate

    Args:
        img_tensors: tensor of normalized images (b, c, h, w)
        targets: list of dicts containing at least the image bboxes; bboxes format (cx, cy, w, h)
        classes: list of unique class names by label index
        output_dir: Path to save the outputs
    """
    # assert img_tensors.shape[0] == targets.shape[0]

    un_norm = Unnormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # assert torch.max(img_tensors) <= 1.0

    img_tensors = to_cpu(img_tensors)

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    # for target in targets:
    #    labels += target["labels"]
    unique_classes = np.unique(np.array(targets[:, 1]))
    num_unique_classes = len(unique_classes)

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, num_unique_classes)]
    bbox_colors = random.sample(colors, num_unique_classes)

    for img_index, image in enumerate(img_tensors):
        fig, ax = plt.subplots(1, 1)

        img_h, img_w = image.shape[1:]

        image = un_norm(image)
        image = image.permute(1, 2, 0)
        # ax.imshow(image.to(dtype=torch.uint8), vmin=0, vmax=255)
        ax.imshow(image)

        for img_idx, label, cx, cy, w, h in targets[targets[:, 0] == img_index]:

            # box coords are normalize [0-1] so we need to scale them to the input size
            cx *= img_w
            cy *= img_h
            w *= img_w
            h *= img_h

            tl_x = cx - w // 2
            tl_y = cy - h // 2

            color = bbox_colors[int(np.where(unique_classes == int(label))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle(
                (tl_x, tl_y), w, h, linewidth=2, edgecolor=color, facecolor="none"
            )
            # Add the bbox to the plot
            # TODO: need to figure out if I need to clip these boxes because sometimes the figure is too large
            # Might be in rescale_boxes() in github code
            ax.add_patch(bbox)
            plt.text(
                tl_x,
                tl_y,
                s=f"{classes[int(label)]}",
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        plt.axis("off")
        fig.savefig(
            f"{output_dir}/image_tensor_{int(img_idx)}.png",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()


def plot_all_detections(img_detections, classes: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (image_path, detections) in enumerate(img_detections):
        plot_detections(
            image_path,
            detections,
            classes,
            save_name=output_dir / f"detection_{index}.jpg",
        )


def plot_detections(image_path: str, detections, classes: List[str], save_name: str):
    """Visualizes the augmented images just before the input of the model; this helps
    manually verify the data augmentation on the images and labels is accurate

    Args:
        image_path: the image path of the image being detected
        detections: detections after non-max suppression (num_detections, 6);
                    detected boxes should be (tl_x, tl_y, br_x, br_y, conf, cls)
        targets: Dictionaries containing at least the ground truth bboxes and label for each
                 object; bboxes should be (tl_x)each element of the list is an image's labels
        classes: list of unique class names by label index
        output_dir: Path to save the outputs
    """
    img = np.array(Image.open(image_path).convert("RGB"))

    plt.figure()
    fig, ax = plt.subplots(1)

    ax.imshow(img)

    if isinstance(detections, torch.Tensor):
        detections = detections.numpy()

    detections[:, 0]
    labels = detections[:, -1].astype(np.uint8)
    unique_classes = np.unique(np.array(labels))
    num_unique_classes = len(unique_classes)

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, num_unique_classes)]
    bbox_colors = random.sample(colors, num_unique_classes)
    try:
        for tl_x, tl_y, br_x, br_y, conf, cls_pred in detections:
            # print(f"tl_x: {tl_x} tl_y: {tl_y} br_x: {br_x} br_y: {br_y} ")
            # fig, ax = plt.subplots(1, 1)

            if tl_x < -1000.0 or tl_y < -1000.0 or br_x > 10000.0 or br_y > 10000.0:
                continue

            box_width = br_x - tl_x
            box_height = br_y - tl_y

            color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]

            # Create a Rectangle patch
            bbox = patches.Rectangle(
                (tl_x, tl_y),
                box_width,
                box_height,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            # Add the bbox to the plot
            ax.add_patch(bbox)
            plt.text(
                tl_x,
                tl_y,
                s=f"{classes[int(cls_pred)]}: {conf:.2f}",
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

        plt.axis("off")
        fig.savefig(save_name, bbox_inches="tight", pad_inches=0.0)
        plt.close()
    except:
        breakpoint()
