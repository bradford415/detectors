# Script to visualize the dataloders used during training and validation;
# this is useful to verify the inputs and labels to the model are as excepted especially during data augmentation
import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from detectors.solvers import schedulers

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import matplotlib.pyplot as plt
import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader

from detectors.data.datasets.coco import build_coco
from detectors.data.coco_utils import get_coco_object
from detectors.data.collate_functions import collate_fn
from detectors.losses import loss_map
from detectors.models import Yolov3, Yolov4
from detectors.models.backbones import backbone_map
from detectors.models.backbones.darknet import Darknet
from detectors.trainer import Trainer
from detectors.utils import reproduce
from detectors.visualize import visualize_norm_img_tensors

dataset_map: Dict[str, Any] = {"CocoDetection": build_coco}

# Initialize the root logger
log = logging.getLogger(__name__)


def main(train_config_path: str):
    """Entrypoint for the project

    Args:
        train_config_path: path to the based configuration file which will be used at training time
    """
    # Load configuration file
    with open(train_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Initialize paths
    output_path = Path("output/visuals")

    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "training.log"

    # Configure logger that prints to a log file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    log.info("initializing...\n")
    log.info("writing outputs to %s", str(output_path))

    # Apply reproducibility seeds
    reproduce.reproducibility(seed=42)

    if torch.mps.is_available():
        base_config["dataset"]["root"] = base_config["dataset"]["root_mac"]

    dataset_kwargs = {"root": base_config["dataset"]["root"]}
    dataset_train = dataset_map[base_config["dataset_name"]](
        dataset_split="train", dev_mode=False, **dataset_kwargs
    )
    dataset_val = dataset_map[base_config["dataset_name"]](
        dataset_split="val", dev_mode=False, **dataset_kwargs
    )

    # number of images for each label; at least one label is in the image
    class_labels = dataset_train.coco.getCatIds()
    class_count = [
        len(dataset_train.coco.getImgIds(catIds=[label])) for label in class_labels
    ]
    # class_count = [count for count in class_count if count != 0]

    _, ax = plt.subplots(1, figsize=(14, 4))

    ax.bar(class_labels, class_count)
    ax.set_xticks(class_labels)
    ax.set_xticklabels(dataset_train.class_names, rotation=90)

    ax.set_xlabel("label")
    ax.set_ylabel("number of images")

    plt.savefig(output_path / "train_distribution.jpg", bbox_inches="tight")


if __name__ == "__main__":
    Fire(main)
