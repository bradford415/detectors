# Script to visualize the dataloders used during training and validation;
# this is useful to verify the inputs and labels to the model are as excepted especially during data augmentation
import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from detectors.solvers import schedulers

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

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

detectors_map: Dict[str, Any] = {"yolov3": Yolov3, "yolov4": Yolov4}

dataset_map: Dict[str, Any] = {"CocoDetection": build_coco}

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

scheduler_map = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "lambda_lr": torch.optim.lr_scheduler.LambdaLR,  # Multiply the initial lr by a factor determined by a user-defined function; it does NOT multiply the factor by the current lr, always the initial lr
}

# Initialize the root logger
log = logging.getLogger(__name__)


def main(train_config_path: str, num_images: int = 1000, epochs: int = 2):
    """Entrypoint for the project

    Args:
        train_config_path: path to the based configuration file which will be used at training time
    """
    # Load configuration file
    with open(train_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Initialize paths
    output_path = (
        Path("output/visualize-dataloaders")
        / f"{datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
    )

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

    # drop_last is true becuase the loss function intializes masks with the first dimension being the batch_size;
    # during the last batch, the batch_size will be different if the length of the dataset is not divisible by the batch_size
    dataloader_train = DataLoader(
        dataset_train,
        collate_fn=collate_fn,
        batch_size=4,
        drop_last=True,
        shuffle=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        collate_fn=collate_fn,
        batch_size=4,
        drop_last=True,
        shuffle=False,
    )

    class_names = dataset_train.class_names

    log.info("plotting train images")

    # Visualize train loader
    for epoch in range(1, epochs + 1):
        for step, (samples, targets, annotations) in enumerate(dataloader_train, 1):
            visualize_norm_img_tensors(
                samples,
                targets,
                annotations,
                step,
                class_names,
                output_path / "dataloader-train" / f"epoch-{epoch}",
            )
            log.info("saved image %d/%d", step, len(dataloader_train))
            if step == num_images:
                break

    # Visualize val loader
    for step, (samples, targets, annotations) in enumerate(dataloader_val, 1):
        visualize_norm_img_tensors(
            samples,
            targets,
            annotations,
            step,
            class_names,
            output_path / "dataloader-val",
        )
        log.info("saved image %d/%d", step, len(dataloader_val))
        if step == num_images:
            break


if __name__ == "__main__":
    Fire(main)
