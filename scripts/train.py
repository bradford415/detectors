from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from detectors.data.coco_minitrain import build_coco_mini
from detectors.data.coco_utils import get_coco_object
from detectors.models.yolov4 import YoloV4
from detectors.trainer import Trainer
from detectors.utils import utils

model_map: Dict[str, Any] = {"YoloV4": YoloV4}

dataset_map: Dict[str, Any] = {"CocoDetectionMiniTrain": build_coco_mini}

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

scheduler_map = {"step_lr": torch.optim.lr_scheduler.StepLR}


def collate_fn(batch):

    batch = list(zip(*batch))

    # This is what will be returned in the main for loop (samples, targets)
    return tuple(batch)

def main(base_config_path: str):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired configuration file

    """

    print("Initializations...\n")

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Apply reproducibility seeds
    utils.reproducibility(**base_config["reproducibility"])

    # Set cuda parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {"batch_size": base_config["train"]["batch_size"], "shuffle": True}
    val_kwargs = {
        "batch_size": base_config["validation"]["batch_size"],
        "shuffle": False,
    }

    if use_cuda:
        print(f"Using {len(base_config['gpus'])} GPU(s): ")
        for gpu in range(len(base_config["gpus"])):
            print(f"    -{torch.cuda.get_device_name(gpu)}")
        cuda_kwargs = {
            "num_workers": base_config["cuda"]["workers"],
            "pin_memory": True,
        }

        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)
    else:
        print("Using CPU")


    dataset_kwargs = base_config["dataset"]
    dataset_train = dataset_map[base_config["dataset_name"]](
        dataset_split="train", **dataset_kwargs
    )
    dataset_val = dataset_map[base_config["dataset_name"]](
        dataset_split="val", **dataset_kwargs
    )

    dataloader_train = DataLoader(
        dataset_train, num_workers=base_config["cuda"]["num_workers"], collate_fn=collate_fn, **train_kwargs
    )
    dataloader_test = DataLoader(
        dataset_val, num_workers=base_config["cuda"]["num_workers"], collate_fn=collate_fn, **val_kwargs
    )

    # Return the Coco object from PyCocoTools
    coco_api = get_coco_object(dataset_train)

    # Initalize model
    model = resnet50()  # Using temp resnet50 model
    criterion = nn.CrossEntropyLoss()

    # Extract the train arguments from base config
    train_args = {**base_config["train"]}

    # Initialize training objects
    optimizer, lr_scheduler = _init_training_objects(
        model_params=model.parameters(),
        optimizer=train_args["optimizer"],
        scheduler=train_args["scheduler"],
        learning_rate=train_args["learning_rate"],
        weight_decay=train_args["weight_decay"],
        lr_drop=train_args["lr_drop"],
    )

    runner = Trainer(output_path=base_config["output_path"])

    ## TODO: Implement checkpointing somewhere around here (or maybe in Trainer)

    model = base_config["model"]

    # Build trainer args used for the training
    trainer_args = {
        "model": model,
        "criterion": criterion,
        "data_loader": dataloader_train,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "device": device,
        **train_args["epochs"],
    }
    runner.train(**trainer_args)


def _init_training_objects(
    model_params: Iterable,
    optimizer: str = "sgd",
    scheduler: str = "step_lr",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    lr_drop: int = 200,
):
    optimizer = optimizer_map[optimizer](
        model_params, lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = scheduler_map[scheduler](optimizer, lr_drop)
    return optimizer, lr_scheduler


if __name__ == "__main__":
    Fire(main)
