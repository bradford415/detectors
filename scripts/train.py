import datetime
import logging
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader

from detectors.data.coco_minitrain import build_coco_mini
from detectors.data.coco_utils import get_coco_object
from detectors.data.collate_functions import collate_fn
from detectors.models.backbones import backbone_map
from detectors.models.darknet import Darknet
from detectors.models.losses.yolo_loss import Yolo_loss, YoloV4Loss
from detectors.models.yolov4 import YoloV4
from detectors.trainer import Trainer
from detectors.utils import misc, schedulers

detectors_map: Dict[str, Any] = {"yolov4": YoloV4}

dataset_map: Dict[str, Any] = {"CocoDetectionMiniTrain": build_coco_mini}

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

loss_map = {
    "cross_entropy": nn.CrossEntropyLoss(),
}

scheduler_map = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "lambda_lr": torch.optim.lr_scheduler.LambdaLR,  # Multiply the initial lr by a factor determined by a user-defined function; it does NOT multiply the factor by the current lr, always the initial lr
}

# Initialize the root logger
log = logging.getLogger(__name__)


def main(base_config_path: str, model_config_path):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired configuration file
        model_config_path: path to the detection model configuration file

    """
    # Load configuration files
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Initialize paths
    output_path = (
        Path(base_config["output_path"])
        / base_config["exp_name"]
        / f"{datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "training.log"

    # Dictionary of logging parameters; used to log training and evaluation progress after certain intervals
    logging_intervals = base_config["logging"]

    # Configure logger that prints to a log file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    log.info("Initializing...\n")

    # Apply reproducibility seeds
    misc.reproducibility(**base_config["reproducibility"])

    # Set cuda parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {"batch_size": base_config["train"]["batch_size"], "shuffle": True}
    val_kwargs = {
        "batch_size": base_config["validation"]["batch_size"],
        "shuffle": False,
    }

    if use_cuda:
        log.info("Using %d GPU(s): ", len(base_config["cuda"]["gpus"]))
        for gpu in range(len(base_config["cuda"]["gpus"])):
            log.info("    -%s", torch.cuda.get_device_name(gpu))

        cuda_kwargs = {
            "num_workers": base_config["dataset"]["num_workers"],
            "pin_memory": True,
        }

        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)
    else:
        log.info("Using CPU")

    dataset_kwargs = {"root": base_config["dataset"]["root"]}
    dataset_train = dataset_map[base_config["dataset_name"]](
        dataset_split="train", debug_mode=base_config["debug_mode"], **dataset_kwargs
    )
    dataset_val = dataset_map[base_config["dataset_name"]](
        dataset_split="val", debug_mode=base_config["debug_mode"], **dataset_kwargs
    )

    # drop_last is true becuase the loss function intializes masks with the first dimension being the batch_size;
    # during the last batch, the batch_size will be different if the length of the dataset is not divisible by the batch_size
    dataloader_train = DataLoader(
        dataset_train,
        collate_fn=collate_fn,
        drop_last=True,
        **train_kwargs,
    )
    dataloader_val = DataLoader(
        dataset_val,
        collate_fn=collate_fn,
        drop_last=True,
        **val_kwargs,
    )

    # Initalize model components
    backbone = backbone_map[model_config["backbone"]["name"]](
        pretrain=model_config["backbone"]["pretrained"],
        remove_top=model_config["backbone"]["remove_top"],
    )

    model_components = {
        "backbone": backbone,
        "num_classes": 80,
        **model_config["priors"],
    }

    # Initialize detection model and transfer to GPU
    model = detectors_map[model_config["detector"]](**model_components)
    # model = Darknet("scripts/configs/yolov4.cfg")
    # model = Yolov4_pytorch(n_classes=80,inference=False)
    model.to(device)

    ## TODO: Apply weights init

    # For the YoloV4Loss function, if the batch size is different than the
    criterion = YoloV4Loss(
        anchors=model_config["priors"]["anchors"],
        batch_size=base_config["train"]["batch_size"],
        device=device,
    )

    # criterion = Yolo_loss(device=device, batch=base_config["train"]["batch_size"], n_classes=80)

    # Extract the train arguments from base config
    train_args = base_config["train"]

    # Extract the learning parameters such as lr, optimizer params and lr scheduler
    learning_config = train_args["learning_config"]
    learning_params = base_config[learning_config]

    # Initialize training objects
    optimizer, lr_scheduler = _init_training_objects(
        model_params=model.parameters(),
        optimizer=learning_params["optimizer"],
        scheduler=learning_params["lr_scheduler"],
        learning_rate=learning_params["learning_rate"],
        weight_decay=learning_params["weight_decay"],
    )

    trainer = Trainer(
        output_path=str(output_path), device=device, logging_intervals=logging_intervals
    )

    ## TODO: Implement checkpointing somewhere around here (or maybe in Trainer)

    # Build trainer args used for the training
    trainer_args = {
        "model": model,
        "criterion": criterion,
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "class_names": dataset_train.class_names,
        **train_args["epochs"],
    }
    trainer.train(**trainer_args)


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
    lr_scheduler = scheduler_map[scheduler](
        optimizer, schedulers.burnin_schedule_modified
    )
    return optimizer, lr_scheduler


if __name__ == "__main__":
    Fire(main)
