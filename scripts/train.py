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
from detectors.models.backbones import backbone_map
from detectors.models.losses.yolo_loss import YoloV4Loss, Yolo_loss
from detectors.models.yolov4 import YoloV4, Yolov4_pytorch
from detectors.models.darknet import Darknet
from detectors.trainer import Trainer
from detectors.utils import misc

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

scheduler_map = {"step_lr": torch.optim.lr_scheduler.StepLR}

# Initialize the root logger
log = logging.getLogger(__name__)


## TODO: Move this to a more appropriate spot
def collate_fn(batch: list[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> None:
    """Collect samples appropriately to be used at each iteration in the train loop

    At each train iteration, the DataLoader returns a batch of samples.
    E.g., for images, annotations in train_loader

    Args:
        batch: A batch of samples from the dataset. The batch is a list of
               samples, each sample containg a tuple of (image, image_annotations).
    """

    # Convert a batch of images and annoations [(image, annoations), (image, annoations), ...]
    # to (image, image), (annotations, annotations), ... ; this operation is called iterable unpacking
    images, annotations = zip(*batch)  # images (C, H, W)
    
    # The below padding method is from the DETR repo here:
    # https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/util/misc.py#L307
    
    # Zero pad images on the right and bottom of the image with the max h/w of the batch;
    # this allows us to batch images of different sizes together;
    # in the current implementation, padding should only be applied for the validation set
    channels, max_h, max_w = images[0].shape
    for image in images[1:]:
        if image.shape[1] > max_h:
            max_h = image.shape[1]
        if image.shape[2] > max_w:
            max_w = image.shape[2]
    
    # Initalize tensor of zeros for 0-padding and copy the images into the top_left of each padded batch
    batch_size = len(batch)
    padded_images = torch.zeros(batch_size, channels, max_h, max_w) # (B, C, batch_max_H, batch_max_W)
    for img, padded_img in zip(images, padded_images):
        padded_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

    # (B, C, H, W)
    #images = torch.stack(images, dim=0) # This was written before the padding above

    # This is what will be returned in the main train for loop (samples, targets)
    return padded_images, annotations


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
    #model = Darknet("scripts/configs/yolov4.cfg")
    #model = Yolov4_pytorch(n_classes=80,inference=False)
    model.to(device)

    # For the YoloV4Loss function, if the batch size is different than the
    criterion = YoloV4Loss(
        anchors=model_config["priors"]["anchors"],
        batch_size=base_config["train"]["batch_size"],
        device=device,
    )
    
    #criterion = Yolo_loss(device=device, batch=base_config["train"]["batch_size"], n_classes=80)

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
    lr_scheduler = scheduler_map[scheduler](optimizer, lr_drop)
    return optimizer, lr_scheduler


if __name__ == "__main__":
    Fire(main)
