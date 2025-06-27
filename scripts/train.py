import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
import torch.distributed as dist
import yaml
from fire import Fire
from torch.utils.data import DataLoader

from detectors.data.coco_ds import build_coco
from detectors.data.collate_functions import collate_fn
from detectors.losses import loss_map
from detectors.models import detectors_map
from detectors.models.backbones import backbone_map
from detectors.solvers.build import build_solvers
from detectors.trainer import Trainer
from detectors.utils import reproduce
from detectors.utils.script import initialize_anchors
from detectors.utils import distributed

dataset_map: Dict[str, Any] = {"CocoDetection": build_coco}

# Initialize the root logger
log = logging.getLogger(__name__)


def main(
    base_config_path: str = "scripts/configs/train-coco-default.yaml",
    model_config_path: str = "scripts/configs/yolov3/model-dn53.yaml",
    dataset_root: Optional[str] = None,
    backbone_weights: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired training configuration file; by default the train-coco-default.yaml file is used which
                          trains from scratch (i.e., no pretrained backbone weights or detector weights)
        model_config_path: path to the detection model configuration file; by default the yolov3 base model with a DarkNet53 backbone is used
        dataset_root: path to the the root directory of the dataset; for coco, this is the path to the dir containing the `images` and `annotations` dirs
        backbone_weights: path to the backbone weights; this can be useful when training from scratch
        checkpoint_path: path to the weights of the entire detector model; this can be used to resume training or inference;
                         if this parameter is not None, the backbone_weights will be ignored
    """

    # Load configuration files
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # initalize torch distributed mode by setting the communication between all procceses
    # and assigning the GPU to use for each proccess; 
    # NOTE: in general, each proccess should run most commands in the program except saving to disk
    distributed.init_distributed_mode(backend=base_config["cuda"]["backend"])

    # Override configuration parameters if CLI arguments are provided; this allows external users
    # to easily run the project without messing with the configuration files
    if dataset_root is not None:
        base_config["dataset"]["root"] = dataset_root

    if checkpoint_path is not None:
        base_config["train"]["checkpoint_path"] = checkpoint_path
        base_config["train"]["backbone_weights"] = None
    elif backbone_weights is not None:
        base_config["train"]["backbone_weights"] = backbone_weights

    dev_mode = base_config["dev_mode"]

    if (
        base_config["train"]["checkpoint_path"] is not None
        and base_config["train"]["backbone_weights"] is not None
    ):
        raise ValueError(
            "checkpoint_path and backbone_weights cannot both have a value. Set one of the values to 'null'."
        )

    # Initialize paths
    output_path = (
        Path(base_config["output_dir"])
        / base_config["exp_name"]
        / f"{datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "training.log"

    ############ START HERE - initialize distributed training

    # Save configuration files and parameters
    reproduce.save_configs(
        config_dicts=[
            (base_config, "base_config.yaml"),
            (model_config, "model_config.yaml"),
        ],
        # solver_dict=(solver_config.to_dict(), "solver_config.json"),
        output_path=output_path / "reproduce",
    )

    # Configure logger that prints to a log file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    # TODO: verify this works; Suppress logs from non-zero ranks
    if dist.is_initialized() and dist.get_rank() != 0:
        log.setLevel(logging.WARNING)  # or ERROR to suppress even more

    log.info("initializing...\n")
    log.info("writing outputs to %s", str(output_path))

    if (
        base_config["train"]["checkpoint_path"] is None
        and base_config["train"]["backbone_weights"] is None
    ):
        log.info("\nNOTE: training from scratch; no pretrained weights provided")

    # Apply reproducibility seeds
    reproduce.reproducibility(**base_config["reproducibility"])

    # Extract training and val params
    train_args = base_config["train"]

    batch_size = train_args["batch_size"]
    effective_bs = train_args["effective_batch_size"]
    epochs = train_args["epochs"]

    val_batch_size = train_args["batch_size"]

    if effective_bs % batch_size == 0 and effective_bs >= batch_size:
        grad_accum_steps = effective_bs // batch_size
    else:
        raise ValueError(
            "grad_accum_bs must be divisible by batch_size and greater than or equal to batch_size"
        )

    if dev_mode:
        log.info("NOTE: executing in dev mode")
        batch_size = 4
        grad_accum_steps = 2

    log.info(
        "\neffective_batch_size: %-5d grad_accum_steps: %-5d batch_size: %d",
        effective_bs,
        grad_accum_steps,
        batch_size,
    )

    # Set gpu parameters
    train_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": base_config["dataset"]["num_workers"] if not dev_mode else 0,
    }
    val_kwargs = {
        "batch_size": val_batch_size,
        "shuffle": False,
        "num_workers": base_config["dataset"]["num_workers"] if not dev_mode else 0,
    }

    # Set device specific characteristics
    use_cpu = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Using %d GPU(s): ", len(base_config["cuda"]["gpus"]))
        for gpu in range(len(base_config["cuda"]["gpus"])):
            log.info("    -%s", torch.cuda.get_device_name(gpu))
    elif torch.mps.is_available():
        base_config["dataset"]["root"] = base_config["dataset"]["root_mac"]
        device = torch.device("mps")
        log.info("Using: %s", device)
    else:
        use_cpu = True
        device = torch.device("cpu")
        log.info("Using CPU")

    if not use_cpu:
        gpu_kwargs = {
            "pin_memory": True,
        }

        train_kwargs.update(gpu_kwargs)
        val_kwargs.update(gpu_kwargs)

    dataset_kwargs = {"root": base_config["dataset"]["root"]}
    dataset_train = dataset_map[base_config["dataset_name"]](
        dataset_split="train", dev_mode=dev_mode, **dataset_kwargs
    )
    dataset_val = dataset_map[base_config["dataset_name"]](
        dataset_split="val", dev_mode=dev_mode, **dataset_kwargs
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

    detector_name = model_config["detector"]

    if "dino" in detector_name:
        from detectors.models.backbones.backbone import build_backbone

        issa_backbone = build_backbone()
    breakpoint()

    if "yolo" in detector_name:  # TODO: should just put this in a create_model.py file
        anchors, num_anchors = initialize_anchors(base_config["priors"]["anchors"])

    # Initalize the detector backbone; typically some feature extractor
    backbone_name = model_config["backbone"]["name"]
    if backbone_name in model_config["backbone"]:
        backbone_params = model_config["backbone"][backbone_name]
    else:
        backbone_params = {}
    backbone = backbone_map[backbone_name](**backbone_params)

    if base_config["train"]["backbone_weights"] is not None:
        log.info("\nloading pretrained weights into the backbone\n")
        bb_weights = torch.load(
            base_config["train"]["backbone_weights"],
            weights_only=True,
            map_location=torch.device(device),
        )
        backbone.load_state_dict(
            bb_weights["state_dict"], strict=False
        )  # "state_dict" is the key to model state_dict for the pretrained weights I found

    # detector args
    model_components = {
        "backbone": backbone,
        "num_classes": dataset_train.num_classes,
        "anchors": anchors,
    }

    # Initialize detection model and transfer to GPU
    model = detectors_map[detector_name](**model_components)

    # Compute and log the number of params in the model
    reproduce.count_parameters(model)

    # model = Darknet("scripts/configs/yolov4.cfg")
    model.to(device)

    ## TODO: Apply weights init maybe

    # initalize loss with specific args
    # TODO:
    if detector_name == "yolov3":
        criterion = loss_map[detector_name](num_anchors=num_anchors, device=device)
    else:
        ValueError(f"loss function for {detector_name} not implemented")

    ## TODO: log the backbone, neck, head, and detector used.
    log.info("\nmodel architecture")
    log.info("\tbackbone: %s", backbone_name)
    log.info("\tdetector: %s", model_config["detector"])

    # Extract the train arguments from base config
    train_args = base_config["train"]

    # Extract solver configs and build the solvers
    solver_config = base_config["solver"]
    optimizer, lr_scheduler = build_solvers(
        model.parameters(), solver_config["optimizer"], solver_config["lr_scheduler"]
    )

    trainer = Trainer(
        output_dir=str(output_path),
        device=device,
        log_train_steps=base_config["log_train_steps"],
    )

    # Build trainer args used for the training
    trainer_args = {
        "model": model,
        "criterion": criterion,
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "class_names": dataset_train.class_names,
        "grad_accum_steps": grad_accum_steps,
        "start_epoch": train_args["start_epoch"],
        "epochs": train_args["epochs"],
        "ckpt_epochs": train_args["ckpt_epochs"],
        "checkpoint_path": train_args["checkpoint_path"],
    }
    trainer.train(**trainer_args)


if __name__ == "__main__":
    Fire(main)
