import datetime
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
import torch.distributed as dist
import yaml
from fire import Fire
from torch.utils.data import DataLoader, DistributedSampler

from detectors.data.coco import build_coco
from detectors.data.collate_functions import get_collate_fn
from detectors.losses import Yolov3Loss, Yolov4Loss, create_dino_loss
from detectors.models.create import create_detector
from detectors.postprocessing.postprocess import PostProcess
from detectors.solvers.build import build_solvers
from detectors.trainer import Trainer
from detectors.utils import distributed, reproduce

dataset_map: Dict[str, Any] = {"CocoDetection": build_coco}

loss_map = {"yolov3": Yolov3Loss, "yolov4": Yolov4Loss, "dino": create_dino_loss}

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
    world_size, global_rank, local_rank, distributed_mode = (
        distributed.init_distributed_mode(backend=base_config["cuda"]["backend"])
    )

    # stagger each process by 20 ms; not required but recommended to help prevent I/O
    # contention on shared file systems like EFS
    time.sleep(global_rank * 0.02)

    # Override configuration parameters if CLI arguments are provided; this allows external users
    # to easily run the project without messing with the configuration files
    if dataset_root is not None:
        base_config["dataset"]["root"] = dataset_root

    # overwrite config values with CLI values if specified
    if checkpoint_path is not None:
        base_config["train"]["checkpoint_path"] = checkpoint_path
        base_config["train"]["backbone_weights"] = None
    elif backbone_weights is not None:
        base_config["train"]["backbone_weights"] = backbone_weights

    if (
        base_config["train"]["checkpoint_path"] is not None
        and base_config["train"]["backbone_weights"] is not None
    ):
        raise ValueError(
            "checkpoint_path and backbone_weights cannot both have a value. Set one of the values to 'null'."
        )

    dev_mode = base_config["dev_mode"]

    # Initialize paths
    checkpoint_path = base_config["train"].get("checkpoint_path", None)
    if checkpoint_path:
        # if resuming checkpoint use the same directory
        output_path = Path(checkpoint_path).parent.parent
        log.info(
            "\nresuming training from the specificed checkpoint %s", checkpoint_path
        )
    else:
        output_path = (
            Path(base_config["output_dir"])
            / base_config["exp_name"]
            / f"{datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
        )
    log_path = output_path / "training.log"

    # create output/ckpt directories and save configuration files on main process
    if global_rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)

        # Save configuration files and parameters
        reproduce.save_configs(
            config_dicts=[
                (base_config, "base_config.yaml"),
                (model_config, "model_config.yaml"),
            ],
            output_path=output_path / "reproduce",
        )

    # Configure logger that prints to a log file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    # suppress logs from non-zero ranks
    if dist.is_initialized() and dist.get_rank() != 0:
        log.setLevel(logging.WARNING)  # or ERROR to suppress even more

    log.info("initializing...\n")
    log.info("writing outputs to %s", str(output_path))

    if (
        base_config["train"]["checkpoint_path"] is None
        and base_config["train"]["backbone_weights"] is None
    ):
        log.info("\ntraining from scratch; no pretrained weights provided")

    # Apply reproducibility seeds
    reproduce.set_seeds(**base_config["reproducibility"])

    # Extract training and val params
    train_args = base_config["train"]

    # batch size per gpu
    batch_size = train_args["batch_size"]
    effective_bs = train_args["effective_batch_size"]

    val_batch_size = train_args["batch_size"]

    # calculate the number of gradient accumulation steps to simulate a larger batch;
    # if using DDP, grad_accum_steps = effective_batch_size // batch_size * num_gpus
    if effective_bs % (batch_size * world_size) == 0 and effective_bs >= (
        batch_size * world_size
    ):
        grad_accum_steps = effective_bs // (batch_size * world_size)
    else:
        raise ValueError(
            "grad_accum_bs must be divisible by batch_size and greater than or equal to batch_size"
        )

    if dev_mode:
        log.info("\nNOTE: executing in dev mode")
        batch_size = 1
        val_batch_size = 2
        grad_accum_steps = 2

    log.info(
        "\neffective_batch_size: %-5d grad_accum_steps: %-5d batch_size_per_gpu: %d",
        effective_bs,
        grad_accum_steps,
        batch_size,
    )

    # Set device specific characteristics
    use_cpu = False
    if torch.cuda.is_available():
        # setup cuda
        device = torch.device("cuda")
        log.info("Using %d GPU(s): ", len(base_config["cuda"]["gpus"]))
        for gpu in range(len(base_config["cuda"]["gpus"])):
            log.info("    -%s", torch.cuda.get_device_name(gpu))
    elif torch.mps.is_available():
        # setup mps (apple silicon)
        base_config["dataset"]["root"] = base_config["dataset"]["root_mac"]
        device = torch.device("mps")
        log.info("Using: %s", device)
    else:
        # setup cpu
        use_cpu = True
        device = torch.device("cpu")
        log.info("Using CPU")

    pin_memory = False
    if not use_cpu:
        pin_memory = True

    dataset_kwargs = {
        "root": base_config["dataset"]["root"],
        "num_classes": base_config["dataset"]["num_classes"],
    }
    dataset_train = dataset_map[base_config["dataset_name"]](
        dataset_split="train", dev_mode=dev_mode, **dataset_kwargs
    )
    dataset_val = dataset_map[base_config["dataset_name"]](
        dataset_split="val", dev_mode=dev_mode, **dataset_kwargs
    )

    if distributed_mode:
        # ensures that each process gets a different subset of the dataset in distributed mode
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        # using RandomSampler and SequentialSampler w/ default parameters is the same as using
        # shuffle=True and shuffle=False in the DataLoader, respectively; if you pass a Sampler into
        # the DataLoader, you cannot set the shuffle parameter (mutually exclusive)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # create a batch sampler for train but not val which means val will only use a batch_size=1
    # similar as above, when you set shuffle in the DataLoader it automatically wraps
    # RandomSampler and SequentialSampler in a BatchSampler
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True
    )

    num_workers = base_config["dataset"]["num_workers"] if not dev_mode else 0
    collate_fn = get_collate_fn(model_config["detector"])

    # drop_last is true becuase the loss function intializes masks with the first dimension being the batch_size;
    # during the last batch, the batch_size will be different if the length of the dataset is not divisible by the batch_size
    dataloader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=val_batch_size,
        sampler=sampler_val,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    detector_name = model_config["detector"]
    detector_params = model_config["params"]

    model = create_detector(
        detector_name=detector_name,
        detector_args=detector_params,
        num_classes=dataset_train.num_classes,
    )
    model.to(device)

    # Wrap the base model in ddp and store a pointer to the model without ddp; when saving the model
    # we want to save the model without ddp for portablility; ddpm wraps the model with additional
    # logic and parameters that are not serializable
    model_without_ddp = model
    if distributed_mode:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False
        )
        model_without_ddp = (
            model.module
        )  # this line is technically not needed but helps for clarity

    # Initalize postprocessor if using DINO DETR
    if "postprocess" in detector_params:
        # converts the models output to the expected output by the coco api, during inference
        # and visualization only; not used during training
        postprocess_args = detector_params["postprocess"]
        postprocessors = {
            "bbox": PostProcess(
                num_select=postprocess_args["num_select"],
                nms_iou_threshold=postprocess_args["nms_iou_threshold"],
            )
        }

    if base_config["train"]["backbone_weights"] is not None:
        log.info("\nloading pretrained weights into the backbone\n")
        bb_weights = torch.load(
            base_config["train"]["backbone_weights"],
            weights_only=True,
            map_location=torch.device(device),
        )

        # TODO fix this
        backbone.load_state_dict(
            bb_weights["state_dict"], strict=False
        )  # "state_dict" is the key to model state_dict for the pretrained weights I found

    # Compute and log the number of params in the model
    reproduce.count_parameters(model)

    ## TODO: Apply weights init maybe

    # initalize loss with specific args
    if detector_name == "yolov3":
        # TODO: fix yolov3
        criterion = loss_map[detector_name](num_anchors=num_anchors, device=device)
    elif detector_name == "dino":
        num_decoder_layers = detector_params["detector"]["transformer"][
            "num_decoder_layers"
        ]
        criterion = loss_map[detector_name](
            num_classes=base_config["dataset"]["num_classes"],
            num_decoder_layers=num_decoder_layers,
            aux_loss=detector_params["aux_loss"],
            two_stage_type=detector_params["detector"]["two_stage"]["type"],
            loss_args=model_config["params"]["loss_weights"],
            matcher_args=model_config["params"]["matcher"],
            device=device,
        )
    else:
        ValueError(f"loss function for {detector_name} not implemented")

    log.info("\nmodel architecture")
    log.info("\tbackbone: %s", detector_params["backbone"]["name"])
    log.info("\tdetector: %s", model_config["detector"])

    # Extract the train arguments from base config
    train_args = base_config["train"]

    # Extract solver configs and build the solvers; parameter strategy craetes the parameter dicts for the
    # optimizer (default: "all" use all parameters in the model in one group)
    solver_config = base_config["solver"]
    optimizer, lr_scheduler = build_solvers(
        model_without_ddp,
        solver_config["optimizer"],
        solver_config["lr_scheduler"],
        parameter_strategy=solver_config.get("parameter_strategy", "all"),
        backbone_lr=solver_config["optimizer"].get("backbone_lr", None),
    )

    trainer = Trainer(
        output_dir=str(output_path),
        model_name=detector_name,
        use_amp=train_args["use_amp"],
        step_lr_on=solver_config["lr_scheduler"]["step_lr_on"],
        device=device,
        log_train_steps=base_config["log_train_steps"],
    )

    if detector_name in ["dino"]:
        coco_api = dataset_val.coco
    else:
        coco_api = None

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
        "coco_api": coco_api,
        "postprocessors": postprocessors,
        "max_norm": train_args["max_norm"],
        "start_epoch": train_args.get("start_epoch", 1),
        "epochs": train_args["epochs"],
        "ckpt_epochs": train_args["ckpt_epochs"],
        "checkpoint_path": train_args["checkpoint_path"],
    }
    trainer.train(**trainer_args)


if __name__ == "__main__":
    Fire(main)
