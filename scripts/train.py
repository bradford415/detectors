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

from detectors.data import create_dataloader, create_dataset

# from detectors.data.datasets.coco import build_coco
from detectors.models.create import create_detector
from detectors.postprocessing.postprocess import PostProcess
from detectors.solvers.build import build_solvers, create_loss
from detectors.solvers.ema_model import create_ema_model
from detectors.trainer import create_trainer
from detectors.utils import config, distributed, reproduce

# Initialize the root logger
log = logging.getLogger(__name__)


def main(
    base_config_path: str = "scripts/configs/train-coco-default.yaml",
    model_config_path: Optional[str] = None,
    dataset_root: Optional[str] = None,
    backbone_weights: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired training configuration file; by default the train-coco-default.yaml file is used which
                          trains from scratch (i.e., no pretrained backbone weights or detector weights)
        model_config_path: path to the detection model configuration file; by default the yolov3 base model with a DarkNet53 backbone is used;
                           right now thi is a legacy mode, going forward all additional configs should be specified in the base_config_path only
        dataset_root: path to the the root directory of the dataset; for coco, this is the path to the dir containing the `images` and `annotations` dirs
        backbone_weights: path to the backbone weights; this can be useful when training from scratch
        checkpoint_path: path to the weights of the entire detector model; this can be used to resume training or inference;
                         if this parameter is not None, the backbone_weights will be ignored
    """

    # Wrap the cli args in a dict to merge into the base config; CLI args override config file values if they exist
    cli_args = {
        "base_config_path": base_config_path,
        "model_config_path": model_config_path,
        "dataset_root": dataset_root,
        "backbone_weights": backbone_weights,
        "checkpoint_path": checkpoint_path,
    }

    # load and merge the base config and other configs included in the base config
    if model_config_path is None:
        base_config = config.load_config(base_config_path)
        model_config = base_config
    else:
        model_config = config.load_config(model_config_path)
        base_config = config.merge_dict(
            config.load_config(base_config_path),
            model_config,
        )

    # add CLI args to the base config and override any existing values
    base_config = config.merge_dict(base_config, cli_args)

    # initalize torch distributed mode by setting the communication between all procceses
    # and assigning the GPU to use for each proccess;
    # NOTE: in general, each proccess should run most commands in the program except saving to disk
    world_size, global_rank, local_rank, distributed_mode = (
        distributed.init_distributed_mode(backend=base_config["cuda"]["backend"])
    )

    # stagger each process by 20 ms; not required but recommended to help prevent I/O
    # contention on shared file systems like EFS
    time.sleep(global_rank * 0.02)

    # TODO: Verify if this new method works instead of hardcoding the overrides
    # Override configuration parameters if CLI arguments are provided; this allows external users
    # to easily run the project without messing with the configuration files
    # if dataset_root is not None:
    #     base_config["dataset"]["root"] = dataset_root

    # # overwrite config values with CLI values if specified
    # if checkpoint_path is not None:
    #     base_config["train"]["checkpoint_path"] = checkpoint_path
    #     base_config["train"]["backbone_weights"] = None
    # elif backbone_weights is not None:
    #     base_config["train"]["backbone_weights"] = backbone_weights

    # if (
    #     base_config["train"]["checkpoint_path"] is not None
    #     and base_config["train"]["backbone_weights"] is not None
    # ):
    #     raise ValueError(
    #         "checkpoint_path and backbone_weights cannot both have a value. Set one of the values to 'null'."
    #     )

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
            / "train"
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

    train_dl_params = base_config["train_dataloader"]
    val_dl_params = base_config["val_dataloader"]

    # batch size per gpu
    batch_size = train_dl_params["batch_size"]
    effective_bs = train_dl_params["effective_batch_size"]
    train_dl_params.pop("effective_batch_size")  # pop unexcepted key to the dataloader

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
        val_batch_size = 1
        grad_accum_steps = 2
        base_config["train_dataloader"]["num_workers"] = 0
        base_config["val_dataloader"]["num_workers"] = 0

    log.info(
        "\neffective_batch_size: %-5d grad_accum_steps: %-5d batch_size_per_gpu: %d",
        effective_bs,
        grad_accum_steps,
        batch_size,
    )

    # Set device specific characteristics
    use_cpu = False
    if not distributed_mode:
        gpu_id = base_config["cuda"]["gpus"][0]
        if torch.cuda.is_available():
            # setup cuda
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
            log.info("Using %d GPU(s): ", len(base_config["cuda"]["gpus"]))
            for gpu in base_config["cuda"]["gpus"]:
                log.info("    -%s", torch.cuda.get_device_name(gpu))
        elif torch.mps.is_available():
            # setup mps (apple silicon)
            base_config["train_dataloader"]["dataset"]["root"] = base_config[
                "train_dataloader"
            ]["dataset"]["root_mac"]
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

    train_ds_params = base_config["train_dataloader"]["dataset"]
    val_ds_params = base_config["val_dataloader"]["dataset"]

    dataset_kwargs = {
        "dataset_name": train_ds_params["dataset_name"],
        "root": train_ds_params["root"],
        "num_classes": train_ds_params["num_classes"],
        "contiguous_cat_ids": train_ds_params["contiguous_cat_ids"],
    }
    dataset_train = create_dataset(
        split="train",
        transforms_config=train_ds_params["transforms"],
        dev_mode=dev_mode,
        **dataset_kwargs,
    )
    dataset_val = create_dataset(
        split="val",
        transforms_config=val_ds_params["transforms"],
        dev_mode=dev_mode,
        **dataset_kwargs,
    )

    # remove unnecessary keys
    train_dl_params.pop("dataset")
    val_dl_params.pop("dataset")

    dataloader_train, sampler_train = create_dataloader(
        is_distributed=distributed_mode,
        dataset=dataset_train,
        collate_name=base_config["collate_fn"]["name"],
        collate_params=base_config["collate_fn"]["params"],
        **train_dl_params,
    )
    dataloader_val, _ = create_dataloader(
        is_distributed=distributed_mode,
        dataset=dataset_val,
        collate_name=base_config["collate_fn"]["name"],
        collate_params=base_config["collate_fn"]["params"],
        **val_dl_params,
    )

    detector_name = base_config["detector_name"]
    detector_params = model_config["params"]

    ##### start hereeee build the model
    #### basically, the GLOBAL_CONFIG is built initially by setting the default parameters
    # from every module decorated with @register, then the config files are loaded and overrided
    # TODO; should put this explanation somewhere maybe?
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

    # Initalize postprocessor if using a detr model
    if "postprocessor" in base_config:
        # converts the models output to the expected output by the coco api, during inference
        # and visualization only; not used during training
        postprocess_args = base_config["postprocessor"]
        postprocessors = {
            "bbox": PostProcess(
                num_select=postprocess_args["num_top_queries"],
                contiguous_cat_ids=train_ds_params["contiguous_cat_ids"],
            )
        }

    if base_config["train"]["backbone_weights"] is not None:
        log.info("\nloading pretrained weights into the backbone\n")
        bb_weights = torch.load(
            base_config["train"]["backbone_weights"],
            weights_only=True,
            map_location=torch.device(device),
        )

        backbone = getattr(model_without_ddp, "backbone", None)
        if backbone is None:
            raise AttributeError(
                "The model does not expose a backbone for loading weights."
            )
        state_dict = (
            bb_weights["state_dict"] if "state_dict" in bb_weights else bb_weights
        )
        backbone.load_state_dict(
            state_dict, strict=False
        )  # "state_dict" is the key to model state_dict for the pretrained weights I found

    # Compute and log the number of params in the model
    reproduce.count_parameters(model)

    # log.info("\nmodel architecture")
    # log.info("\tbackbone: %s", detector_params["backbone"]["name"])
    # log.info("\tdetector: %s", model_config["detector"])

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
    )

    criterion = create_loss(
        model_name=base_config["detector_name"],
        num_classes=dataset_train.num_classes,
        base_config=base_config,
        device=device,
    )

    ema_model = None
    if base_config["solver"]["use_ema"]:
        ema_model = create_ema_model(
            model, ema_params=base_config["solver"]["ema_params"]
        )

    trainer = create_trainer(
        model_name=detector_name,
        model=model,
        ema_model=ema_model,
        criterion=criterion,
        output_dir=str(output_path),
        use_amp=train_args["use_amp"],
        is_distributed=distributed_mode,
        step_lr_on=solver_config["lr_scheduler"]["step_lr_on"],
        device=device,
        log_train_steps=base_config["log_train_steps"],
    )

    if hasattr(dataset_val, "coco"):
        coco_api = dataset_val.coco
    else:
        coco_api = None

    # Build trainer args used for the training
    trainer_args = {
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "sampler_train": sampler_train,
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

    distributed.cleanup()


if __name__ == "__main__":
    Fire(main)
