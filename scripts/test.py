import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader

from detectors.data.coco_minitrain import build_coco_mini
from detectors.data.collate_functions import collate_fn_test
from detectors.evaluate import evaluate, load_model_state_dict
from detectors.models.backbones import backbone_map
from detectors.models.darknet import Darknet
from detectors.models.yolov4 import YoloV4
from detectors.utils import reproduce

# TODO: should move this to its own file
detectors_map: Dict[str, Any] = {"yolov4": YoloV4}

dataset_map: Dict[str, Any] = {"CocoDetectionMiniTrain": build_coco_mini}

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
    log_path = output_path / "testing.log"

    # Configure logger that prints to a log file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    log.info("Initializing...\n")

    # Apply reproducibility seeds
    reproduce.reproducibility(**base_config["reproducibility"])

    # Set cuda parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    test_kwargs = {
        "batch_size": 1,
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

        test_kwargs.update(cuda_kwargs)
    else:
        log.info("Using CPU")

    dataset_kwargs = {"root": base_config["dataset"]["root"]}
    dataset_test = dataset_map[base_config["dataset_name"]](
        dataset_split="val", debug_mode=base_config["debug_mode"], **dataset_kwargs
    )

    dataloader_test = DataLoader(
        dataset_test,
        collate_fn=collate_fn_test,
        drop_last=True,
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

    # Initialize detection model and load its state_dict
    model = detectors_map[model_config["detector"]](**model_components)
    model = load_model_state_dict(model, base_config["state_dict_path"])
    model.to(device)

    reproduce.save_configs(
        config_dicts=[base_config, model_config],
        save_names=["base_config.json", "model_config.json"],
        output_path=output_path / "reproduce",
    )
    # Build trainer args used for the training
    evaluation_args = {
        "output_path": output_path,
        "model": model,
        "dataloader_test": dataloader_test,
        "class_names": dataset_test.class_names,
    }
    evaluate(**evaluation_args)


if __name__ == "__main__":
    Fire(main)
