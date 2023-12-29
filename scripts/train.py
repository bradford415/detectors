from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from fire import Fire

from detectors.data.coco_minitrain import build_coco_mini
from detectors.models.yolov4 import YoloV4
from detectors.trainer import Trainer
from detectors.utils import utils

model_map: Dict[str, Any] = {"YoloV4": YoloV4}

dataset_map: Dict[str, Any] = {"CocoDetectionMiniTrain": build_coco_mini}


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
        split="train", **dataset_kwargs
    )
    dataset_val = dataset_map[base_config["dataset_name"]](
        split="val", **dataset_kwargs
    )
    # work on getting the dataset loaded, need to modifi self.prepare to just normalize to yolo
    dataset_train[5]
    exit()

    corpus_path = Path(base_config["root_dir"]) / base_config["input_data"]
    corpus = utils.load_text_file(corpus_path)

    runner = Trainer(corpus)

    model = base_config["model"]
    runner_args = {
        "text": corpus,
        "model": base_config["model"],
        "model_args": base_config[model],
        **base_config["train"],
    }
    runner.train(**runner_args)


if __name__ == "__main__":
    Fire(main)
