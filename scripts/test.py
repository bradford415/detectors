import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader

from detectors.data.coco import build_coco
from detectors.data.collate_functions import get_collate_fn
from detectors.evaluate import evaluate, load_model_checkpoint, test_detr
from detectors.models.backbones import backbone_map
from detectors.models.create import create_detector
from detectors.postprocessing.postprocess import PostProcess
from detectors.utils import reproduce
from detectors.visualize import plot_all_detections, plot_test_detections

dataset_map: Dict[str, Any] = {"CocoDetection": build_coco}

# Initialize the root logger
log = logging.getLogger(__name__)


def main(base_config_path: str, model_config_path: str):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired base configuration file
        model_config_path: path to the detection model configuration file

    """
    # Load configuration files
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    test_config = base_config["test"]
    batch_size = test_config["batch_size"]

    checkpoint_path = test_config.get("checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError("a model's checkpoint file must be specified for testing")

    dev_mode = base_config["dev_mode"]

    # Initialize output path from the checkpoint path
    output_path = Path(checkpoint_path).parent.parent.parent / "test"
    log.info("\noutputs  %s", checkpoint_path)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "testing.log"

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

    if dev_mode:
        log.info("NOTE: executing in dev mode")
        base_config["test"]["batch_size"] = 1

    log.info("initializing...")
    log.info("outputs beings saved to %s\n", str(output_path))

    # apply reproducibility seeds
    reproduce.set_seeds(**base_config["reproducibility"])

    # Set cuda parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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

    pin_memory = False
    if not use_cpu:
        pin_memory = True

    dataset_kwargs = {
        "root": base_config["dataset"]["root"],
        "num_classes": base_config["dataset"]["num_classes"],
    }
    dataset_test = dataset_map[base_config["dataset_name"]](
        dataset_split="val", dev_mode=base_config["dev_mode"], **dataset_kwargs
    )

    # NOTE: not using a batch sampler because the padding applied with batching
    #       slightly hurts the mAP so a batch size of 1 is used currently
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    num_workers = base_config["dataset"]["num_workers"] if not dev_mode else 0
    collate_fn = get_collate_fn(model_config["detector"])

    dataloader_test = DataLoader(
        dataset_test,
        sampler=sampler_test,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    detector_name = model_config["detector"]
    detector_params = model_config["params"]

    # initialize the detector for inference and load the desired weights
    model = create_detector(
        detector_name=detector_name,
        detector_args=detector_params,
        num_classes=dataset_test.num_classes,
    )
    model.to(device)
    _ = load_model_checkpoint(
        checkpoint_path=checkpoint_path, model=model, device=device
    )

    reproduce.count_parameters(model)

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

    # # number of anchors per scale
    # num_anchors = len(anchors[0])

    # # strip away the outer list to make it a 2d python list
    # anchors = [anchor for anchor_scale in anchors for anchor in anchor_scale]

    # Build trainer args used for the training
    test_args = {
        "device": device,
    }

    if detector_name in ["dino"]:
        coco_api = dataset_test.coco
    else:
        coco_api = None

    if model_config["detector"] == "dino":
        stats, detections = test_detr(
            model, dataloader_test, coco_api, postprocessors, **test_args
        )
    else:
        metrics_output, detections, val_loss = evaluate(
            model,
            dataloader_test,
            class_names,
            criterion=criterion,
            output_path=self.output_dir,
            device=self.device,
        )

    viz_params = base_config["visualize"]
    if viz_params["plot_detections"]:
        output_dir = output_path / "visuals"
        log.info(
            "\nplotting detections on %d images at the path: %s",
            viz_params["plot_n_images"],
            str(output_dir),
        )
        plot_test_detections(
            detections,
            viz_params["conf_threshold"],
            classes=dataset_test.class_names,
            plot_n_images=viz_params["plot_n_images"],
            output_dir=output_dir,
        )

    log.info("\nfinished")


if __name__ == "__main__":
    Fire(main)
