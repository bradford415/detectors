import argparse
import glob
from pathlib import Path

from PIL import Image

from detectors.data.create import make_config_transforms
from detectors.inference import create_inferencer
from detectors.postprocessing.postprocess import PostProcess
from detectors.utils import config


def cli_parser():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Export a trained model to ONNX format."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the config file used for training the model; this will be used to initialize the model for inference.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["onnx", "torch"],
        help="The inference backend to use.",
    )

    return parser.parse_args()


def main(cli_args: argparse.Namespace):

    base_config = config.load_config(cli_args.config)

    detector_name = base_config["detector_name"]

    # TODO implement for PyTorchInferences
    # detector_params = base_config["params"]
    # num_classes = base_config["train_dataloader"]["dataset"]["num_classes"]

    if cli_args.backend == "onnx":
        print(f"Using ONNX Runtime as the backend for inference.")

    transforms_config = base_config["transforms"]
    data_transforms = make_config_transforms(transforms_config)

    # Initalize postprocessor
    # converts the models output to the expected output by the coco api, during inference
    # and visualization only; not used during training
    postprocess_args = base_config["postprocessor"]
    postprocessors = {
        "bbox": PostProcess(
            num_select=postprocess_args["num_top_queries"],
            contiguous_cat_ids=True,
            # contiguous_cat_ids=base_config["train_dataloader"]["dataset"][
            #     "contiguous_cat_ids"
            # ],
        )
    }

    output_dir = (
        Path(base_config["output_dir"]) / "inference" / cli_args.backend / detector_name
    )
    inferencer = create_inferencer(
        cli_args.backend,
        model_path=base_config["trained_model_path"],
        transforms=data_transforms,
        postprocessor=postprocessors,
        output_dir=output_dir,
        viz_n_images=base_config["viz_n_images"],
        detector_name=detector_name,
        detector_params=base_config["params"],
        num_classes=base_config["train_dataloader"]["dataset"]["num_classes"],
    )

    img_paths = glob.glob(
        str(Path(base_config["images_dir"]) / "**" / "*.jpg"), recursive=True
    )

    viz_n_images = base_config.get("viz_n_images", 10)
    for idx, img_path in enumerate(img_paths, 1):
        print(f"Running inference on {img_path}...")
        inferencer.inference_image(img_path)

        if idx == viz_n_images:
            break
        
    print(f"Saved {viz_n_images} visualizations of the detections to {output_dir}")


if __name__ == "__main__":
    cli_args = cli_parser()
    main(cli_args)
