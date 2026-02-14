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
        choices=["onnx"],
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
            contiguous_cat_ids=base_config["train_dataloader"]["dataset"][
                "contiguous_cat_ids"
            ],
        )
    }

    inferencer = create_inferencer(
        cli_args.backend,
        model_path=base_config["trained_model_path"],
        transforms=data_transforms,
        postprocessor=postprocessors,
    )

    img_paths = glob.glob(
        str(Path(base_config["images_dir"]) / "**" / "*.jpg"), recursive=True
    )

    for img_path in img_paths:
        print(f"Running inference on {img_path}...")
        input_data = Image.open(img_path).convert("RGB")
        detections = inferencer.inference_image(input_data)
        print(f"Detections: {detections}")
        break

    breakpoint()

    output_dir = Path("output") / "onnx" / detector_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = output_dir / f"{detector_name}.onnx"

    onnx_model.save(model_save_path)
    print(f"Saved onnx model to {model_save_path}")


if __name__ == "__main__":
    cli_args = cli_parser()
    main(cli_args)
