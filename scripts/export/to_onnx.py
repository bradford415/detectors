import argparse
from pathlib import Path

import onnx
import torch
import yaml
from torch import device

from detectors.models.create import create_detector
from detectors.utils import config


def cli_parser():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Export a trained model to ONNX format."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the config file used for training the model.",
    )

    return parser.parse_args()


def main(cli_args: argparse.Namespace):

    base_config = config.load_config(cli_args.config)

    detector_name = base_config["detector_name"]
    detector_params = base_config["params"]
    num_classes = base_config["train_dataloader"]["dataset"]["num_classes"]

    model = create_detector(
        detector_name=detector_name,
        detector_args=detector_params,
        num_classes=num_classes,
    )
    print(f"Initialized model: {detector_name}")

    # set model to eval so it does not require labels (for denoising)
    model.eval()

    # create dummy input for ONNX export
    dummy_inputs = (torch.randn(1, 3, 640, 640),)
    print(f"Created dummy input for ONNX export of shape: {dummy_inputs[0].shape}")

    # Notes:
    #     - `input_names` and `output_names` are the `name` param when you call
    #       `bind_cpu_input` and `bind_output` during onnxruntime inference
    #     - onnx runtime cannot handle dictionaries as outputs, which is what detr models return, so
    #       they will be flattened to a list of outputs, so when we call
    #       `outputs = copy_outputs_to_cpu[0]` outputs[0] = pred_logits and outputs[1] = pred_boxes
    onnx_model = torch.onnx.export(
        model,
        dummy_inputs,
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        dynamo=True,
    )
    print("Exported model to ONNX format")

    output_dir = Path("output") / "onnx" / detector_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = output_dir / f"{detector_name}.onnx"

    onnx_model.save(model_save_path)
    print(f"Saved onnx model to {model_save_path}")

    print("Checking ONNX model...")
    model = onnx.load(model_save_path)
    onnx.checker.check_model(model)
    print("ONNX model is valid!")


if __name__ == "__main__":
    cli_args = cli_parser()
    main(cli_args)
