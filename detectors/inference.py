from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime
import PIL
import torch
from PIL import Image
from torchvision.transforms import Compose

from detectors.data.coco_utils import mscoco_category2name
from detectors.evaluate import load_model_checkpoint
from detectors.models.create import create_detector
from detectors.visualize import plot_all_detections


class BaseInference(ABC):

    def __init__(
        self,
        transforms: Compose,
        postprocessor,
        output_dir: Path,
        viz_n_images: int = 10,
    ):
        self.transforms = transforms
        self.postprocessor = postprocessor
        self.viz_n_images = viz_n_images
        self.output_dir = output_dir
        self._img_counter = 0

    @abstractmethod
    def inference_image(self, input_data):
        """Run inference on the input data."""
        pass

    def _visualize_detections(self, detections: dict):
        """Visualize the detections on the input image."""
        plot_all_detections(
            detections,
            conf_threshold=0.5,
            classes=mscoco_category2name,
            plot_n_images=self.viz_n_images,
            output_dir=self.output_dir,
            img_id=self._img_counter,
        )


class ONNXInference(BaseInference):

    def __init__(self, model_path: str, **base_kwargs):
        super().__init__(**base_kwargs)
        self.inference_session = onnxruntime.InferenceSession(
            model_path, providers=["CUDAExecutionProvider"]
        )

        # print graph details
        print("ONNX model graph details:")
        for output in self.inference_session.get_outputs():
            print("Name:", output.name)
            print("Shape:", output.shape)
            print("Type:", output.type)
            print("-----")

    def inference_image(self, img_path: str):
        """Inference TODO"""
        input_data = Image.open(img_path).convert("RGB")

        # save the original image size so we can scale the predicted bboxes back
        orig_w, orig_h = input_data.size
        orig_dims = torch.tensor([[orig_h, orig_w]])  # (b, 2)

        transformed_data, _ = self.transforms(
            image=input_data
        )  # _ ignores the labels which are not need for inference

        # add batch dimension
        transformed_data = transformed_data[None, ...]

        # prepare the input and output data to be loaded on the gpu (does not perform move or allocate memory yet)
        # https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html
        io_binding = self.inference_session.io_binding()
        io_binding.bind_cpu_input("images", np.array(transformed_data))
        io_binding.bind_output("pred_logits")
        io_binding.bind_output("pred_boxes")

        # copy inputs to gpu, run inference on gpu, and store model outputs on gpu
        self.inference_session.run_with_iobinding(io_binding)

        # move detections back to cpu; detections[0] = pred_logits (b, top_k, num_classes) and
        # detections[1] = pred_boxes (b, top_k, 4)
        detections = io_binding.copy_outputs_to_cpu()

        # pack back into a dictionary
        detections = {
            "pred_logits": torch.from_numpy(detections[0]),
            "pred_boxes": torch.from_numpy(detections[1]),
        }

        postprocessed_detections = self.postprocessor["bbox"](detections, orig_dims)

        postprocessed_detections[0]["image_path"] = img_path

        ### start here run the code see if it visualizes

        self._visualize_detections(postprocessed_detections)
        self._img_counter += 1

        return postprocessed_detections


# TODO
class PyTorchInference(BaseInference):

    def __init__(
        self,
        model_path: str,
        detector_name: str = None,
        detector_params: dict = None,
        num_classes: int = None,
        device: str | torch.device = None,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = create_detector(
            detector_name=detector_name,
            detector_args=detector_params,
            num_classes=num_classes,
        )
        self.model.to(self.device)
        _ = load_model_checkpoint(
            checkpoint_path=model_path,
            model=self.model,
            device=self.device,
        )

        # set model to eval so it does not require labels (for denoising)
        self.model.eval()

    def inference_image(self, img_path: str):
        """Inference TODO"""
        input_data = Image.open(img_path).convert("RGB")

        # save the original image size so we can scale the predicted bboxes back
        orig_w, orig_h = input_data.size
        orig_dims = torch.tensor([[orig_h, orig_w]])  # (b, 2)

        with torch.inference_mode():
            transformed_data, _ = self.transforms(
                image=input_data
            )  # _ ignores the labels which are not need for inference

            # add batch dimension and move to gpu
            transformed_data = transformed_data[None, ...].to(self.device)
            
            detections = self.model(transformed_data)

        # move detections back to cpu; detections[0] = pred_logits (b, top_k, num_classes) and
        # detections[1] = pred_boxes (b, top_k, 4)
        #detections = detections.to("cpu")

        detections = {
            "pred_logits": detections["pred_logits"].cpu(),
            "pred_boxes": detections["pred_boxes"].cpu(),
        }

        postprocessed_detections = self.postprocessor["bbox"](detections, orig_dims)

        postprocessed_detections[0]["image_path"] = img_path

        ### start here run the code see if it visualizes

        self._visualize_detections(postprocessed_detections)
        self._img_counter += 1

        return postprocessed_detections


def create_inferencer(
    backend: str,
    model_path: str,
    transforms: Compose,
    postprocessor,
    output_dir: Path,
    viz_n_images: int,
    num_classes: Optional[int] = None,
    detector_name: Optional[str] = None,
    detector_params: Optional[dict] = None,
) -> BaseInference:
    if backend == "onnx":
        return ONNXInference(
            model_path,
            transforms=transforms,
            postprocessor=postprocessor,
            output_dir=output_dir,
            viz_n_images=viz_n_images,
        )
    elif backend == "torch":
        return PyTorchInference(
            model_path=model_path,
            detector_name=detector_name,
            detector_params=detector_params,
            num_classes=num_classes,
            transforms=transforms,
            postprocessor=postprocessor,
            output_dir=output_dir,
            viz_n_images=viz_n_images,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
