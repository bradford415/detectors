from abc import ABC, abstractmethod

import numpy as np
import onnxruntime
import PIL
from torchvision.transforms import Compose


class BaseInference(ABC):

    def __init__(self, transforms: Compose):
        self.transforms = transforms

    @abstractmethod
    def inference_image(self, input_data):
        """Run inference on the input data."""
        pass


class ONNXInference(BaseInference):

    def __init__(self, model_path: str, **base_kwargs):
        super().__init__(**base_kwargs)
        self.model = onnxruntime.InferenceSession(model_path)

    def inference_image(self, input_data: PIL.Image):
        """Inference TODO"""
        breakpoint()
        transformed_data, _ = self.transforms(image=input_data) # _ ignores the labels which are not need for inference
        #### start here !!!
        detections = self.model(transformed_data)
        return detections


# TODO
class PyTorchInference(BaseInference):

    def __init__(self, model_path: str):
        # TODO
        # self.model =    model = create_detector(
        #     detector_name=detector_name,
        #     detector_args=detector_params,
        #     num_classes=num_classes,
        # )

        # set model to eval so it does not require labels (for denoising)
        self.model.eval()


def create_inferencer(
    backend: str, model_path: str, transforms: Compose
) -> BaseInference:
    if backend == "onnx":
        return ONNXInference(model_path, transforms=transforms)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
