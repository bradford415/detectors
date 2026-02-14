from abc import ABC, abstractmethod

import numpy as np
import onnxruntime
import PIL
from torchvision.transforms import Compose


class BaseInference(ABC):

    def __init__(self, transforms: Compose, postprocessor):
        self.transforms = transforms
        self.postprocessor = postprocessor

    @abstractmethod
    def inference_image(self, input_data):
        """Run inference on the input data."""
        pass


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

    def inference_image(self, input_data: PIL.Image):
        """Inference TODO"""
        breakpoint()
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
        detections = {"pred_logits": detections[0], "pred_boxes": detections[1]}

        self.postprocessor(
            detections,
        )

        breakpoint()

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
    backend: str, model_path: str, transforms: Compose, postprocessor
) -> BaseInference:
    if backend == "onnx":
        return ONNXInference(
            model_path, transforms=transforms, postprocessor=postprocessor
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
