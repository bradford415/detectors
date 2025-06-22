from .dino import build_dino
from .yolov3 import Yolov3
from .yolov4 import Yolov4

detectors_map = {
    "yolov3": Yolov3,
    "yolov4": Yolov4,
    "dino": build_dino,
}


def create_detector(detector_name: str, detector_args: dict[str, any]):
    """Initialize the desired object detection model

    Args:
        detector_args: a dictionary of the parameters specific to the detector class
    """

    if detector_name == "yolov3":
        raise NotImplementedError
    elif detector_name == "yolov4":
        raise NotImplementedError
    elif detector_name == "dino":
        _create_dino(**detector_args)
    else:
        raise ValueError(f"detctor: {detector_name}")


def _create_dino(detector_args):
    """Create the dino detector, loss function, and postprocessor

    TODO: consider intializing the criterion/postprocessor separate from the model

    Args:
        detector_args: a dictionary of parameters specific to the build_dino() function;
                       see models.dino.build_dino() docstring for these parameters
    """
    model, criterion, postprocessor = detectors_map["dino"](
        num_classes=detector_args["num_classes"],
        backbone_args=detector_args["backbone_args"],
        dino_args=detector_args["dino_args"],
        criterion_args=detector_args["criterion_args"],
        matcher_args=detector_args["matcher_args"],
        loss_args=detector_args["loss_args"],
        postprocess_args=detector_args["postprocess_args"],
        device=detector_args["device"],
    )
    return model, criterion, postprocessor
