from typing import Optional

from detectors.models.backbones import backbone_map
from detectors.utils.script import initialize_anchors

from .dino import build_dino
from .yolov3 import Yolov3
from .yolov4 import Yolov4

detectors_map = {
    "yolov3": Yolov3,
    "yolov4": Yolov4,
    "dino": build_dino,
}


def create_detector(
    detector_name: str,
    detector_args: dict[str, any],
    num_classes: int,
    anchors: Optional[list] = None,
):
    """Initialize the desired object detection model

    Args:
        detector_name: name of the detector architecture to initialize
        detector_args: a dictionary of the parameters specific to the detector class
        anchors: a list of anchors if using a YOLO-based object detector;
                anchor format in the config file:
                ---
                priors:
                    anchors:
                        - [[10, 13], [16, 30],  [33, 23]]     # lowest resolution scale
                        - [[30, 61],  [62, 45],  [59, 119]]   # medium resolution scale
                        - [[116, 90], [156, 198], [373, 326]] # high resolution scale
    """

    if detector_name == "yolov3":
        model = _create_yolov3()
    elif detector_name == "yolov4":
        raise NotImplementedError
    elif detector_name == "dino":
        model = _create_dino(num_classes, detector_args)
    else:
        raise ValueError(f"detctor: {detector_name} not recognized")

    return model


def _create_dino(num_classes: int, detector_args: dict[str, any]):
    """Create the dino detector, loss function, and postprocessor

    TODO: consider intializing the criterion/postprocessor separate from the model

    Args:
        detector_args: a dictionary of parameters specific to the build_dino() function;
                       see models.dino.build_dino() docstring for these parameters
    """
    breakpoint()
    ############### START HERE --- unpack the args appropriately bl
    model, criterion, postprocessor = detectors_map["dino"](
        num_classes=num_classes,
        backbone_args=detector_args["backbone_args"],
        dino_args=detector_args["dino_args"],
        criterion_args=detector_args["criterion_args"],
        matcher_args=detector_args["matcher_args"],
        loss_args=detector_args["loss_args"],
        postprocess_args=detector_args["postprocess_args"],
        device=detector_args["device"],
    )
    return model, criterion, postprocessor


def _create_yolov3(detector_args: dict[str, any], anchors):
    """TODO"""

    anchors, num_anchors = initialize_anchors(anchors)

    # extract backbone parameters if they're supplied; some backbones do not take parameters
    backbone_name = detector_args["backbone_name"]
    backbone_params = detector_args.get("backbone_params", {})

    # Initalize the detector backbone; typically some feature extractor
    backbone = backbone_map[backbone_name](**backbone_params)

    # detector args
    model_components = {
        "backbone": backbone,
        "num_classes": dataset_train.num_classes,
        "anchors": anchors,
    }
    # Initialize detection model and transfer to GPU
    model = detectors_map[detector_name](**model_components)
