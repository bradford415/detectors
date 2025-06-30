from typing import Optional

from detectors.models.backbones import backbone_map
from detectors.postprocessing.postprocess import PostProcess
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
        model = _create_yolov3(detector_name, num_classes, detector_args)
    elif detector_name == "yolov4":
        raise NotImplementedError
    elif detector_name == "dino":
        model = _create_dino(detector_name, num_classes, detector_args)
    else:
        raise ValueError(f"detctor: {detector_name} not recognized")

    return model


def _create_dino(detector_name: str, num_classes: int, detector_args: dict[str, any]):
    """Create the dino detector, loss function, and postprocessor

    TODO: consider intializing the criterion/postprocessor separate from the model

    Args:
        detector_name: now of the object detection model
        num_classes: the max_obj_id + 1 (background); for coco this should be 91
        detector_args: a dictionary of parameters specific to the build_dino() function;
                       see models.dino.build_dino() docstring for these parameters
    """
    model = detectors_map[detector_name](
        num_classes=num_classes,
        backbone_args=detector_args["backbone_params"],
        dino_args=detector_args["detector_params"],
        aux_loss=detector_args["aux_loss"],
    )
    return model


def _create_yolov3(
    detector_name: str, num_classes: str, detector_args: dict[str, any], anchors
):
    """TODO

    Args:
        detector_name: the name of the object detector to use
        num_classes: the number of classes in the dataset; for coco this should be 80 and
                     these mapping should be contiguous
    """

    anchors, num_anchors = initialize_anchors(anchors)

    # extract backbone parameters if they're supplied; some backbones do not take parameters
    backbone_name = detector_args["backbone_name"]
    backbone_params = detector_args.get("backbone_params", {})

    # Initalize the detector backbone; typically some feature extractor
    backbone = backbone_map[backbone_name](**backbone_params)

    # detector args
    model_components = {
        "backbone": backbone,
        "num_classes": num_classes,
        "anchors": anchors,
    }
    # Initialize detection model and transfer to GPU
    model = detectors_map[detector_name](**model_components)
