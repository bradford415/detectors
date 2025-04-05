from .yolov3 import Yolov3
from .yolov4 import Yolov4

detectors_map = {
    "yolov3": Yolov3,
    "yolov4": Yolov4,
}
