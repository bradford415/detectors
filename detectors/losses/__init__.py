from .yolov3_loss import Yolov3Loss
from .yolov4_loss import Yolov4Loss

loss_map = {"yolov3": Yolov3Loss, "yolov4": Yolov4Loss}
