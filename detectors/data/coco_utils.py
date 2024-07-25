from typing import Any, Dict, Tuple

import numpy as np
import pycocotools.mask as mask_util
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torch.utils.data import Dataset


class PreprocessCoco(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks
        self.coco_class_91_to_80 = coco91_to_coco80_class()

    def __call__(
        self, image: Image.Image, target: Dict[str, Any]
    ) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """Preprocesses the coco formatted dataset in the following way:

            1. Converts the coco annotation keys to tensors
            2. Removes objects that are labeled as "crowds"
            3. converts bboxes from [tl_x, tl_y, w, h] to [tl_x, tl_y, br_x, br_y]

            Note: The bboxes are normalized and converted to Yolo format in data/transforms/Normalize()

        Args:
            image: singular PIL image
            target dictionary with keys image_id and annotations

        """
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        # Remove objects which are "crowds"; training with crowds could reduce accuracy
        # https://github.com/AlexeyAB/darknet/issues/5567#issuecomment-626758944
        annotations = target["annotations"]
        annotations = [
            obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0
        ]

        boxes = [obj["bbox"] for obj in annotations]

        # guard against no boxes via resizing (not really sure what this means)
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # Convert w & h to br_x & br_y: [tl_x, tl_y, w, h] -> [tl_x, tl_y, tl_x + w, tl_y + h]
        boxes[:, 2:] += boxes[:, :2]

        # Clip the the x and y coordinates to the image size; guards against boxes being larger than image
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Create list of object labels, shift the coco ids so they are between 0-79, and convert to tensor
        ## TODO: This may not be the correct way to convert the classes
        classes = [self.coco_class_91_to_80[obj["category_id"]] for obj in annotations]
        classes = torch.tensor(classes, dtype=torch.int64)
        


        # Validate the br coordinate is greater than the tl; this should rarely be untrue
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        # Update ground truth labels with the preprocessed boxes and classes
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # Extract area and iscrowd for conversion to coco api
        # Area is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
        area = torch.tensor([obj["area"] for obj in annotations])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # TODO comment this
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target
    
    
def coco91_to_coco80_class():
    """
    Converts 91-index COCO class IDs to 80-index COCO class IDs.
    
    The Coco 2017 object detection annotations use a total of 80 classes, however, the paper specifies 
    91 classes. When the official dataset was released, only 80 classes were used and many of the 
    classes omitted were inbetween the 80 class ids used. Therefore, the Coco dataset class ids are not
    contiguous so they need to be squished together so that they are continuous For example, the class
    id 90 is used in the dataset but it should actually be 79 since there are only 80 classes.
    I believe that some object detection models can handle these non-contiguous ideas 
    because of how they index, but I think most Yolo models require them continuous.

    Returns:
        (list): A list of 91 class IDs where the index represents the 80-index class ID and the value is the
            corresponding 91-index class ID.
    """
    return [
        None,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]



def get_coco_object(dataset: Dataset):
    """Return COCO object from pycocotools

    Args:
        dataset: torch dataset containing the COCO object
    """

    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    else:
        ValueError("Dataset type not recognized.")


def explore_coco(coco_annotation: COCO):
    print("\nDisplaying COCO information:")
    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")

    img_ids = coco_annotation.getImgIds()
    print(f"Number of Images: {len(img_ids)}")
