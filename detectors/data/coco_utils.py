import contextlib
import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import pycocotools.mask as mask_util
import torch
import torchvision
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class PreprocessCoco:
    """Preprocess the coco dataset before any augmentations are applied"""

    def __init__(self, return_masks=False):
        self.return_masks = return_masks

        # Used to make the dataset labels sequential
        self.coco_class_91_to_80 = coco91_to_coco80_class()

    def __call__(
        self, image: Image.Image, target: Dict[str, Any]
    ) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """Preprocesses the coco formatted dataset in the following way:

            1. Converts the coco annotation keys to tensors
            2. Removes objects that are labeled as "crowds"
            3. converts bboxes from [tl_x, tl_y, w, h] to [tl_x, tl_y, br_x, br_y]; this is done to perform transforms
               - NOTE: In transforms.Normalize the labels are converted to [cx, cy, w, h] and normalized by image size;
                       this happensafter PreprocessCoco() is called.
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

        # Convert w & h to br_x & br_y: [tl_x, tl_y, w, h] -> [tl_x, tl_y, br_x, br_y]
        boxes[:, 2:] += boxes[:, :2]

        # Clip the the x and y coordinates to the image size; guards against boxes being larger than image
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Create list of object labels, shift the coco ids so they are between 0-79 (i.e., sequential),
        # and convert to tensor
        ## TODO: This may not be the correct way to convert the classes
        classes = [self.coco_class_91_to_80[obj["category_id"]] for obj in annotations]
        classes = torch.tensor(classes, dtype=torch.int64)

        # Validate the br coordinate is greater than the tl; this should rarely be untrue
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        image_path = target["image_path"]

        # Update ground truth labels with the preprocessed boxes and classes
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["image_path"] = str(image_path)

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


def coco_stats(dataset: torchvision.datasets.CocoDetection, split: str):
    """Display dataset information based on the coco format

    Args:
        dataset: Dataset instance that is dervied from torchvision.datasets.CocoDetection
    """
    cat_ids = dataset.coco.getCatIds()
    img_ids = dataset.coco.getImgIds()

    log.info("\n%s set stats", split)

    log.info("\tunique categories: %d", len(cat_ids))
    log.info("\timages in the entire dataset: %d", len(img_ids))
    log.info("\tnumber of images in used for the current run: %d", len(dataset.ids))


def convert_to_coco_api(ds, bbox_fmt="voc"):
    """This function is required to create a proper coco api from your dataset.
    Not all object detection Repos use this e.g., detr and I'm not sure why

    Args:
        ds: The dataset object from torch.utils.data.Dataset; a cool trick is you can
            extract the dataset from the dataloader with dataloader.dataset
        bbox_fmt: The format of the bounding boxes i.e.,
                  what the elements of the bbox represent
                  1. voc: [topleft_x, topleft_y, botright_x, botright_y]
                  2. coco: [topleft_x, topleft_y, w, h]
                  3. yolo: [center_x, center_y, w, h]
    """
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"]

        # Convert to COCO format: tl_x, tl_y, w, h;
        # bbox_fmt is the format that the bboxes are CURRENTLY in and then they
        # will be converted to coco format; in this implementation, the
        # transform Normalize and ToTensorNoNormalize converts the bboxes to yolo, so for the evaluator
        # we need to convert it back to coco format
        if bbox_fmt.lower() == "voc":  # xmin, ymin, xmax, ymax
            bboxes[:, 2:] -= bboxes[:, :2]
        elif bbox_fmt.lower() == "yolo":  # xcen, ycen, w, h
            bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:] / 2
        elif bbox_fmt.lower() == "coco":
            pass
        else:
            raise ValueError(f"bounding box format {bbox_fmt} not supported!")
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]

    coco_ds.dataset = dataset
    # suppress pycocotools prints
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            coco_ds.createIndex()
    return coco_ds
