from typing import Any, Dict, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor


class PreprocessCoco(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(
        self, image: Image.Image, target: Dict[str, Any]
    ) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """Preprocesses the coco formatted dataset in the following way:

            1. Converts the coco annotation keys to tensors
            2. Removes objects that are labeled as "crowds"
            3. converts bbox from [tl_x, tl_y, w, h] to [tl_x, tl_y, br_x, br_y]

        Args:
            image: pil image
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

        # Create list of object labels and convert to tensor
        classes = [obj["category_id"] for obj in annotations]
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

        # Extract area and iscrowd for conversion to coco api; required keys for the cocapi but no used in object detection
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


def explore_coco(coco_annotation: COCO):
    print("\nDisplaying COCO information:")
    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")

    img_ids = coco_annotation.getImgIds()
    print(f"Number of Images: {len(img_ids)}")
