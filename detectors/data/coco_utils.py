from typing import Dict, Any

import torch
from PIL import Image

class ConvertCocoToYolo(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image, target: Dict[str, Any]):
        """Converts the Coco bbox to Yolo bbox

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
        annotations = [obj for obj in annotations if 'iscrowd' not in obj or obj['iscrowd'] == 0]

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

        ############## START HERE: finish commenting and understanding this class################
        exit()
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target
