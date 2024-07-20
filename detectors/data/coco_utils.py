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


class CocoEvaluator(object):
    """Evaluates Coco detections"""

    def __init__(self, coco_gt, iou_types: list | tuple, bbox_fmt: str = "coco"):
        """Initialize the CocoEvaluator

        Args:
            coco_gt:
            iou_types: list or tuple of the IoU type; can be one of 'segm', 'bbox' or 'keypoints'

        """
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.bbox_fmt = bbox_fmt.lower()
        assert self.bbox_fmt in ["voc", "coco", "yolo"]

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_dt = loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(
                self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type]
            )

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            if self.bbox_fmt == "coco":
                boxes = prediction["boxes"].tolist()
            else:
                boxes = prediction["boxes"]
                boxes = convert_to_xywh(boxes, fmt=self.bbox_fmt).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(
                    np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
                )[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            # boxes = prediction["boxes"]
            # boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def get_coco_object(dataset: Dataset):
    """Return COCO object from pycocotools

    Args:
        dataset: torch dataset containing the COCO object
    """

    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def explore_coco(coco_annotation: COCO):
    print("\nDisplaying COCO information:")
    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")

    img_ids = coco_annotation.getImgIds()
    print(f"Number of Images: {len(img_ids)}")
