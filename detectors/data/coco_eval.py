import contextlib
import copy
import io
import logging
import os
import sys
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectors.utils import misc
from detectors.utils.distributed import all_gather

log = logging.getLogger(__name__)


class CocoEvaluator:
    """CocoEvaluator to calculate various APs used in many object detectors.

    Metrics and evaluation code is explained a little bit here: https://cocodataset.org/#detection-eval
    """

    def __init__(self, coco_gt, iou_types, useCats=True):
        """TODO
        Args:
            useCats: whether to use class labels in evaluation; this is standard for mAP calculation
            because the metric evaluates both localization and classification accuracy; pycocotools
            sets this to True by default:
                https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L511
        """
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].useCats = useCats

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.useCats = useCats

    def update(self, predictions: dict[dict]):
        """Compute the per image, per-category, per-area-range IoU metrics for the batch
        and store in a list self.eval_imgs["bbox"]

        Example: per-category will give us an IoU for each class (dog, cat, mouse)

        Args:
            predictions: a dictionary with keys as the image_id (from the coco file) for each image
                and values of a dict of model predictions keys:
                    scores: top `num_select` class probabilites which represent confidence (num_select,);
                            the top values are selected across num_queries*num_classes, therefore,
                            tehcnically a single query can have multiple classes predictions
                    labels: the class_id predictions chosen by topk class probs (num_select,)
                    boxes: the aboslute bboxes [0, orig_img_w/h] predictions chosen by topk class probs
                            (num_select, 4) where 4 = (x1, y1, x2, y2) in absolute coordinates
        """
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            # convert all detections across all images to a single list of results;
            # len(results) = num_images * min(num_select, num_detections)
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            # compute metrics `eval_imgs` of shape (num_classes, num_area_ranges, num_images);
            # this represents the per image, per-category, per-area-range evaluation results
            # for the batch;;
            # example: per-category will give us an IoU for each class (dog, cat, mouse)
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.params.useCats = self.useCats
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        """Update the coco_eval object with the image ids and evaluations from all processes"""
        for iou_type in self.iou_types:
            # self.eval_imgs["box"] is a list of evaluations for each batch, each elemnt
            # has shape (num_classes, num_area_ranges, batch_size), so concatenating along
            # the 2nd dim essentially stacks all the evaluations across batches together
            # (num_classes, num_area_ranges, batch_size * num_batches)
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(
                self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type]
            )

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            log.info("IoU metric: %s", iou_type)

            with print_to_logger():
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
        """Prepare results for COCO detection evaluation.

        1. Convert bounding boxes from XYXY to COCO format XYWH
        2. Convert labels and scores to lists if they are not already
        3. Create a list of dictionaries for the processed predictions

        Returns:
            coco_results: a list of dictionaries containing box predictions across all imagse in the
                          batch; contains keys:
                - image_id: the original image id from the COCO dataset
                - category_id: the class id of the predicted object
                - bbox: the bounding box in COCO format [x, y, width, height]
                - score: the confidence score of the prediction
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            if not isinstance(prediction["scores"], list):
                scores = prediction["scores"].tolist()
            else:
                scores = prediction["scores"]
            if not isinstance(prediction["labels"], list):
                labels = prediction["labels"].tolist()
            else:
                labels = prediction["labels"]

            try:
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
            except:
                import ipdb

                ipdb.set_trace()
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

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
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


@contextmanager
def print_to_logger(level=logging.INFO):
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer  # <-- redirect prints

    try:
        yield  # <-- run the "inside of the with-block" here
    finally:
        sys.stdout = old_stdout  # <-- restore stdout
        for line in buffer.getvalue().splitlines():
            log.log(level, line)  # <-- flush captured prints to logger


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):

    # collect a list of image ids and evaluations from all proccesses
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    # combine all image ids and evaluations from all processes into a single list
    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)
    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    # convert lists into numpy arrays
    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)  # along img dimension

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    """"""
    # collect the image ids and evaluations from all processes;
    # eval_imgs are a dictionary of evaluations w/ shape (num_classes, num_area_ranges, num_images)
    # see the end of the evaluate() function for each specific evaluation dictionary key
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    # update evaluations and image ids in the coco_eval object with the image ids and eval_imgs
    # from all processes
    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print(
            "useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType)
        )

    # remove duplicate image IDs and Category IDs
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))

    # sort the maxDets in ascending order; default is [1, 10, 100]
    # I'm not sure what this is used for but it's set here: https://github.com/nightrome/cocostuffapi/blob/675ff387d8297af0d95acd56ece4a36d9945893a/PythonAPI/pycocotools/cocoeval.py#L509
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    # determine the IoU computation method
    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU  # for bbox and segmentation
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks  # only for keypoints

    # compute IoU for each (image, category) pair
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds
    }

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]

    # run per-image evluation;
    # evaluateImg works by:
    #   1. sorting detection by score
    #   2. matches the detctions to GT using IoU threshold
    #   3. computes TP, FP, FN for each IoU threshold and area range
    #      - there's usually 10 IoU thresholds that are [0.5, 0.55, ..., 0.95]
    #      - `areaRng` represents the object size and is used to compute metrics
    #       for small, medium, and large objects
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]

    # each element in evalImgs is a dict with keys:
    #   - image_id: the original image id from the dataset
    #   - category_id: the class id of the class that was evaluated
    #     (we loop through all classes to get a per-class IoU )
    #   - aRng: the area range of the object (small, medium, large)
    #           for example: gt bboxes with pixel area between [0, 32^2) are small
    #           see here for the exact ranges: https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L509
    #   - maxDet: the maximum number of detections to consider; defined as [1, 10, 100]
    #             in which most metrics are reported for maxDet=100; if there
    #             are less than 100 detections, the rest are ignored which is
    #             e.g., if there's 13 gt objects we only want our model to predict
    #                   13 objects, not 100
    #   - dtIds: list of unique detection IDs (after sorting by confidence and truncating at maxDet)
    #            which essentially correspond to which object prediction it refers to, almost like
    #            indexing the decttion; NOTE: this is NOT the class id
    #            these predictions are specific to the image and category pair; they're not reused
    #            across images or categories; coco.loadRes() buckets these predictions and then they
    #            are indexed when evaluateImg() is called
    #   - gtIds: list of uniuqe ground truth ids of the objects for the specific image and category pair
    #            (after sorting so that non-ignored ones come first); again, this is NOT the class id
    #   NOTE: dtMatches and gtMatches are mirrors of each other, they work together to compute the final
    #         metrics
    #   - dtMatches: the matches of the detections to the ground truth objects; stores the
    #                gtId of the BEST matched ground truth or 0 if unmatched; the best match is the
    #                the largest IoU between the detection and all the gt objects as long as it is above
    #                the IoU threshold; a detection is unmatched if it's mean below the  IoU threshold or
    #                all the GT objects are already matched to other detections; 0s are treated as
    #                fasle positives (FP); shape is
    #                (num_iou_thresholds, num_detections) num_iou_thresholds=[0.50, 0.55, …, 0.95]
    #                and num_detections is the number of detections for the (image, category) pair
    #   - gtMatches: the matches of the ground truth objects to the detections; stores the
    #                detection id which matched the groud truth (again not class ids)
    #                (num_iou_thresholds, num_gt_objects)
    #   - dtScores: a list of detection confidence scores aligned with dtIDs; later, COCO sorts
    #               detections globally by score when building precision-recall curves
    #     NOTE: i don't fully know what these are used for but I don't think they are used often,
    #           they might be in the coco file
    #   - gtIgnore: marks which ground turhts are ignored (outside area range, difficult, etc)
    #   - dtIgnore:marks which detections are ignored

    # this is NOT in the pycocotools code, but could be done outside;
    # reshape evalImgs to (num_classes, num_area_ranges, num_images)
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)

    return p.imgIds, evalImgs


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
