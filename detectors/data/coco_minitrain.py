# Dataset class for the COCO dataset
# Mostly taken from here: https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from detectors.data import transforms as T
from detectors.data.coco_utils import PreprocessCoco, explore_coco


class CocoDetectionMiniTrain(torchvision.datasets.CocoDetection):
    """COCO Minitrain dataset. This dataset is a curated set of 25,000 train images
    from the 2017 Train COOO dataset.

    Dataset can be found here https://github.com/giddyyupp/coco-minitrain
    """

    def __init__(self, image_folder: str, annotation_file: str, transforms: T = None):
        """Initialize the COCO dataset class

        Args:
            image_folder: path to the images
            annotation_file: path to the .json annotation file in coco format
        """
        super().__init__(root=image_folder, annFile=annotation_file)
        self._transforms = transforms

        explore_coco(self.coco)

        self.prepare = PreprocessCoco()

    def __getitem__(self, index):
        """Retrieve and preprocess samples from the dataset"""

        # Retrieve the pil image and its annotations
        # Annotations is a list of dicts; each dict in the list is an object in the image
        # Each dict contains ground truth information of the object such as bbox, segementation and image_id
        image, annotations = super().__getitem__(index)

        # Match the randomly sampled index with the image_id
        image_id = self.ids[index]

        # Preprocess the input data before passing it to the model; see PreprocessCoco() for more info
        target = {"image_id": image_id, "annotations": annotations}
        image, target = self.prepare(image, target)
        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target


def make_coco_transforms(dataset_split):
    """Initialize transforms for the coco dataset

    These transforms are based on torchvision transforms but are overrided in data/transforms.py
    This allows for slight modifications in the the transform

    Args:
        dataset_split: which dataset split to use; `train` or `val`

    """

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    scales = [1024]

    if dataset_split == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomResize(scales),
                T.RandomSizeCrop(512, 512),
                # T.RandomSelect(
                #     T.Compose(
                #         [
                #             T.RandomResize(scales),
                #             T.RandomSizeCrop(384, 600),
                #             T.RandomResize(scales, max_size=1333),
                #         ]
                #     ),
                # ),
                normalize,
            ]
        )

    if dataset_split == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

    raise ValueError(f"unknown {dataset_split}")


def build_coco_mini(
    root: str,
    dataset_split: str,
):
    """Initialize the COCO dataset class

    Args:
        root: full path to the dataset root
        split: which dataset split to use; `train` or `val`
    """
    coco_root = Path(root)

    # Set path to images and annotations
    if dataset_split == "train":
        images_dir = coco_root / "images" / "train2017"
        annotation_file = coco_root / "annotations" / "instances_minitrain2017.json"
    elif dataset_split == "val":
        images_dir = coco_root / "images" / "val2017"
        annotation_file = coco_root / "annotations" / "instances_val2017.json"

    # Create the data augmentation transforms
    data_transforms = make_coco_transforms(dataset_split)

    dataset = CocoDetectionMiniTrain(
        image_folder=images_dir,
        annotation_file=annotation_file,
        transforms=data_transforms,
    )

    return dataset
