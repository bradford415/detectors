# Dataset class for the COCO dataset
# Mostly taken from here: https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
import contextlib
import os
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from detectors.data import transforms as T
from detectors.data.coco_utils import PreprocessCoco, coco_stats


class CocoDetection(torchvision.datasets.CocoDetection):
    ## TODO: can probably convert this to regular coco
    """COCO object detection dataset from 2017 which has 80 classes.
    The dataset can be downloaded by running the script in scripts/bash/download_coco.sh
    or  can be found here: https://cocodataset.org/#home

    One could also use the coco minitrain dataset. This dataset is a curated set of
    25,000 train images from the 2017 Train COOO dataset. This dataset can be found here
    https://github.com/giddyyupp/coco-minitrain.

    Official Coco annotation format: https://cocodataset.org/#format-data

    Create a file heiarchy as the following:
    coco/
    ├─ images/
    │  ├─ train_2017
    │  │  ├─ train_images.jpg
    │  ├─ val_2017
    │  │  ├─ train_images.jpg
    │  ├─ annotations
    │  │  ├─ instances_train2017.json
    │  │  ├─ instances_val2017.json

    Inside "coco_minitrain_25k" create another directory named "annotations" and place the
    "instances_minitrain2017.json" and "instances_val2017.json" inside.
    The "instances_val2017.json" is from the original coco2017 dataset and can be found there.
    """

    def __init__(
        self,
        image_folder: str,
        annotation_file: str,
        split: str,
        transforms: T = None,
        dev_mode: bool = False,
    ):
        """Initialize the COCO dataset class

        Args:
            image_folder: path to the images
            annotation_file: path to the .json annotation file in coco format
            split: the dataset split type; train, val, or test
        """
        # Suppress coco prints while loading the image folder and annoation file
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                super().__init__(root=image_folder, annFile=annotation_file)

        self._transforms = transforms

        self.prepare = PreprocessCoco()

        self.num_classes = 80

        # Extract dataset ontology
        categories_list = self.coco.loadCats(self.coco.getCatIds())
        self.class_names = [category["name"] for category in categories_list]

        # Substantially reduces the dataset size to quickly test code
        if dev_mode:
            self.ids = self.ids[:32]

        # Display coco information of the current dataset; this should be placed at the end of the __init__()
        coco_stats(self, split)

    def __getitem__(self, index):
        """Retrieve and preprocess samples from the dataset"""

        # Retrieve the pil image and its annotations
        # Annotations is a list of dicts; each dict in the list is an object in the image
        # Each dict contains ground truth information of the object such as bbox, segementation and image_id
        image, annotations = super().__getitem__(index)

        # Match the randomly sampled index with the image_id; self.ids contains the image_ids in the train set
        image_id = self.ids[index]

        file_name = self.coco.loadImgs(image_id)[0]["file_name"]
        image_path = self.root / file_name

        # Preprocess the input data before passing it to the model; see PreprocessCoco() for more info
        target = {
            "image_id": image_id,
            "image_path": image_path,
            "annotations": annotations,
        }

        image, target = self.prepare(image, target)

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        # create a tensor of the sample index, object label and bboxes (num_objects, 6) where 6 = (sample_index, obj_class_id, cx, cy, h, w)
        # NOTE: the sample_index will be 0 for now but in the collate_fn it is filled in; this is the sample_index only within the batch
        target_tens = torch.zeros(len(target["labels"]), 6)
        cls_boxes = torch.cat((target["labels"][:, None], target["boxes"]), dim=1)
        target_tens[:, 1:] = cls_boxes

        return image, target_tens, target


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

    if dataset_split == "train":
        return T.Compose(
            [
                # T.RandomHorizontalFlip(),
                # T.RandomResize(scales),
                # T.CenterCrop((512, 512)),
                normalize,
            ]
        )
    elif dataset_split == "val":
        return T.Compose(
            [
                # T.RandomResize([512]),
                normalize,
            ]
        )
    elif dataset_split == "test":
        return T.Compose(
            [
                normalize,
            ]
        )
    else:
        raise ValueError(f"unknown dataset split {dataset_split}")


def make_coco_transforms_album(dataset_split):
    """Initialize transforms for the coco dataset using the Albumentations library

    Args:
        dataset_split: which dataset split to use; `train` or `val`
    """

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    if dataset_split == "train":
        album_transforms = A.Compose(
            [
                # # Rescale an image so that maximum side is equal to image_size
                # A.LongestMaxSize(max_size=image_size),
                # # Pad remaining areas with zeros
                # A.PadIfNeeded(
                #     min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
                # ),
                A.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", min_visibility=0.0, label_fields=[]
            ),
        )
        return T.Compose(
            [
                # T.RandomHorizontalFlip(),
                # T.RandomResize(scales),
                # T.CenterCrop((512, 512)),
                album_transforms,
                normalize,
            ]
        )
    elif dataset_split == "val":
        return T.Compose(
            [
                # T.RandomResize([512]),
                normalize,
            ]
        )
    elif dataset_split == "test":
        return T.Compose(
            [
                normalize,
            ]
        )
    else:
        raise ValueError(f"unknown dataset split {dataset_split}")


## TODO: This would probably make the most since as a cls method and name it from_data_split()
def build_coco(
    root: str,
    dataset_split: str,
    dev_mode: bool = False,
):
    """Initialize the COCO dataset class

    Args:
        root: full path to the dataset root
        split: which dataset split to use; `train` or `val`
        dev_mode: Whether to build the dataset in dev mode; if true, this only uses a few samples
                         to quickly run the code
    """
    coco_root = Path(root)

    # Set path to images and annotations
    if dataset_split == "train":
        images_dir = coco_root / "images" / "train2017"
        annotation_file = coco_root / "annotations" / "instances_train2017.json"
    elif dataset_split == "val":
        images_dir = coco_root / "images" / "val2017"
        annotation_file = coco_root / "annotations" / "instances_val2017.json"
    elif dataset_split == "test":
        images_dir = coco_root / "images" / "test2017"
        annotation_file = coco_root / "annotations" / "instances_test2017.json"

    # Create the data augmentation transforms
    data_transforms = make_coco_transforms(dataset_split)

    dataset = CocoDetection(
        image_folder=images_dir,
        annotation_file=annotation_file,
        transforms=data_transforms,
        dev_mode=dev_mode,
        split=dataset_split,
    )

    return dataset


def build_coco_mini(
    root: str,
    dataset_split: str,
    dev_mode: bool = False,
):
    """Initialize the COCO dataset class

    Args:
        root: full path to the dataset root
        split: which dataset split to use; `train` or `val`
        dev_mode: Whether to build the dataset in dev mode; if true, this only uses a few samples
                         to quickly run the code
    """
    coco_root = Path(root)

    # Set path to images and annotations
    if dataset_split == "train":
        images_dir = coco_root / "images" / "train2017"
        annotation_file = coco_root / "annotations" / "instances_minitrain2017.json"
    elif dataset_split == "val":
        images_dir = coco_root / "images" / "val2017"
        annotation_file = coco_root / "annotations" / "instances_val2017.json"
    elif dataset_split == "test":
        images_dir = coco_root / "images" / "test2017"
        annotation_file = coco_root / "annotations" / "instances_test2017.json"

    # Create the data augmentation transforms
    data_transforms = make_coco_transforms(dataset_split)

    dataset = CocoDetection(
        image_folder=images_dir,
        annotation_file=annotation_file,
        transforms=data_transforms,
        dev_mode=dev_mode,
        split=dataset_split,
    )

    return dataset
