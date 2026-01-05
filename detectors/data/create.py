from pathlib import Path
from typing import Any, Optional

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from detectors.data.collate import get_collate
from detectors.data.datasets.coco import CocoDetectionDETR, CocoDetectionYolo
from detectors.data.transforms import TRANSFORM_REGISTRY
from detectors.data.transforms import transforms as T

dataset_map = {
    "coco_detection_detr": CocoDetectionDETR,
    "coco_detection_yolo": CocoDetectionYolo,
}


def make_dino_detr_transforms(dataset_split):
    """Initialize the DINO DETR transforms for the coco dataset

    dino detr utilizes the following transforms:
        train:
            - RandomHorizontalFlip (p=0.5)
            - RandomSelect (p=0.5 for both transforms)
                - Transform 1:
                    - RandomResize(short_side=random(`short_side_scales`), max_size=1333)
                - Transform 2:
                    - RandomResize(short_side=random(`short_side_scales_2`))
                    - RandomSizeCrop(min=`scales_crop[0]`, max=`scales_crop[1]`)
                    - RandomResize(short_size=random(`short_side_scales`), max_size=1333)
            - Normalize ([0, 1] -> subtract mean, divide by std)
        val:
            - Resize the shorter side to 800 and the longer side to max_size=1333
            - Normalize ([0, 1] -> subtract mean, divide by std)
        test:
            - Resize the shorter side to 800 and the longer side to max_size=1333
            - Normalize ([0, 1] -> subtract mean, divide by std)

    Args:
        dataset_split: which dataset split to use; `train` or `val`

    """

    # scales (randomly chosen from) to resize the shorter side of the image to;
    # this is during just RandomResize (if selected) and the 2nd RandomResize after cropping
    # (if this transform is randomly selected)
    short_side_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    # the maximum allowable size of the longer side when resizing; if the longer side would
    # be greater than this, a new short side is calculated such that the longer side would be `max_size`
    max_size = 1333

    # scales (randomly chosen from) to resize the shorter side of the image to before cropping if
    # this transform is randomly selected
    short_side_scales_2 = [400, 500, 600]

    # the height and width of the crop size randomly chosen between [min, max]
    # (or the h/w of the image if it exceeds); the height and width of the crop start at a randomly
    # selected point representing the top-left corner of the crop region; top-left corner is chosen
    # randomly between [0, image_height - h] and [0, image_width - w]
    scales_crop = [384, 600]

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    if dataset_split == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(short_side_scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize(short_side_scales_2),
                            T.RandomSizeCrop(*scales_crop),
                            T.RandomResize(short_side_scales, max_size=max_size),
                        ]
                    ),
                ),
                normalize,
            ]
        )
    elif dataset_split == "val" or dataset_split == "test":
        return T.Compose(
            [
                T.RandomResize([max(short_side_scales)], max_size=max_size),
                normalize,
            ]
        )
    else:
        raise ValueError(f"unknown dataset split {dataset_split}")


def make_config_transforms(config_transforms: dict):
    """Initialize the transformations to use from a config file"""

    all_transforms = []
    for transform in config_transforms["transforms"]:
        all_transforms.append(
            TRANSFORM_REGISTRY.get(transform["type"])(**transform.get("params", {}))
        )

    return T.Compose(all_transforms, config_transforms.get("policy"))


def make_yolo_transforms(dataset_split, image_size: int = 416):
    """Initialize transforms for the coco dataset using the Albumentations library

    Args:
        dataset_split: which dataset split to use; `train` or `val`
    """

    normalize_album = A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
            ),
            # Convert the image to PyTorch tensor
            ToTensorV2(),
        ]
    )

    if dataset_split == "train":
        album_transforms = A.Compose(
            [
                # Rescale an image so that maximum side is equal to image_size
                A.LongestMaxSize(max_size=image_size),
                # Pad remaining areas with zeros
                A.PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                normalize_album,
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", min_visibility=0.0, label_fields=[]
            ),
        )
        return album_transforms
    elif dataset_split == "val":
        return A.Compose(
            [
                A.LongestMaxSize(max_size=image_size),
                # Pad remaining areas with zeros
                A.PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                normalize_album,
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", min_visibility=0.0, label_fields=[]
            ),
        )
    elif dataset_split == "test":
        return A.Compose(
            [
                normalize_album,
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", min_visibility=0.0, label_fields=[]
            ),
        )
    else:
        raise ValueError(f"unknown dataset split {dataset_split}")


def create_dataset(
    dataset_name: str,
    split: str,
    root: str,
    num_classes: int,
    transforms_config: dict[str],
    model_name: Optional[str] = None,
    dev_mode: bool = False,
):
    """Initialize the dataset class

    Args:
        dataset_name: the name of the dataset to intitialize; should be in `dataset_map`
        TODO
    """
    root = Path(root)

    # Set path to images and annotations; TODO: specfic to COCO so need to make it more modular
    # maybe make it a class property
    if split == "train":
        images_dir = root / "images" / "train2017"
        annotation_file = root / "annotations" / "instances_train2017.json"
    elif split == "val":
        images_dir = root / "images" / "val2017"
        annotation_file = root / "annotations" / "instances_val2017.json"
    elif split == "test":
        images_dir = root / "images" / "test2017"
        annotation_file = root / "annotations" / "instances_test2017.json"

    # create the transforms for the specific model
    if model_name == "dino_detr":
        data_transforms = make_dino_detr_transforms(split)
    elif model_name == "yolo":
        data_transforms = make_yolo_transforms(split)
    else:
        # rt-detr
        data_transforms = make_config_transforms(transforms_config)

    if dataset_name == "coco_detection_detr":
        dataset = CocoDetectionDETR(
            images_dir, annotation_file, num_classes, split, data_transforms, dev_mode
        )
    else:
        raise ValueError(f"dataset: {dataset_name} not recognized")

    return dataset


def create_dataloader(
    is_distributed: bool,
    dataset: Dataset,
    batch_size_per_gpu: int,
    num_workers: int,
    collate_name: str,
    collate_params: dict[str, Any],
    drop_last: bool = True,
    shuffle: bool = False,
    pin_memory: bool = False,
):
    if is_distributed:
        # ensures that each process gets a different subset of the dataset in distributed mode
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        # using RandomSampler and SequentialSampler w/ default parameters is the same as using
        # shuffle=True and shuffle=False in the DataLoader, respectively; if you pass a Sampler into
        # the DataLoader, you cannot set the shuffle parameter (mutually exclusive)
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            # used for validation
            sampler = torch.utils.data.SequentialSampler(dataset)

    batch_sampler = torch.utils.data.BatchSampler(
        sampler, batch_size=batch_size_per_gpu, drop_last=drop_last
    )

    collate = get_collate(collate_name, collate_params)

    # NOTE: Consider adding ability to skip batch_sampler (e.g., pass sampler insted of batch_sampler)
    #       for example batched images in balidation sometimes hurt accuracy due to padding,
    #       though I don't fully understand why since the padding is masked
    if batch_sampler is not None:
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
    else:
        # TODO: this isn't possible at the moment since we always create a batch_sampler above
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            sampler=sampler,
            collate_fn=collate,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    return dataloader
