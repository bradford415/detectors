# Dataset class for the COCO dataset
# Mostly taken from here: https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
import contextlib
import os
from pathlib import Path

import faster_coco_eval

# monkey-patches pycocotools to use the faster coco eval API; this must be imported before any imports to
# pycoco tools such as torchvision.datasets.CocoDetection; NOTE: using `import torchvision` does not import
# coco detection, it just calls torchvision.__init__. Once  this line is ran `class CocoDetectionDETR(torchvision.datasets.CocoDetection):`
# then python tries to import torchvision.datasets.CocoDetection which then imports pycocol tools;
# monkey patching dynamically replaces one module with another module, in the faster_coco_eval case, the modules
# being replaced are defined here: https://github.com/MiXaiLL76/faster_coco_eval/blob/488e6f04912d37ba470bd289e06c748ea2eb2f2c/faster_coco_eval/__init__.py#L10
faster_coco_eval.init_as_pycocotools()

import numpy as np
import torch
import torchvision

from detectors.data.coco_utils import PreprocessCoco, coco_stats
from detectors.data.transforms import transforms as T
from detectors.utils.box_ops import box_xyxy_to_cxcywh


class CocoDetectionDETR(torchvision.datasets.CocoDetection):
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
    ├─ annotations
    │  ├─ instances_train2017.json
    │  ├─ instances_val2017.json


    coco mintrain 25k:
    Inside "coco_minitrain_25k" create another directory named "annotations" and place the
    "instances_minitrain2017.json" and "instances_val2017.json" inside.
    The "instances_val2017.json" is from the original coco2017 dataset and can be found there.
    """

    def __init__(
        self,
        image_folder: str,
        annotation_file: str,
        num_classes: int,
        split: str,
        transforms: T = None,
        dev_mode: bool = False,
    ):
        """Initialize the COCO dataset class

        Args:
            image_folder: path to the root of the dataset images
            annotation_file: path to the .json annotation file in coco format
            num_classes: number of classes in the dataset; for yolo architectures this should be 80,
                          for detr-based architectures this should be max_class_id + 1 which is 91
            split: the dataset split type; train, val, or test
        """
        # Suppress coco prints while loading the image folder and annoation file
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                super().__init__(root=image_folder, annFile=annotation_file)

        self._transforms = transforms

        self.prepare = PreprocessCoco(return_masks=False, contiguous_cat_ids=False)

        # for DINO DETR, num_classes should always be set to the max_class_id + 1; if you check the COCO 2017 file
        # `instances_train2017.json` and look at the "categories" key, the max_id is "90" for "toothbrush" therefore
        # for DINO DETR we should set num_classes=91; example, if you have a dataset with ids 1-20, num_classes=21;
        # an example of different datasets is shown in the DINO code here:
        # https://github.com/IDEA-Research/DINO/blob/8758cf02146f306dc36babab4fff1f09c114c682/models/dino/dino.py#L721;
        # as reminder, COCO file indexing typically starts with 1;
        # NOTE: there's technicall an id=91=hairbrush but this is not included in the official annotation file so the max
        #       id in application is 90
        # explanation for why 91 instead of 80: https://github.com/facebookresearch/detr/issues/23#issuecomment-636322576
        # basically it's so they don't need a class mapping since the 80 `thing` classes which are actually
        # in the dataset are not contiguous; these missing objects will be treated as the background class
        # but since there's no actual labels the object never predicts and they saw no drop in performance with this;
        # the only con is that there will be a few extra parameters for the class prediction;
        # additionally, 91 is used because coco has IDs (0-90 = 91 classes) and with adding the no object class
        # we get 92 classes, so num_classes=91 because we care about the index value (even though there's technically
        # 92 w/ the no object) this post explains a little more:
        #   https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
        # list of the 91 coco class: https://gist.github.com/tersekmatija/9d00c4683d52d94cf348acae29e8db1a
        self.num_classes = num_classes

        # Extract dataset ontology
        categories_list = self.coco.loadCats(self.coco.getCatIds())
        self.class_names = {cat["id"]: cat["name"] for cat in categories_list}

        # Substantially reduces the dataset size to quickly test code
        if dev_mode:
            self.ids = self.ids[:32]

        # Display coco information of the current dataset; this should be placed at the end of the __init__()
        coco_stats(self, split)

    def __getitem__(self, index) -> tuple[torch.Tensor,]:
        """Retrieve and preprocess samples from the dataset

        Returns:
            1. a tensor of the augmented image
            2. a dictionary of processed labels for the image with keys:
                   boxes: normalized box coordinates [0, 1] for each object in the image
                          (num_objects, 4) where 4 = CXCYWH
                   labels: class ids for each object in the image (num_objects,)
                   image_id: the image_id from the coco annotation file
                   iscrowd: False if it is not a crowd  (num_objects,); the self.prepare
                            should have removed all crowd annotations
                   orig_size: the original image size (h, w)
                   size: the current image size after data augmentations like resizing (new_h, new_w)

        """

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

            image, target = self._transforms(image=image, target=target)

        return image, target


class CocoDetectionYolo(torchvision.datasets.CocoDetection):
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
    ├─ annotations
    │  ├─ instances_train2017.json
    │  ├─ instances_val2017.json


    coco mintrain 2k:
    Inside "coco_minitrain_25k" create another directory named "annotations" and place the
    "instances_minitrain2017.json" and "instances_val2017.json" inside.
    The "instances_val2017.json" is from the original coco2017 dataset and can be found there.
    """

    def __init__(
        self,
        image_folder: str,
        annotation_file: str,
        num_classes: int,
        split: str,
        transforms: T = None,
        dev_mode: bool = False,
    ):
        """Initialize the COCO dataset class

        Args:
            image_folder: path to the root of the dataset images
            annotation_file: path to the .json annotation file in coco format
            num_classes: number of classes in the dataset; for yolo architectures this should be 80,
                          for detr-based architectures this should be max_class_id + 1 which is 91
            split: the dataset split type; train, val, or test
        """
        # Suppress coco prints while loading the image folder and annoation file
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                super().__init__(root=image_folder, annFile=annotation_file)

        self._transforms = transforms

        self.prepare = PreprocessCoco(return_masks=False, contiguous_cat_ids=True)

        # for DINO DETR, num_classes should always be set to the max_class_id + 1; if you check the COCO 2017 file
        # `instances_train2017.json` and look at the "categories" key, the max_id is "90" for "toothbrush" therefore
        # for DINO DETR we should set num_classes=91; example, if you have a dataset with ids 1-20, num_classes=21;
        # an example of different datasets is shown in the DINO code here:
        # https://github.com/IDEA-Research/DINO/blob/8758cf02146f306dc36babab4fff1f09c114c682/models/dino/dino.py#L721;
        # as reminder, COCO file indexing typically starts with 1;
        # NOTE: there's technicall an id=91=hairbrush but this is not included in the official annotation file so the max
        #       id in application is 90
        # explanation for why 91 instead of 80: https://github.com/facebookresearch/detr/issues/23#issuecomment-636322576
        # basically it's so they don't need a class mapping since the 80 `thing` classes which are actually
        # in the dataset are not contiguous; these missing objects will be treated as the background class
        # but since there's no actual labels the object never predicts and they saw no drop in performance with this;
        # the only con is that there will be a few extra parameters for the class prediction;
        # additionally, 91 is used because coco has IDs (0-90 = 91 classes) and with adding the no object class
        # we get 92 classes, so num_classes=91 because we care about the index value (even though there's technically
        # 92 w/ the no object) this post explains a little more:
        #   https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
        # list of the 91 coco class: https://gist.github.com/tersekmatija/9d00c4683d52d94cf348acae29e8db1a
        self.num_classes = num_classes

        # Extract dataset ontology
        categories_list = self.coco.loadCats(self.coco.getCatIds())
        self.class_names = [category["name"] for category in categories_list]

        # Substantially reduces the dataset size to quickly test code
        if dev_mode:
            self.ids = self.ids[:128]

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

        # For albumentations
        # Convert bounding boxes from pascal_voc format to Yolo format including normalize between [0, 1];

        if self._transforms is not None:

            augs = self._transforms(
                image=np.array(image), bboxes=target["boxes"].numpy()
            )

            image = augs["image"]

            bboxes = torch.tensor(augs["bboxes"])

            # tl_x, tl_y, br_x, br_y -> cx, cy, w, h
            bboxes = box_xyxy_to_cxcywh(bboxes)

            # Normalize boxes between [0, 1]
            h, w = image.shape[-2:]
            bboxes = bboxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        # create a tensor of the sample index, object label and bboxes (num_objects, 6) where 6 = (sample_index, obj_class_id, cx, cy, h, w)
        # NOTE: the sample_index will be 0 for now but in the collate_fn it is filled in; this is the sample_index only within the batch
        target_tens = torch.zeros(len(target["labels"]), 6)
        cls_boxes = torch.cat((target["labels"][:, None], bboxes), dim=1)
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


def make_coco_transforms_dino_detr(dataset_split):
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


## TODO: This would probably make the most since as a cls method and name it from_data_split()
def build_coco(
    root: str,
    num_classes: int,
    dataset_split: str,
    model_type: str = "detr",
    dev_mode: bool = False,
):
    """Initialize the COCO dataset class

    Args:
        root: full path to the dataset root
        num_classes: number of classes in the dataset; for yolo architectures this should be 80,
                     for detr-based architectures this should be max_class_id + 1 which is 91
                     for the coco dataset; NOTE: the max_id of the "categories" key in the coco
                     annotation file is 90 (so 90 + 1 = 91)
        split: which dataset split to use; `train` or `val`
        model_tyoe: the name of the model to use (currently only `yolo` and `detr` are supported);
                    this is used to determine the transforms to use and if the coco classes need to
                    be converted to contiguous IDs;
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
    if "detr" in model_type.lower():
        data_transforms = make_coco_transforms_dino_detr(dataset_split)
        dataset = CocoDetectionDETR(
            image_folder=images_dir,
            annotation_file=annotation_file,
            num_classes=num_classes,
            transforms=data_transforms,
            dev_mode=dev_mode,
            split=dataset_split,
        )
    elif "yolo" in model_type.lower():
        data_transforms = make_coco_transforms_album(dataset_split)

        dataset = CocoDetectionYolo(
            image_folder=images_dir,
            annotation_file=annotation_file,
            num_classes=num_classes,
            transforms=data_transforms,
            dev_mode=dev_mode,
            split=dataset_split,
        )

    return dataset
