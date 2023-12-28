# Dataset class for the COCO dataset
# Mostly taken from here: https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from pycocotools import mask as coco_mask

from detectors.data.coco_utils import ConvertCocoToYolo
from detectors.utils.display import explore_coco


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

        #self.prepare = ConvertCocoToYolo()

    def __getitem__(self, index):
        """Retrieve and preprocess samples from the dataset"""
        
        # Retrieve the pil image and its annotations
        # annotations is a list of dicts corresponding to every object in the image
        # Each dict contains ground truth information of the object such as bbox and segementation
        image, annotations = super().__getitem__(index)

        # Match the randomly sampled index with the image_id
        image_id = self.ids[index]
        print()

        # Preprocess the input data before passing it to the model
        target = {"image_id": image_id, "annotations": annotations}
        #image, target = self.prepare(image, target)
        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target


class Preprocess:
    """
    """

    def __init__(self):
        pass
    
    def __call__(self, ):
        pass



def build_coco_mini(
    root: str,
    split: str,
):
    """Initialize the COCO dataset class

    Args:
        root: full path to the dataset root
        split: which dataset split to use
    """
    coco_root = Path(root)

    if split == "train":
        images_dir = coco_root / "images" / "train2017"
        annotation_file = coco_root / "annotations" / "instances_minitrain2017.json"
    elif split == "val":
        images_dir = coco_root / "images" / "val2017"
        annotation_file = coco_root / "annotations" / "instances_val2017.json"

    dataset = CocoDetectionMiniTrain(images_dir, annotation_file)

    return dataset
