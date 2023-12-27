# Dataset class for the COCO dataset
# Mostly taken from here: https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask


class CocoDetectionMiniTrain(torchvision.datasets.CocoDetection):
    """COCO Minitrain dataset. This dataset is a curated set of 25,000 train images
    from the 2017 Train COOO dataset. 
    
    Dataset can be found here https://github.com/giddyyupp/coco-minitrain
    """

    def __init__(self, image_folder: str, annotation_file: str, transforms):
        """Initialize the COCO dataset class
        
        Args:
            image_folder: path to the images
            annotation_file: path to the .json annotation file in coco format
        """
        super().__init__(root=image_folder, annFile=annotation_file)
        self._transforms = transforms
        
        #self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        """Retrieve samples from the dataset"""

        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    

def build_coco_mini(root: str, split: str, ):
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

    return
    

