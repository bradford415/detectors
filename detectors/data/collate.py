import random
from typing import Dict, Optional, Tuple

import torch
from torch.nn import functional as F

from detectors.data.data import NestedTensor


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def collate_fn_pad(batch: list[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> None:
    """Collect samples appropriately to be used at each iteration in the train loop

    At each train iteration, the DataLoader returns a batch of samples.
    E.g., for images, annotations in train_loader

    Args:
        batch: A batch of samples from the dataset. The batch is a list of
            samples, each sample containg a tuple of (image, image_annotations).
    """

    # Convert a batch of images and annoations [(image, annoations), (image, annoations), ...]
    # to (image, image), (annotations, annotations), ... ; this operation is called iterable unpacking
    images, targets, annotations = zip(*batch)  # images (C, H, W)

    # Resize images to input shape
    images = torch.stack([resize(img, 416) for img in images])

    # The below padding method is from the DETR repo here:
    # https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/util/misc.py#L307

    # Zero pad images on the right and bottom of the image with the max h/w of the batch;
    # this allows us to batch images of different sizes together;
    # in the current implementation, padding should only be applied for the validation set
    # TODO: make this more efficient now that targets is a tensor
    channels, max_h, max_w = images[0].shape
    for image in images[1:]:
        if image.shape[1] > max_h:
            max_h = image.shape[1]
        if image.shape[2] > max_w:
            max_w = image.shape[2]

    # Initalize tensor of zeros for 0-padding and copy the images into the top_left of each padded batch
    batch_size = len(batch)
    padded_images = torch.zeros(
        batch_size, channels, max_h, max_w
    )  # (B, C, batch_max_H, batch_max_W)
    for img, padded_img in zip(images, padded_images):
        padded_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    # (B, C, H, W)
    # images = torch.stack(images, dim=0) # This was written before the padding above

    # Add sample index to targets
    for i, boxes in enumerate(targets):
        boxes[:, 0] = i
    targets = torch.cat(targets, 0)
    # This is what will be returned in the main train for loop (samples, targets)
    return padded_images, targets, annotations


## TODO remove padding
def collate_fn(batch: list[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> None:
    """Collect samples appropriately to be used at each iteration in the train loop

    At each train iteration, the DataLoader returns a batch of samples.
    E.g., for images, annotations in train_loader

    Args:
        batch: A batch of samples from the dataset. The batch is a list of
            samples, each sample containg a tuple of (image, image_annotations).
    """

    # Convert a batch of images and annoations [(image, annoations), (image, annoations), ...]
    # to (image, image), (annotations, annotations), ... ; this operation is called iterable unpacking
    images, targets, annotations = zip(*batch)  # images (C, H, W)

    # Resize images to input shape
    # images = torch.stack([resize(img, 416) for img in images])

    # The below padding method is from the DETR repo here:
    # https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/util/misc.py#L307

    # Zero pad images on the right and bottom of the image with the max h/w of the batch;
    # this allows us to batch images of different sizes together;
    # in the current implementation, padding should only be applied for the validation set
    # TODO: make this more efficient now that targets is a tensor
    # channels, max_h, max_w = images[0].shape
    # for image in images[1:]:
    #     if image.shape[1] > max_h:
    #         max_h = image.shape[1]
    #     if image.shape[2] > max_w:
    #         max_w = image.shape[2]

    # # Initalize tensor of zeros for 0-padding and copy the images into the top_left of each padded batch
    # batch_size = len(batch)
    # padded_images = torch.zeros(
    #     batch_size, channels, max_h, max_w
    # )  # (B, C, batch_max_H, batch_max_W)
    # for img, padded_img in zip(images, padded_images):
    #     padded_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    # (B, C, H, W)
    images = torch.stack(images, dim=0)  # This was written before the padding above

    # Add sample index to targets
    for i, boxes in enumerate(targets):
        boxes[:, 0] = i
    targets = torch.cat(targets, 0)
    # This is what will be returned in the main train for loop (samples, targets)
    return images, targets, annotations


def collate_fn_nested_tensor(
    batch: list[tuple[torch.tensor, dict]],
) -> tuple[NestedTensor, tuple[dict]]:
    """Collate function used for detr-based detectors

    Args:
        batch: a list of samples where len() is batch_size and each element is a tuple of:
                   1. image tensor (c, h, w)
                   2. dictionary of keys:
                          ['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size']

    Returns:
        a tuple of:
            1. a NestedTensor object
            2. a tuple of dictionaries corresponding to the labels & metadata for the tensors
               in the NestedTensor.tensors
    """
    # Create a list of two elements where:
    #   - batch[0] = tuple of image tensors
    #   - batch[1] = dict which correspond to the labels of the image tensors
    batch = list(zip(*batch))

    # Create a nested tensor from the tuple of tensors
    batch[0] = NestedTensor.from_tensor_list(batch[0])
    return tuple(batch)


def collate_fn_test(batch: list[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> None:
    """Pack samples for the test set; no padding is needed because the batch size is 1

    Args:
        batch: A batch of samples from the dataset. The batch is a list of
            samples, each sample containg a tuple of (image, image_annotations).
    """

    # Convert a batch of images and annoations [(image, annoations), (image, annoations), ...]
    # to (image, image), (annotations, annotations), ... ; this operation is called iterable unpacking
    images, annotations = zip(*batch)  # images (C, H, W)

    images = torch.stack(images)

    # This is what will be returned in the main train for loop (samples, targets)
    return images, annotations


def get_collate(collate_name: str, collate_params: Optional[dict]) -> callable:
    """Return the appropriate collate function based on the detector

    Agrgs:
        detector_name: the name of the detector
    """

    if collate_name == "collate_fn_nested_tensor":
        # dino
        return collate_fn_nested_tensor
    elif collate_name == "BatchImageCollateFuncion":
        # rt detr
        return BatchImageCollateFuncion(**collate_params)
    elif collate_name == "collate_fn":
        # yolov3 yolov4
        return collate_fn
    else:
        raise ValueError(f"No collate_fn found for: {collate_name}")


# TODO START HERE consider not need this base collate function
class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    def __call__(self, items):
        raise NotImplementedError("")


class BatchImageCollateFuncion(BaseCollateFunction):
    def __init__(
        self,
        scales=None,
        stop_epoch=None,
    ) -> None:
        """TODO"""
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        # self.interpolation = interpolation

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            # NOTE: we should not need to resize the box coordintes here since
            #       they are normalized (percentage based)
            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)

            # NOTE: masks not implemented
            if "masks" in targets[0]:
                for tg in targets:
                    tg["masks"] = F.interpolate(tg["masks"], size=sz, mode="nearest")
                raise NotImplementedError("")

        return images, targets
