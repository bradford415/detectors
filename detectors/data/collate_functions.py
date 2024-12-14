from typing import Dict, Tuple

import torch
from torch.nn import functional as F


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


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
