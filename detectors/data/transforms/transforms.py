# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# The primary use of overwriting the transforms is to handle
# the bounding box transformations as well
"""
Transforms and data augmentation for both image + bbox.
"""
import random
import sys
from this import d
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as T
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.transforms import RandomCrop, RandomErasing
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Mask

from detectors.utils.box_ops import box_xyxy_to_cxcywh
from detectors.utils.misc import interpolate


def crop(image, target, region):
    """Crop the image and adjust the target bounding boxes and masks accordingly.

    Args:
        image: a PIL image to be cropped
        target: a dictionary of ground truth information containing keys:
                  boxes (tl_x, tl_y, br_x, br_y) labels, image_id, area, iscrowd, orig_size, size
        region: a tuple (i, j, h, w) where i and j are the top-left corner coordinates
                and h and w are the height and width of the crop region
    """
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target["masks"] = target["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target["masks"].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return flipped_image, target


def resize(
    image: PIL.Image, target, size: Union[int, Tuple], max_size: Optional[int] = None
):
    """Resize an image, and adjusts the targets' bboxes, such that the shortest
    side of the image is resized to `size` preserving the aspect ratio, but if doing
    so would cause the longest side to exceed `max_size`, then it calculats a new `size`
    by assuming the new longer side = `max_size` (still keeping the aspect ratio)

    Args:
        image: a PIL image to be resized; NOTE: image.size returns (w, h) but when you
               access the pixels with np.array the dims are returned (h, w, c)
        target: a dictionary of ground truth information containing keys:
                  boxes (tl_x, tl_y, br_x, br_y) labels, image_id, area, iscrowd, orig_size, size
        size: if integer (default), the randomly selected size to resize the shortest side
                of the image to while maintaining the aspect ratio
              if tuple, the (height, width) to resize the image to
        max_size: the upper bound for longest side of the image when resizing
                 (the aspect ratio is still maintained); default 1333
    """
    # size can be min_size (scalar) or (w, h) tuple

    # Get the aspect ratio
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        """Calculate the new h/w of the resized image; if the long side would exceed max_size
        then calculate the new short side size given the new long side size = max_size
        """
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))

            # if the longer size would be larger than the desired size, then compute the
            # new size of the shorter side assuming the longer side is max_size;
            # these are just proportion calculations;
            #   Ex: long_side=640 & short_side=480 -> (640 / 480 = max_size / size)
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        # return if the shorter image side already equals the desired size
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        # Compute the size of the longer side; the shoter side will be the desired size
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(
        float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size)
    )
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target["masks"] = (
            interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0]
            > 0.5
        )

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target["masks"] = torch.nn.functional.pad(
            target["masks"], (0, padding[0], 0, padding[1])
        )
    return padded_image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop:
    """Randomly crops an image to a random size between `min_size` and `max_size`
    (but not larger than the image size); the location of the crop is randomly chosen
    specified by i, j (row, col) which represents the top-left corner of the crop region and
    i = [0, image_height - h] and j = [0, image_width - w] (0 stays because its the top-left corner);
    the target bboxes are cropped with the region such that if their new area is positive
    (even very small), they are kept, this means if their new area is  <= 0 they are removed
    from the new cropped image.
    """

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = Tv1.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize:
    """Resize an image, and adjusts the targets' bboxes, such that the shortest
    side of the image is resized to `size` preserving the aspect ratio, but if doing
    so would cause the longest side to exceed `max_size`, then it calculats a new `size`
    by assuming the new longer side = `max_size` (still keeping the aspect ratio)
    """

    def __init__(self, sizes: List[int], max_size=None):
        """
        Args:
            sizes: list of sizes to randomly resize from (e.g., [512, 608, 1024])
        """
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img: PIL.Image, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class ToTensorNoNormalization:
    def __call__(self, pil_image: PILImage, target) -> torch.Tensor:
        """Converts a PIL Image (H, W, C) to a Tensor (B, C, H, W) without normalization

        This also converts the target["bbox"]

        Most of the code is from here: https://pytorch.org/vision/main/_modules/torchvision/transforms/functional.html#to_tensor

        pil_image: PIL image to be converted to a tensor during training
        target: Gt detection labels; TODO: these may have to be normalized when I put this back in
        """

        # handle PIL Image
        mode_to_nptype = {
            "I": np.int32,
            "I;16" if sys.byteorder == "little" else "I;16B": np.int16,
            "F": np.float32,
        }

        # Convert pil to tensor
        img = torch.from_numpy(
            np.array(pil_image, mode_to_nptype.get(pil_image.mode, np.uint8), copy=True)
        )

        if pil_image.mode == "1":
            img = 255 * img
        img = img.view(pil_image.size[1], pil_image.size[0], img.shape[-1])
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()

        # Convert bounding boxes from Coco format to Yolo format; tl_x, tl_y, br_x, br_y -> cx, cy, w, h
        target = target.copy()
        h, w = img.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            # boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes

        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=torch.get_default_dtype()), target
        else:
            return img, target


class RandomErasing(object):
    def __init__(self, *args, **kwargs):
        self.eraser = RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class RandomIoUCrop(T.RandomIoUCrop):
    """A wrapper around torchvision.transforms.v2.RandomIoUCrop to add a proability
    of this transform being used
    """

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale,
            max_scale,
            min_aspect_ratio,
            max_aspect_ratio,
            sampler_options,
            trials,
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


class Normalize:
    """Normalize an image by mean and standard deviation and convert the bounding box
    coordinates XYXY -> CXCYWH and normalize between [0, 1] (i.e., normalize by image dimensions)
    """

    def __init__(
        self,
        mean: list[float],
        std: list[float],
        bbox_fmt: str = "cxcywh",
        convert_to_tv_tensor: bool = False,
    ):
        """Intializes the normalization transforms

        Args:
            mean: the mean of the dataset for each channel
            std: the standard deviation of the dataset for each channel
            convert_to_tv_tensor: whether to convert the bounding boxes to torchvision.tv_tensors.BoundingBoxes;
                                  used for models like rtdetrv2
        """
        self.mean = mean
        self.std = std
        self.bbox_fmt = bbox_fmt
        self.convert_to_tv_tensor = convert_to_tv_tensor

    def __call__(self, image, target=None) -> Optional[torch.tensor]:
        """Normalize an image by the mean/std and convert the target
        bounding boxes to yolo format [center_x, center_y, w, h] normalized by
        the image dimensions (0-1).
        """

        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None

        # Convert bounding boxes from XYXY to CXCYWH format including normalize between [0, 1];
        # tl_x, tl_y, br_x, br_y -> cx, cy, w, h
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)

            # Normalize boxes between [0, 1]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)

            if self.convert_to_tv_tensor:
                boxes = convert_to_tv_tensor(boxes, "boxes", self.bbox_fmt, (h, w))
            target["boxes"] = boxes
        return image, target


def convert_to_tv_tensor(
    tensor: Tensor, key: str, box_format="cxcywh", spatial_size=None
) -> Tensor:
    """Convert bounding boxes to torchvision.tv_tensors.BoundingBoxes

    Used for torchvision.transforms.v2

    Args:
        tensor (Tensor): input tensor
        key (str): transform to key
        box_format: the format of the bounding boxes
        spatial_size: the spatial size of the image (h, w); also known as canvas_size
    Return:
        Dict[str, TV_Tensor]
    """
    assert key in ("boxes",), "Only support 'boxes'"

    _boxes_keys = ["format", "canvas_size"]

    if key == "boxes":
        # NOTE: even though canvas_size is (h, w), the bounding boxes a `tensor` should match the
        #       `box_format`
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == "masks":
        return Mask(tensor)


class Unnormalize:
    """Unormalize a tensor that normalized by torchvision.transforms.Normalize

    Normalize subtracts mean and divides by std dev so to Unnormalize we need to
    multiply by the std dev and add the mean
    """

    def __init__(self, mean: List[float], std: List[float], inplace=False):
        """Initialize the unnormalize class

        Args:
            mean: the mean of the dataset for each channel
            std: the standard deviation of the dataset for each channel
        """
        assert len(mean) == len(std)
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor):
        """
        This code is largely based on: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L905
        Args:
            tensor: Tensor image of size (B, C, H, W) or (C, H, W) to be unnormalized.
        Returns:
            Tensor: Unormalized image.
        """

        if tensor.ndim < 3:
            raise ValueError(
                f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {tensor.size()}"
            )

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)

        if not self.inplace:
            tensor = tensor.clone()

        # Change shape to broadcast; i.e., for mean = [0.5, 0.4, 0.3] (3,)
        # becomes (3, 1, 1) to broadcast across spatial dimensions
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        # Modifies in place `_`
        tensor.mul_(std).add_(mean)
        # The normalize code -> t.sub_(m).div_(s)

        return tensor


class Compose:
    """Stores a list of transforms and applies them sequentially on the image and target label"""

    def __init__(self, transforms: list, policy: Optional[dict] = None):
        """Initalize the compose class

        Args:
            transforms: a list of torch transforms or custom transform classes
            policy: dictionary of parameters which dictates any special behavior for the transforms;
                    for example, rtdetrv2 stops certain transforms at specific epochs
        """
        self.transforms = transforms
        self.policy = policy

    def forward(self, *inputs: Any):
        """Selects and calls the appropriate forward method based on the policy

        Args:
           inputs: positional arguments to be passed to the appropriate forward method
        """
        return self._get_forward(self.policy["name"])(*inputs)

    def _get_forward(self, policy_name: str):
        """Selects the forward method to use"""
        forward_methods = {
            "default": self.default_forward,
            "stop_epoch": self.stop_epoch_forward,
        }
        return forward_methods[policy_name]

    def default_forward(self, image: PIL.Image, target: dict):
        """The standard behavior for the compose class

        Args:
            image: a PIL image to be augmented
            target: a dictionary of image annoations containing keys:
                        boxes: (XYXY: top_left, bottom_right) (num_objects, 4)
                        labels: class IDs for each object (num_objects,)
                        orig_size: the original image size before data augmentation
                        size: the current image size; updated later in the pipeline after augmentation

        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]

        cur_epoch = dataset.epoch
        policy_ops = self.policy["ops"]
        policy_epoch = self.policy["epoch"]

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                pass
            else:
                sample = transform(sample)

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
