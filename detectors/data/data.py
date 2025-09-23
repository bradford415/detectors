from typing import Optional, Sequence

import torch
from torch import Tensor


class NestedTensor:
    """Class to store the input data `tensors` and the binary `mask`

    The mask is True where padding exists and False where "real" values exist. The
    NestedTensor class was designed to handle different size images in a batch. All the images
    in the batch will be padded to the maximum height and width of the batch. The mask is used
    to tell the model and positional embeddings where this padding occurs; i.e., where the real
    pixels values and the padded pixel values occur
    """

    def __init__(self, tensors: Tensor, mask: Optional[Tensor]):
        """Initialize the NestedTensor

        Args:
            tensors: TODO (b, c, h, w)
            mask: binary mask which corresponds to each image in the batch where True is
                  a padded pixel and False is a "real" pixel (b, h, w)
        """
        self.tensors = tensors
        self.mask = mask
        if mask == "auto":
            self.mask = torch.zeros_like(tensors).to(tensors.device)
            if self.mask.dim() == 3:
                self.mask = self.mask.sum(0).to(bool)
            elif self.mask.dim() == 4:
                self.mask = self.mask.sum(1).to(bool)
            else:
                raise ValueError(
                    "tensors dim must be 3 or 4 but {}({})".format(
                        self.tensors.dim(), self.tensors.shape
                    )
                )

    def imgsize(self):
        res = []
        for i in range(self.tensors.shape[0]):
            mask = self.mask[i]
            maxH = (~mask).sum(0).max()
            maxW = (~mask).sum(1).max()
            res.append(torch.Tensor([maxH, maxW]))
        return res

    def to(self, device: torch.device) -> "NestedTensor":
        """Move the tensor and mask to the specified device

        Args:
            device: the device to move the tensors to
        """
        cast_tensor = self.tensors.to(device)
        mask = self.mask

        # move mask tensor to device if it exists
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def to_img_list_single(self, tensor, mask):
        assert tensor.dim() == 3, "dim of tensor should be 3 but {}".format(
            tensor.dim()
        )
        maxH = (~mask).sum(0).max()
        maxW = (~mask).sum(1).max()
        img = tensor[:, :maxH, :maxW]
        return img

    def to_img_list(self):
        """remove the padding and convert to img list

        Returns:
            [type]: [description]
        """
        if self.tensors.dim() == 3:
            return self.to_img_list_single(self.tensors, self.mask)
        else:
            res = []
            for i in range(self.tensors.shape[0]):
                tensor_i = self.tensors[i]
                mask_i = self.mask[i]
                res.append(self.to_img_list_single(tensor_i, mask_i))
            return res

    @property
    def device(self):
        return self.tensors.device

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        """Returns the shape of the tensors and masks"""
        return {"tensors.shape": self.tensors.shape, "mask.shape": self.mask.shape}

    @classmethod
    def from_tensor_list(cls, img_tensors: Sequence[Tensor]) -> "NestedTensor":
        """Creates a NestTensor object from a list of tensors; this creates padded images in the
        batch and a binary mask representing where the padding exists (True=Padding)

        This is intended to be called from the collate_fn (i.e., no batch dim for tensors); currently
        called in data.collate_functions.collate_fn_nested_tensor()

        Args:
            img_tensors: a sequence of image tensors where each element has dims (c, h, w)

        Returns:
            a NestedTensor
        """
        if isinstance(img_tensors, torch.Tensor):
            n_dims = img_tensors.ndim
            img_tensors = [img_tensors]
        if isinstance(img_tensors, list) or isinstance(img_tensors, tuple):
            n_dims = img_tensors[0].ndim
        if n_dims == 3:
            # TODO make it support different-sized images; this was from original detr code

            # find the max value for each dimension in the batch [c, h, w]
            max_size = cls._max_by_axis([list(img.shape) for img in img_tensors])

            # prepend a batch dimension [b, c, h, w] to the list
            batch_shape = [len(img_tensors)] + max_size

            b, c, h, w = batch_shape
            dtype = img_tensors[0].dtype
            device = img_tensors[0].device

            # Create a tensor of
            #   - 0s for the padded images (b, max_batch_c, max_batch_h, max_batch_w)
            #   - 1s for the padding_mask where 1=padding & 0=real_pix (b, max_batch_h, max_batch_w)
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

            # for each image in the batch, build a padded_image and a binary padding_mask
            for img, pad_img, pad_mask in zip(img_tensors, tensor, mask):
                # copy the real image into the top left corner of the padded image, leaving the right
                # and bot with 0 padded; NOTE: visually the padding will look gray due to normalization
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

                # set the spatial locations (h, w) of the padding_mask to 0
                pad_mask[: img.shape[1], : img.shape[2]] = False
        else:
            raise ValueError("tensors must have 3 dims (c, h, w)")

        return cls(tensor, mask)

    @staticmethod
    def _max_by_axis(img_sizes: list[list]) -> list[int]:
        """Create a list of the maximum value for each dimension in the batch [c, h, w];
        used to pad the images and create the padding mask

        Args:
            img_sizes: a list of image dimensions in the batch; cannot have a batch dimension

        Returns:
            a list of the maximum value for each dimension in the batch [c, h, w]
        """
        # Initalize maxes with the first tensor dimensions
        maxes = img_sizes[0]

        # loop through each dim for all image dimensions in the batch, starting w/ the 2nd image
        for img_dims in img_sizes[1:]:
            for index, dims in enumerate(img_dims):  # (c, h, w)

                # replace the value in `maxes` if the current image_dim is larger
                maxes[index] = max(maxes[index], dims)
        return maxes
