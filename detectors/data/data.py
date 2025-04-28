from typing import Optional

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
        return {"tensors.shape": self.tensors.shape, "mask.shape": self.mask.shape}

    @classmethod
    def from_tensor_list(cls, tensor_list: list[Tensor]) -> "NestedTensor":
        """TODO

        Args:
            tensor_list:
        """
        if tensor_list[0].ndim == 3:
            if torchvision._is_tracing():
                # nested_tensor_from_tensor_list() does not export well to ONNX
                # call _onnx_nested_tensor_from_tensor_list() instead
                return _onnx_nested_tensor_from_tensor_list(tensor_list)

            # TODO make it support different-sized images
            max_size = _max_by_axis([list(img.shape) for img in tensor_list])
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = [len(tensor_list)] + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        else:
            raise ValueError("not supported")
        return cls(tensor, mask)
