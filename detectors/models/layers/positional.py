import math

import torch
from torch import nn

from detectors.data import NestedTensor


class PositionEmbeddingSineHW(nn.Module):
    """Create 2D positional embeddings for the images patches/tokens passed into the encoder; the images patches
    correspond to the feature map of the backbone network

    This implements the formulas:
        PE(pos, 2i) = sin(pos / temperature_h^(2i/dim))    and
        PE(pos, 2i+1) = cos(pos / temperature_w^(2i/dim))

    A lower temperature means positions change more rapidly while a higher temperature means positions change more slowly;
    intuitively this makes sense because a lower temperature means the fraction inside sin & cos will be higher and thus the sine and cosine
    functions will complete cycles quicker i.e., sin(pos/low_number) vs sin(pos/high_number)

    For example, if the output of the backbone is 256x16x16 (c, h, w) then each patch will have a
    positional embedding of 128x1x1; the overall shape of the positional embedding will be (b, 128, 16, 16);
    128 is the embedding dimension for each position; typically, this is half of the token embedding dim (the input to the encoder)

    This was introduced in DETR; DINO modifies this to control the temperature parameter for the height and width
    but in their config files temperature_h = temperature_w  so I don't think there's much of a difference
    """

    def __init__(
        self,
        num_pos_feats=128,  # TODO; maybe find a better name for this
        temperature_h=10000,
        temperature_w=10000,
        normalize=False,
        scale=None,
    ):
        """Initialize the Postional Embedding Module

        Args:
            num_pos_feats: embedding size of each sine and cosine positional embedding; the output embedding size
                           will be num_pos_feats*2 because we alternate sine and cos; this dimension will match
                           the dimension the feature_maps get projected to in models.dino.DINO.__init__()
                           (i.e., num_post_featrs*2) this is typically half of the token embedding dim
                           (the input to the encoder)
            temperature_h: temperature for height; a lower temperature means positions change more rapidly
            temperature_w: temperature for width; a lower temperature means positions change more rapidly
            normalize: whether to row normalize and col normalize the positional coordinates (right before creating embeddings)
                       from [0, 2pi)
            scale: goes hand-in-hand with normalize; the values are first normalized to [0, 1) and then multiplied by scale [0, 2pi)
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature_h = temperature_h
        self.temperature_w = temperature_w
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        """Compute the 2D positional embeddings

        Args:
            tensor_list: NestedTensor containing the
                            image_patches/feature_map: (b, c, h, w) the output of the backbone network (c is the embedding dim)
                            masks: (b, h, w) a boolean mask indicating valid pixels (False) and padding (True)

        Returns:
            positional embeddings of shape (b, num_pos_feats, h, w);
        """
        image_patches = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None

        # Invert the mask so that the padding is False and valid pixels are True
        not_mask = ~mask

        # The goal of cumsum is to create continuous, meaningful positional coords for x and y axes,
        # even when there are masks or irregular shapes in the image;
        #   - this essentially creates a meshgrid indicating the location of valid pixels which will be used for the
        #     pos variable in the PE equation;
        #   - if a value is padded (False in the not_mask) it will have the same value as the previous location;
        # NOTE: padding should only be around the image edges so this technique ensures that the valid position
        #       values start from 0 and skip the padded areas; e.g., padding at the start of the image will be all 0s
        #       until the valid regions and padding after the valid positions will repeat the last valid position value
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # (b, h, w)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # Normalize between [0, 2pi]
        if self.normalize:
            eps = 1e-6
            y_embed = (
                y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            )  # column normalize
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # row normalize

        # Calculate the arguments [pos / temperature^(2i/dim)] to sin/cos of the PE equation where
        # i is the index of the emb_dim and dim is total dimensions;
        # NOTE: we perform integer division (dim_t // 2) in this code to group the dimensions into pairs
        #       i.e., from [0, 1, 2, ..., 127] to [0, 0, 1, 1, ..., 63, 63], this allows us to
        #       use the same frequency for sin & cos in the final positional embedding;
        #       i.e., dim_tx[0] & dim_tx[1] share a frequency, dim_tx[2] & dim_tx[3] share a different one;
        #       I believe this is slightly different than the original transformer paper
        dim_tx = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=image_patches.device
        )
        dim_tx = self.temperature_w ** (
            2 * (dim_tx // 2) / self.num_pos_feats
        )  # denominator

        # (b, h, w, 1) / (num_pos_feats,) = (b, h, w, num_pos_feats);  this essentially
        # divides each spatial locations in x_embed num_pos_feats times to embed that location;
        # this computes the full argument to sin/cos in the PE equation
        pos_x = x_embed[:, :, :, None] / dim_tx

        # Same as above but for the y positions
        dim_ty = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=image_patches.device
        )
        dim_ty = self.temperature_h ** (2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        assert len(dim_tx) == self.num_pos_feats and len(dim_ty) == self.num_pos_feats

        # Form the final positonal embeddings by alternating sin to the even positions and cos to the odd positions
        # (b, h, w, num_pos_feats) -> (b, h, w, num_pos_feats/2, 2) -> (b, h, w, num_pos_feats)
        # NOTE: the dims (..., num_pos_feats/2, 2) stacks the sin values in the first column and the cos values in the second column
        #       therefore, when you flatten the last two dimensions, the sin and cos values alternate
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        # Concatenate the y and x positional embeddings and permute the dimensions to match the encoder input; TODO verify this enocder input
        # each image_patch now has a positional embedding of (num_pos_feats,) and the encoder expects the input to be (b, num_pos_feats, h, w)
        # (b, h, w, num_pos_feats/2, 2) -> (b, h, w, num_pos_feats) -> (b, num_pos_feats, h, w)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


def build_positional_encodings():
    """Builds 2d posiitional encodings for the images patches/tokens passed into the encoder;
    the images patches correspond to the feature map of the backbone network

    Args:
        num_pos_feats:
        
    """
    raise NotImplementedError


def gen_sineembed_for_position(pos_tensor):
    """Generate fixed sinusoidal embeddings very similar to PositionEmbeddingSineHW()
    except the temperature is fixed at 10000 like in the original DETR;
    since the temperature is the same we can use a single dim_t for the frequencies,
    we don't need dim_tx and dim_ty as PositionEmbeddingSineHW()

    Args:
        pos_tensor: TODO (num_queries, b, 4) where 4 = (cx, cy, w, h)

    Returns:
        the sinusoidal embeddings for the reference points (pos_tensor); shape (num_queries, b, 512)
        where 512 = 128*4 so each segment of 128 is x,y,w,h respectively
    """
    # Full unit circle value
    scale = 2 * math.pi

    # Calculate the arguments [pos / temperature^(2i/dim)] to sin/cos of the PE equation where
    # i is the index of the emb_dim and dim is total dimensions;
    # NOTE: we perform integer division (dim_t // 2) in this code to group the dimensions into pairs
    #       i.e., from [0, 1, 2, ..., 127] to [0, 0, 1, 1, ..., 63, 63], this allows us to
    #       use the same frequency for sin & cos in the final positional embedding;
    #       i.e., dim_tx[0] & dim_tx[1] share a frequency, dim_tx[2] & dim_tx[3] share a different one;
    #       I believe this is slightly different than the original transformer paper
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)

    # Scale the pos_tensor (reference_points) by 2pi; 0 = x, 1 = y
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale

    # this broadcasts to divide each ref_point in x_embed for each num_pos_feats to
    # embed that location; (num_queries, b, 1) / (num_pos_feats,) = (num_queries, b, num_pos_feats);
    # e.g., x_embed[:, :, 0] / dim_t[0], x_embed[:, :, 1] / dim_t[1] ..., x_embed[:, :, 127] / dim_t[127]
    # this computes the full argument to sin/cos in the PE equation
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t

    # Form the final positonal embeddings by alternating sin to the even positions and cos to the odd positions
    # (num_queries, b, num_pos_feats) -> (num_queries, b, num_pos_feats/2, 2) -> (num_queries, b, num_pos_feats)
    # NOTE: the dims (..., num_pos_feats/2, 2) stacks the sin values in the first column and the cos values in the second column
    #       therefore, when you flatten the last two dimensions, the sin and cos values alternate
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)

    if pos_tensor.size(-1) == 2:  # skipped
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        # Extract the width and scale [0, 2pi]
        w_embed = pos_tensor[:, :, 2] * scale

        # divide by the frequencies (broadcasted)
        # (num_queries, b, 1) / (num_pos_feats) = (num_queries, b, num_pos_feats)
        pos_w = w_embed[:, :, None] / dim_t

        # form the final positonal embeddings by alternating sin & cos (even & odd) just
        # as we did for x/y
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        # repeat the above but for the h
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        # combine all the positional embeddings into a single tensor
        # (num_queries, b, hidden_dim * 2) where hidden_dim * 2 = 128*4 so each segment of
        # 128 is x,y,w,h respectively
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos
