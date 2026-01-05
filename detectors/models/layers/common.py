# Common pytorch layers used across models
import torch
from torch import nn
from torch.nn import functional as F

activation_map = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,  # silu is swish with beta=1 (essentially the same)
    "swish": nn.SiLU,
    None: nn.Identity,
}


class ConvNormLRelu(nn.Module):
    """Module which performs a convolution, batchnorm, and leakyReLU sequentially"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        leaky_slope: float = 0.1,
        conv_bias: bool = False,
    ):
        """Initialize the module layers

        Note: the default parameters do not downsample the feature maps

        Args:
            in_channels: Number of input channels to the Conv2D
            out_channels: Number of outpu tchannels in the feature map after the Conv2D
            kernel_size: Size of the Conv2D kernel
            stride: Stide of the ConvTranspose
            padding: Padding of the ConvTranspose
            leaky_slope: Negative slope of the leaky relu;
                         the default value is from here:
                         https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/1d621c8489e22c76ceb93bb2397ac6c8dfb5ceb7/pytorchyolo/models.py#L67
            conv_bias: Whether to use a bias in Conv2D; typically this is false if BatchNorm is the following layer
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=conv_bias,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)

    def forward(self, x):
        """Forward pass through the module

        Args:
            x: Input data
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class Upsample(nn.Module):
    """Upsample feature map to specific dimensions"""

    def __init__(self, scale_factor: float, mode: str = "nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class MLP(nn.Module):
    """Very simple MLP/FFN used by DINO"""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        """Initalize the MLP

        Args:
            input_dim: the number of input dimension to the MLP
            hidden_dim: the number of hidden dimensions for the MLP
            output_dim: the number of output dimensions for the last linear layer in the MLP
            num_layers: the number of linear layers in the MLP
        """
        super().__init__()
        self.num_layers = num_layers

        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        self.layers = nn.ModuleList()

        # for n layers, we'll have n-1 hidden_dim in the output of the layers
        hidden_dims = [hidden_dim] * (num_layers - 1)

        # Build the MLP with num_layers linear layers
        in_chs = [input_dim] + hidden_dims
        out_chs = hidden_dims + [output_dim]
        for in_ch, out_ch in zip(in_chs, out_chs):
            self.layers.append(nn.Linear(in_features=in_ch, out_features=out_ch))

    def forward(self, x):
        """Forward pass through the MLP

        Args:
            x: input tensors

        returns:
            TODO put shape
        """
        for index, layer in enumerate(self.layers):

            # ReLU activation after every linear layer output except the last one
            if index < self.num_layers - 1:
                x = layer(x)
                x = F.relu(x)
            else:
                x = layer(x)

        return x
