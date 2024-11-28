# Common pytorch layers used across models
from torch import nn


class ConvNormLRelu(nn.Module):
    """Module which performs a convolution, batchnorm, and leakyReLU sequentially"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        leaky_slope: float = 0.01,
        conv_bias: bool = False,
    ):
        """Initialize the module layers
        
        Note: the default parameters do not downsample the feature maps

        Args:
            in_channels: Number of input channels to the Conv2D
            out_channels: Number of output channels in the feature map after the Conv2D
            kernel_size: Size of the Conv2D kernel
            stride: Stide of the ConvTranspose
            padding: Padding of the ConvTranspose
            leaky_slope: Negative slope of the leaky relu; the default parameter is the same as the pytorch default
            conv_bias: Whether to use a bias in Conv2D; typically this is false if BatchNorm is the following layer
        """
        super().__init__()
        _conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=conv_bias,
        )
        _bn = nn.BatchNorm2d(num_features=out_channels)
        _leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)
        
        self.sequential = nn.Sequential(_conv, _bn, _leaky_relu)

    def forward(self, x):
        """Forward pass through the module

        Args:
            x: Input data
        """
        x = self.sequential(x)
        return x