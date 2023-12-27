import torch
from torch import nn
from torch.nn import functional as F


class DecoderBlock(nn.Module):
    """Transformer decoder block without cross attention"""

    def __init__(self, n_heads, n_embed, block_size, inner_dim_multiplier):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, n_embed, head_size, block_size)
        self.ffwd = FeedForward(n_embed, inner_dim_multiplier)

    def forward(self, x):
        # compute masked self-attention and the feed-forward block
        # add the residual after each computation
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, n_embed, head_size, block_size):
        """Initialize the MHA module

        Args:
            num_heads: number of attention heads to run in parallel
            n_embed: embedding size of the tokens
            head_size: dimension size to project the k, q, v vectors to in a linear layer (head_size/num_heads)
            block_size: sequence length
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embed, head_size, block_size) for _ in range(num_heads)]
        )
        self.projection = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        # Loop through the list of attention heads, and concatenate their output along the C dimension
        x = torch.cat([h(x) for h in self.heads], dim=-1)

        # Projection at the end of the MHA
        x = self.projection(x)
        return x


class Head(nn.Module):
    """One head of masked self-attention"""

    def __init__(self, n_embed, head_size, block_size):
        """Initialize one SA head

        Args:
            n_embed: the token embedding size
            head_size: the size to project the q, k, v vecotrs (input to Head() is head_size//n_heads)
            block_size: sequence length
        """
        super().__init__()

        # Create linear layers to project input to higher dimension; typically biases are not used
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # Create "tril" variable. Since tril is not a parameter, so in PyTorch naming
        # conventions we need to create a register_buffer if we want to
        # assign it to the module
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        # Compute attention scores ("affinities")
        weights = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, C ) @ (B, C, T) -> (B, T, T)
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        weights = F.softmax(weights, dim=-1)  # (B, T ,T)

        # Perform a weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = weights @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity. Note, the residual
    was included in MultiHeadAttention
    """

    def __init__(self, n_embed, inner_dim_multiplier):
        """

        Args:
            inner_dim_multiplier: how much to multiply the inner dimension of the FFN by,
                                  specified in section 3.3 dff/dmodel "Attention is all you need"
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, inner_dim_multiplier * n_embed),
            nn.ReLU(),
            nn.Linear(inner_dim_multiplier * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)
