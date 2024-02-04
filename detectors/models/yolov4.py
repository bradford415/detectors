import torch
import torch.nn as nn
from torch.nn import functional as F

 ####################### START HERE and LOOK THROUGH PYTORCH MODEL yolv4 github ###################
class YoloV4(nn.Module):
    def __init__(
        self,
    ):
        """
        Args:

        """
        super().__init__()

    def forward(self, x):
        """Forward pass through the model

        Args:

        """
        B, T = idx.shape
        # idx and targets are both (batch, time_steps) tensor of integers
        token_embeddings = self.token_embedding_table(
            idx
        )  # (batch, time_steps, n_embed)

        # Create positional embeddings; arange allows you to specify the order [0, 1, ..., T-1]
        postional_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)

        # Encode the input as token plus positional embeddings, then compute self-attention
        x = token_embeddings + postional_embeddings  # (B, T, C)
        # x = self.decoder_blocks(x)  # (B, T, C)
        # x = self.ffwd(x)  # (B, T, C)
        x = self.decoder_blocks(x)
        # Finally, predict the next word
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshape logits as cross_entropy expects
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, block_size, max_new_tokens):
        """Generate a sequence from the starting context idx

        Args:
            idx: (B, T) tensor as the starting context for the sequence; idx is fed into the model as context
            max_new_tokens: number of predicted tokens to generate
            block_size: sequence length used for training; also represents the positional encoding size
        """

        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens (So you don't index positional embeddings outside its index)
            # Ex: If idx 80 tokens long, use tokens [80-block_size, 80]
            idx_cropped = idx[:, -block_size:]

            # Get the predictions
            logits, loss = self(
                idx_cropped
            )  # self(idx) is like using model(idx); it goes to the forward() function

            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)

            # Apply softmax to get probabilities
            # This line is a little confusing because its generating probabilities along the embed_dim
            # rather than the number vocab dim
            # I think it means based soley on this token (w/ embed_dim 65), what is the probability of the next token
            # it's not necessaryly doing a classification task on all the tokens
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Sample from the distribution to get the next index prediction
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
