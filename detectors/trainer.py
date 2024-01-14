import time
from typing import Any, Dict

import numpy as np
import torch

from detectors.utils import utils
from detectors.vocab import Vocab

optimizer_map = {"adam": torch.optim.Adam,
                 "adamw": torch.optim.AdamW,
                 "sgd": torch.optim.SGD}


class Trainer:
    """Trainer TODO: comment"""

    def __init__(self, corpus):
        """Constructor for the Trainer class

        Args:
            corpus: body of text to encode/decode
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device}")

        # Initialize training objects
        self.optimizer = 
        self.lr_scheduler

    def run(
        self,
        text: str,
        model: str,
        model_args: Dict[str, any],
        batch_size: int,
        block_size: int,
        learning_rate=1e-3,
        max_iters: int = 1000,
        eval_interval: int = 100,
        eval_iters: int = 1000,
    ):
        """Excecute the Runner class

        Args:
            text: The text to encode and decode
        """
        # New line for formatting output
        print()

        # Instantiate the Vocab class
        nlp = Vocab(text)

        encoded_text = nlp.encode(text)

        # Split encoded data into 90% train and 10% val
        train_split_n = int(0.9 * len(encoded_text))
        train_data = encoded_text[:train_split_n]
        val_data = encoded_text[train_split_n:]

        block_size = 8
        print(train_data[: block_size + 1])  # first 9 chars in the sequence

        x = train_data[:block_size]  # first 8 input chars
        y = train_data[
            1 : block_size + 1
        ]  # labels chars 1-9; offset by 1 to predict the next character
        for t in range(block_size):
            context = x[0 : t + 1]
            target = y[t]
            print(f"When the input is {context} the target: {target}")

        x_batch, y_batch = Vocab.get_batch(
            train_data, batch_size, block_size, self.device
        )

        model = model_map[model](
            **model_args,
            block_size=block_size,
            vocab_size=len(nlp),
        )
        model = model.to(self.device)
        logits, loss = model(x_batch, y_batch)

        # Create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        self.train(
            train_data,
            val_data,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            block_size=block_size,
            max_iters=max_iters,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
        )

        # Feed in token 0 to 'kick off' the generation; token 0 is the new line character in our vocab
        idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        # Generate a sequence of 100 tokens, grab the batch output with [0], and decode the predicted tokens
        print(nlp.decode(model.generate(idx, block_size, max_new_tokens=100)[0]))

        print(loss.item())

        # decoded_sentence = nlp.decode(encoded_sentence)

        # print(f"Encoded corpus: {encoded_sentence}")
        # print(f"Decoded corpus: {decoded_sentence}\n")

    def _train_one_epoch(self):
        pass

    # def train():

    def train(
        self,
        start_epoch,
        epochs,

    ):
        """ """
        print("Start training")
        start_time = time.time()
        #################### START HERE work on training loop, similar to detr paper#############
        for epoch in range(start_epoch, epochs):
            train_stats = self._train_one_epoch()
            lr_scheduler.step()

    @torch.no_grad()
    def estimate_loss(
        self, train_data, val_data, model, eval_iters, batch_size, block_size
    ) -> Dict[str, float]:
        """Estimate the loss of the train and val split

        Args:

        """
        out = {}
        model.eval()
        all_data = {"train": train_data, "val": val_data}
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = Vocab.get_batch(
                    all_data[split], batch_size, block_size, self.device
                )
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
