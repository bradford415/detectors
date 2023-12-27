from typing import Dict, Tuple, Union

import numpy as np
import torch


class Vocab:
    """Class which stores and controls the methods for our input vocabulary"""

    def __init__(self, full_text: str) -> None:
        """Constructor for the Vocab class

        Args:
            full_text: A list of words of the entire body of text to encode/decode
        """
        # A list of all unique words (vocab) in the full_text
        self.vocab = sorted(list(set(full_text)))

        self.word_to_integer = self._create_vocab_mapping()
        self.integer_to_word = self._create_vocab_mapping(reverse=True)
        self._display_information()

    def __len__(self):
        """Return the length of the vocab via: len(vocab_object)"""
        return len(self.vocab)

    def _create_vocab_mapping(
        self, reverse: bool = False
    ) -> Union[Dict[str, int], Dict[int, str]]:
        """Create the vocabulary (unique characters) of the full text by encoding unique
        words as integers.

        Args:
            reverse: Whether to reverse the dictionary mapping
        """

        if not reverse:
            return {char: index for index, char in enumerate(self.vocab)}
        else:
            return {index: char for index, char in enumerate(self.vocab)}

    def _display_information(self):
        print(f"Vocabulary of corpus: {''.join(self.vocab)}")
        print(f"Length of vocabulary: {len(self.vocab)}")

    def encode(self, sentence: str) -> torch.tensor:
        """Encoded a sentence to integers using the word_to_integer dict

        Args:
            sentence: A list of strings/words forming a sentence
        """
        return torch.tensor([self.word_to_integer[word] for word in sentence])

    def decode(self, encoded_sequence: torch.tensor) -> torch.tensor:
        """Decode an encoded sentence of integers

        Args:
            encoded_sequence: An encoded sequence encoded with the word_to_integer mapping
        """
        decoded_sequence = [
            self.integer_to_word[encoded_char.item()]
            for encoded_char in encoded_sequence
        ]
        return "".join(decoded_sequence)

    @staticmethod
    def get_batch(
        encoded_input: torch.tensor,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Grab a batch  train data and labels of size (batch_size, block_size)

        Args:
            encoded_input: encoded train or val data
            batch_size: number of sequences to process at once
            block_size: length of sequence
        """
        # Generate random tensor of ints; ix: (batch_size,)
        ix = torch.randint(len(encoded_input) - block_size, (batch_size,))

        # Use each random int to take a chunk of block_size from the input data and create a batch (batch_size, block_size)
        # x = input_data, y = labels
        x = torch.stack([encoded_input[i : i + block_size] for i in ix])
        y = torch.stack(
            [encoded_input[i + 1 : i + block_size + 1] for i in ix]
        )  # +1 for predicted the next token

        x = x.to(device)
        y = y.to(device)
        return x, y
