"""Feed forward neural network module."""

from collections.abc import Callable
from typing import cast

import numpy as np

from transformers_lab.weight import xavier_init


class FeedForward:
    """Position-wise Feed Forward Network used in Transformer."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        init_weight_fn: Callable[[int, int], np.ndarray] = xavier_init,
    ):
        """Initialize the feed-forward module.

        Parameters
        ----------
        d_model : int
            Model dimensionality
        d_ff : int
            Hidden layer dimensionality
        init_weight_fn : Callable[[int, int], np.ndarray]
            Weight initialization function. Defaults to xavier_init.
        """
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = init_weight_fn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)

        self.w2 = init_weight_fn(d_ff, d_model)
        self.b2 = np.zeros(d_model)

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return cast(np.ndarray, np.maximum(x, 0))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run forward pass.

        Parameters
        ----------
        x : np.ndarray
            Shape (seq_len, d_model)

        Returns:
        -------
        np.ndarray
            Shape (seq_len, d_model)
        """
        hidden: np.ndarray = cast(np.ndarray, x @ self.w1 + self.b1)  # (seq_len, d_ff)
        hidden = self.relu(hidden)  # (seq_len, d_ff)

        output: np.ndarray = cast(np.ndarray, hidden @ self.w2 + self.b2)  # (seq_len, d_model)

        return output
