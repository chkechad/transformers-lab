"""Feed forward neural network module."""

import numpy as np


class FeedForward:
    """Position-wise Feed Forward Network used in Transformer."""

    def __init__(self, d_model: int, d_ff: int):
        """Initialize the feed-forward module.

        Parameters
        ----------
        d_model : int
            Model dimensionality
        d_ff : int
            Hidden layer dimensionality
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Xavier-like init
        self.w1 = np.random.normal(size=(d_model, d_ff)) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)

        self.w2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(x, 0)

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
        hidden = x @ self.w1 + self.b1
        hidden = self.relu(hidden)

        output = hidden @ self.w2 + self.b2

        return output
