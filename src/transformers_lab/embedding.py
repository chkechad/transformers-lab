"""Embedding layer for mapping token IDs to dense vectors."""

import numpy as np


class Embedding:
    """Embedding layer for mapping token IDs to dense vectors."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """Initialize the embedding layer.

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary (number of unique tokens).
        d_model : int
            Dimensionality of the embedding vectors.
        """
        self.weights = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
        self.d_model = d_model

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """Map token IDs to scaled embedding vectors.

        Parameters
        ----------
        token_ids : np.ndarray
            Token indices of shape (seq_len,) or (batch_size, seq_len).

        Returns:
        -------
        np.ndarray
            Embedding vectors of shape (*token_ids.shape, d_model).
        """
        return self.weights[token_ids] * np.sqrt(self.d_model)
