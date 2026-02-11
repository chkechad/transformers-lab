"""Positional encoding."""

import numpy as np


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Compute sinusoidal positional encoding.

    Parameters
    ----------
    seq_len : int
        Length of the input sequence

    d_model : int
        Dimensionality of the model

    Returns:
    -------
    np.ndarray
        Positional encoding matrix of shape (seq_len, d_model)
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe
