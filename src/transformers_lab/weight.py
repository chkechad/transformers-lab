"""Weight init."""

import numpy as np


def xavier_init(n1: int, n2: int, n_heads: int | None = None) -> np.ndarray:
    """Xavier Glorot initialization for weight matrices.

    Parameters
    ----------
    n1 : int
        Input dimension.
    n2 : int
        Output dimension.
    n_heads : int, optional
        If provided, returns shape (n_heads, n1, n2).
        If None, returns shape (n1, n2).

    Returns:
    -------
    np.ndarray
        Shape (n_heads, n1, n2) or (n1, n2).
    """
    std = np.sqrt(2.0 / (n1 + n2))
    if n_heads is not None:
        return np.random.randn(n_heads, n1, n2) * std
    return np.random.randn(n1, n2) * std
