"""Softmax is used to!

- transform arbitrary real-valued scores into probabilities
- ensure all values are positive
- ensure the probabilities sum to 1
- normalize scores along a specified axis

In Transformers, softmax is used to convert
similarity scores (attention scores) into attention weights.

"""

from typing import cast

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute the softmax function in a numerically stable way.

    The softmax function transforms arbitrary real-valued scores
    into positive probabilities that sum to 1 along the specified axis.

    In Transformers, it is commonly used to convert attention
    similarity scores into attention weights.

    Parameters
    ----------
    x : np.ndarray
        Input array containing the scores.
    axis : int, optional
        Axis along which the normalization is applied. Defaults to the last axis.

    Returns:
    -------
    np.ndarray
        Array of the same shape as x containing the normalized probabilities.
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return cast(np.ndarray, exp_x / np.sum(exp_x, axis=axis, keepdims=True))
