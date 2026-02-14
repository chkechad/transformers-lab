"""Compute scaled_dot_product_attention."""

import numpy as np

from transformers_lab.softmax import softmax


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute scaled dot-product attention.

    Parameters
    ----------
    q : np.ndarray
        Query matrix of shape (seq_len, d_k)

    k : np.ndarray
        Key matrix of shape (seq_len, d_k)

    v : np.ndarray
        Value matrix of shape (seq_len, d_k)
    mask: np.ndarray or None
        Mask to apply during attention computation. Must be of shape (seq_len, seq_len)

    Returns:
    -------
    np.ndarray
        Output matrix of shape (seq_len, d_k)
    """
    d_k = q.shape[-1]
    scores = q @ k.T
    scores = scores / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights: np.ndarray = softmax(scores, axis=-1)
    return weights @ v
