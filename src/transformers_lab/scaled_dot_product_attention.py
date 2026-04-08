"""Compute scaled_dot_product_attention."""

from typing import cast

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
    scores: np.ndarray = cast(np.ndarray, q @ k.T / np.sqrt(d_k))  # (seq_len, seq_len)
    if mask is not None:
        scores = scores + mask  # (seq_len, seq_len)
    weights: np.ndarray = softmax(scores, axis=-1)  # (seq_len, seq_len)
    return cast(np.ndarray, weights @ v)  # (seq_len, d_k)
