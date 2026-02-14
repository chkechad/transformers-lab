"""Compute self-attention."""

import numpy as np

from transformers_lab.softmax import softmax


def causal_mask(seq_len: int) -> np.ndarray:
    """Create a causal (look-ahead) mask.

    Prevents each position from attending to future positions.
    Upper triangle is -inf, diagonal and below are 0.

    Parameters
    ----------
    seq_len : int
        Length of the sequence.

    Returns:
    -------
    np.ndarray
        Mask of shape (seq_len, seq_len) with 0s and -inf.

    Example:
    -------
    k=1 (diagonale exclue)
    ┌─────────────┐
    │ 0  0  0  0   │
    │ 0  0  0  0   │
    │ 0  0  0  0   │
    │ 0  0  0  0   │
    └─────────────┘
    mask = np.triu(np.ones((4, 4)), k=1) * -np.inf
    ┌──────────────────────┐
    │  0   -inf  -inf  -inf  │
    │  0     0   -inf  -inf  │
    │  0     0     0   -inf  │
    │  0     0     0     0   │
    └──────────────────────┘
    """
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf  # triangular upper matrix with 1s above the diagonal
    return mask


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


def self_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute single-head self-attention.

    Parameters
    ----------
    x : np.ndarray
        Input embeddings of shape (seq_len, d_model)

    w_q : np.ndarray
        Query projection matrix of shape (d_model, d_k)

    w_k : np.ndarray
        Key projection matrix of shape (d_model, d_k)

    w_v : np.ndarray
        Value projection matrix of shape (d_model, d_k)

    mask : np.ndarray, optional
        Mask of shape (seq_len, seq_len). Defaults to None.

    Returns:
    -------
    np.ndarray
        Output of shape (seq_len, d_k)
    """
    q: np.ndarray = x @ w_q
    k: np.ndarray = x @ w_k
    v: np.ndarray = x @ w_v

    return scaled_dot_product_attention(q, k, v, mask=mask)
