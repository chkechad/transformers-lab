"""Compute self-attention."""

import numpy as np

from transformers_lab.scaled_dot_product_attention import scaled_dot_product_attention


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
