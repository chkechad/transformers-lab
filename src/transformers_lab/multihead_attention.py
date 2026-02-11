"""Compute multi-head-attention."""

import numpy as np

from transformers_lab.self_attention import scaled_dot_product_attention


def multi_head_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    n_heads: int,
) -> np.ndarray:
    """Computes multi-head attention.

    Parameters
    ----------
    x : np.ndarray
        Input tensor of shape (seq_len, d_model)

    w_q : np.ndarray
        Query weights of shape (n_heads, d_model, d_k)

    w_k : np.ndarray
        Key weights of shape (n_heads, d_model, d_k)

    w_v : np.ndarray
        Value weights of shape (n_heads, d_model, d_k)

    w_o : np.ndarray
        Output projection matrix of shape (d_model, d_model)

    n_heads : int
        Number of attention heads

    Returns:
    -------
    np.ndarray
        Output tensor of shape (seq_len, d_model)
    """
    heads: list[np.ndarray] = []

    for i in range(n_heads):
        q = x @ w_q[i]
        k = x @ w_k[i]
        v = x @ w_v[i]

        head = scaled_dot_product_attention(q, k, v)
        heads.append(head)

    concat = np.concatenate(heads, axis=-1)

    return concat @ w_o
