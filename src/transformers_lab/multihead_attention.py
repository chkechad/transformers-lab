"""Compute multi-head-attention."""

from typing import cast

import numpy as np

from transformers_lab.scaled_dot_product_attention import scaled_dot_product_attention


def multi_head_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    n_heads: int,
    x_cross: np.ndarray | None = None,
    mask: np.ndarray | None = None,
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

    x_cross : np.ndarray, optional
        Encoder output of shape (src_seq_len, d_model).
        If provided, keys and values come from x_cross (cross-attention).
        If None, keys and values come from x (self-attention).
    mask : np.ndarray, optional
        Mask of shape (seq_len, seq_len).
        Use make_causal_mask() for masked self-attention in the decoder.

    Returns:
    -------
    np.ndarray
        Output tensor of shape (seq_len, d_model)
    """
    kv_source: np.ndarray = x_cross if x_cross is not None else x

    heads: list[np.ndarray] = []

    for i in range(n_heads):
        q: np.ndarray = cast(np.ndarray, x @ w_q[i])  # (seq_len, d_k)
        k: np.ndarray = cast(np.ndarray, kv_source @ w_k[i])  # (seq_len, d_k)
        v: np.ndarray = cast(np.ndarray, kv_source @ w_v[i])  # (seq_len, d_k)

        head = scaled_dot_product_attention(q, k, v, mask=mask)  # (seq_len, d_k)
        heads.append(head)

    concat = np.concatenate(heads, axis=-1)  # (seq_len, d_model)

    return cast(np.ndarray, concat @ w_o)  # (seq_len, d_model)
