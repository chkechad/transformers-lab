"""Compute self-attention."""

import numpy as np

from transformers_lab.softmax import softmax


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
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

    Returns:
    -------
    np.ndarray
        Output matrix of shape (seq_len, d_k)
    """
    d_k = q.shape[-1]
    scores = q @ k.T
    scores = scores / np.sqrt(d_k)
    weights: np.ndarray = softmax(scores, axis=-1)
    return weights @ v


def self_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
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

    Returns:
    -------
    np.ndarray
        Output of shape (seq_len, d_k)
    """
    q: np.ndarray = x @ w_q
    k: np.ndarray = x @ w_k
    v: np.ndarray = x @ w_v

    return scaled_dot_product_attention(q, k, v)
