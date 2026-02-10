"""Compute self-attention."""
import numpy as np

from softmax import softmax


def self_attention(x: np.ndarray, d_k: int) -> np.ndarray:
    """Computes self-attention for a given input sequence.

    :param x: Input sequence of shape (seq_len, d_model)
    :param d_k: Dimensionality of the key vectors
    :return: Output of self-attention of shape (seq_len, d_model)
    """
    seq_len, d_model = x.shape

    # matrix projection
    w_q = np.random.randn(d_model, d_k)
    w_k = np.random.randn(d_model, d_k)
    w_v = np.random.randn(d_model, d_k)

    # Compute q, k, v
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    # Scaled dot-product
    scores = q @ k.T
    # Scaling
    scores = scores / np.sqrt(d_k)

    # calcul of softmax
    attention_weights = softmax(scores, axis=1)

    # Weighted sum of values
    output = attention_weights @ v

    return output
