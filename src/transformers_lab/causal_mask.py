"""Create a causal mask for transformer attention."""

import numpy as np


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
    Step 1 — initialize to zeros:
    ┌─────────────┐
    │ 0  0  0  0   │
    │ 0  0  0  0   │
    │ 0  0  0  0   │
    │ 0  0  0  0   │
    └─────────────┘
    Step 2 — set upper triangle (k=1, diagonal excluded) to -inf:
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
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
