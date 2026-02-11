"""layer normalisation implementation."""

import numpy as np


class LayerNorm:
    """Layer normalization implementation."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """Initialize the LayerNorm module.

        Parameters
        ----------
        d_model : int
            Dimensionality of the input.
        eps : float, optional
            Value added to the denominator for numerical stability.
            Default is 1e-6.

        Attributes
        ----------
        gamma : np.ndarray
            Scale parameter of shape (d_model,).
        beta : np.ndarray
            Shift parameter of shape (d_model,).
        eps : float
            Numerical stability constant.
        """
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (..., d_model).

        Returns
        -------
        np.ndarray
            Normalized tensor with the same shape as input.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta
