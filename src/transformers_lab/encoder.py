"""Encoder: stack of N EncoderBlock layers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from transformers_lab.encoder_block import EncoderBlock
from transformers_lab.feed_forward import FeedForward
from transformers_lab.layer_norm import LayerNorm
from transformers_lab.weight import xavier_init


class Encoder:
    """Stack of N identical EncoderBlock layers.

    From "Attention is All You Need" (Vaswani et al., 2017):
        The encoder is composed of a stack of N=6 identical layers.

    Parameters
    ----------
    n_layers : int
        Number of stacked encoder blocks. Paper default: 6.
    d_model : int
        Model dimensionality. Paper default: 512.
    n_heads : int
        Number of attention heads. Paper default: 8.
    d_hidden_ff : int
        Hidden dimensionality of the feed-forward network. Paper default: 2048.
    weight_fn : Callable[..., np.ndarray]
        Weight initialization function. Defaults to xavier_init.
    feed_forward_fn : Callable[[int, int], FeedForward]
        Factory for the feed-forward sub-layer. Defaults to FeedForward.
    layer_norm_fn : Callable[[int], LayerNorm]
        Factory for layer normalization. Defaults to LayerNorm.

    Examples:
    --------
    >>> encoder = Encoder(n_layers=6, d_model=512, n_heads=8, d_hidden_ff=2048)
    >>> x = np.random.randn(10, 512)
    >>> out = encoder.forward(x)
    >>> out.shape
    (10, 512)
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_hidden_ff: int,
        weight_fn: Callable[..., np.ndarray] = xavier_init,
        feed_forward_fn: Callable[[int, int], FeedForward] = FeedForward,
        layer_norm_fn: Callable[[int], LayerNorm] = LayerNorm,
    ) -> None:
        """Initialize the Encoder stack."""
        self.layers: list[EncoderBlock] = [
            EncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_hidden_ff=d_hidden_ff,
                weight_fn=weight_fn,
                feed_forward_fn=feed_forward_fn,
                layer_norm_fn=layer_norm_fn,
            )
            for _ in range(n_layers)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all encoder blocks.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (seq_len, d_model).

        Returns:
        -------
        np.ndarray
            Output tensor of shape (seq_len, d_model).
        """
        for layer in self.layers:
            x = layer.forward(x)  # (seq_len, d_model) → (seq_len, d_model)
        return x
