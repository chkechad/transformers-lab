"""Decoder: stack of N DecoderBlock layers."""

from collections.abc import Callable

import numpy as np

from transformers_lab.decoder_block import DecoderBlock
from transformers_lab.feed_forward import FeedForward
from transformers_lab.layer_norm import LayerNorm
from transformers_lab.weight import xavier_init


class Decoder:
    """Stack of N identical DecoderBlock layers.

    From "Attention is All You Need" (Vaswani et al., 2017):
        The decoder is composed of a stack of N=6 identical layers.

    Parameters
    ----------
    n_layers : int
        Number of stacked decoder blocks. Paper default: 6.
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
    >>> decoder = Decoder(n_layers=6, d_model=512, n_heads=8, d_hidden_ff=2048)
    >>> x = np.random.randn(10, 512)
    >>> enc_out = np.random.randn(12, 512)
    >>> out = decoder.forward(x, enc_out)
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
        """Initialize the Decoder stack."""
        self.layers: list[DecoderBlock] = [
            DecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_hidden_ff=d_hidden_ff,
                weight_fn=weight_fn,
                feed_forward_fn=feed_forward_fn,
                layer_norm_fn=layer_norm_fn,
            )
            for _ in range(n_layers)
        ]

    def forward(
        self,
        x: np.ndarray,
        encoder_output: np.ndarray,
    ) -> np.ndarray:
        """Forward pass through all decoder blocks.

        Parameters
        ----------
        x : np.ndarray
            Target sequence of shape (tgt_seq_len, d_model).
        encoder_output : np.ndarray
            Encoder output of shape (src_seq_len, d_model).
            Passed to every decoder block for cross-attention.

        Returns:
        -------
        np.ndarray
            Output tensor of shape (tgt_seq_len, d_model).
        """
        for layer in self.layers:
            x = layer.forward(x, encoder_output)
        return x
