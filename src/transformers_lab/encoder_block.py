"""Encoder block for Transformer models."""

from collections.abc import Callable

import numpy as np

from transformers_lab.feed_forward import FeedForward
from transformers_lab.layer_norm import LayerNorm
from transformers_lab.multihead_attention import multi_head_attention
from transformers_lab.weight import xavier_init


class EncoderBlock:
    """One Transformer encoder block.

    Applies the following sequence (Vaswani et al., 2017):
        x -> MultiHeadAttention -> Add & Norm -> FeedForward -> Add & Norm

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    n_heads : int
        Number of attention heads. Must divide d_model evenly.
    d_hidden_ff : int
        Hidden dimensionality of the feed-forward network.
    weight_fn : Callable[[int, int, int | None], np.ndarray]
        Weight initialization function. Signature: (n1, n2, n_heads=None).
        Defaults to xavier_init.
    feed_forward_fn : Callable[[int, int], FeedForward]
        Factory for the feed-forward sub-layer. Defaults to FeedForward.
    layer_norm_fn : Callable[[int], LayerNorm]
        Factory for layer normalization. Defaults to LayerNorm.

    Raises:
    ------
    ValueError
        If d_model is not divisible by n_heads.

    Examples:
    --------
    >>> block = EncoderBlock(d_model=512, n_heads=8, d_hidden_ff=2048)
    >>> x = np.random.randn(10, 512)
    >>> out = block.forward(x)
    >>> out.shape
    (10, 512)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_hidden_ff: int,
        weight_fn: Callable[..., np.ndarray] = xavier_init,
        feed_forward_fn: Callable[[int, int], FeedForward] = FeedForward,
        layer_norm_fn: Callable[[int], LayerNorm] = LayerNorm,
    ) -> None:
        """Initialize the EncoderBlock.

        Parameters
        ----------
        d_model : int
            Model dimensionality.
        n_heads : int
            Number of attention heads. Must divide d_model evenly.
        d_hidden_ff : int
            Hidden dimensionality of the feed-forward network.
        weight_fn : Callable[..., np.ndarray]
            Weight initialization function. Defaults to xavier_init.
        feed_forward_fn : Callable[[int, int], FeedForward]
            Factory for the feed-forward sub-layer. Defaults to FeedForward.
        layer_norm_fn : Callable[[int], LayerNorm]
            Factory for layer normalization. Defaults to LayerNorm.
        """
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}). "
                f"Got d_model % n_heads = {d_model % n_heads}."
            )

        self.n_heads: int = n_heads
        self.d_k: int = d_model // n_heads

        self.w_q: np.ndarray = weight_fn(d_model, self.d_k, n_heads)
        self.w_k: np.ndarray = weight_fn(d_model, self.d_k, n_heads)
        self.w_v: np.ndarray = weight_fn(d_model, self.d_k, n_heads)

        self.feed_forward_fn = feed_forward_fn
        self.layer_norm_fn = layer_norm_fn

        self.w_o: np.ndarray = weight_fn(d_model, d_model, None)

        # Sub-layers
        self.ffn: FeedForward = self.feed_forward_fn(d_model, d_hidden_ff)
        self.norm1: LayerNorm = layer_norm_fn(d_model)
        self.norm2: LayerNorm = layer_norm_fn(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through one encoder block.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (seq_len, d_model).

        Returns:
        -------
        np.ndarray
            Output tensor of shape (seq_len, d_model).
        """
        # Sub-layer 1: Multi-Head Self-Attention + Add & Norm
        attn_out: np.ndarray = multi_head_attention(
            x,
            w_q=self.w_q,
            w_k=self.w_k,
            w_v=self.w_v,
            w_o=self.w_o,
            n_heads=self.n_heads,
        )
        x = self.norm1(x + attn_out)

        # Sub-layer 2: Feed-Forward + Add & Norm
        ffn_out: np.ndarray = self.ffn.forward(x)
        return self.norm2(x + ffn_out)
