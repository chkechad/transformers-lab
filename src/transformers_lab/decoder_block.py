"""Decoder block for Transformer models."""

from collections.abc import Callable

import numpy as np

from transformers_lab.causal_mask import causal_mask
from transformers_lab.feed_forward import FeedForward
from transformers_lab.layer_norm import LayerNorm
from transformers_lab.multihead_attention import multi_head_attention
from transformers_lab.weight import xavier_init


class DecoderBlock:
    """One Transformer decoder block.

    Applies the following sequence (Vaswani et al., 2017):
        x -> Masked MultiHeadAttention -> Add & Norm
          -> Cross MultiHeadAttention  -> Add & Norm
          -> FeedForward               -> Add & Norm

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
    >>> block = DecoderBlock(d_model=512, n_heads=8, d_hidden_ff=2048)
    >>> x = np.random.randn(10, 512)
    >>> enc_out = np.random.randn(12, 512)
    >>> out = block.forward(x, enc_out)
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
        causal_mask: Callable[[int], np.ndarray] = causal_mask,
    ) -> None:
        """Initialize the DecoderBlock.

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
        causal_mask : Callable[[int], causal_mask]
            Factory for causal_mask. Defaults to causal_mask.
        """
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}). "
                f"Got d_model % n_heads = {d_model % n_heads}."
            )

        self.n_heads: int = n_heads
        self.d_k: int = d_model // n_heads

        self.weight_fn = weight_fn

        # --- Masked Self-Attention weights
        self.w_q1: np.ndarray = self.weight_fn(d_model, self.d_k, n_heads)
        self.w_k1: np.ndarray = self.weight_fn(d_model, self.d_k, n_heads)
        self.w_v1: np.ndarray = self.weight_fn(d_model, self.d_k, n_heads)
        self.w_o1: np.ndarray = self.weight_fn(d_model, d_model)

        # --- Cross-Attention weights
        self.w_q2: np.ndarray = self.weight_fn(d_model, self.d_k, n_heads)
        self.w_k2: np.ndarray = self.weight_fn(d_model, self.d_k, n_heads)
        self.w_v2: np.ndarray = self.weight_fn(d_model, self.d_k, n_heads)
        self.w_o2: np.ndarray = self.weight_fn(d_model, d_model)

        self.feed_forward_fn = feed_forward_fn
        self.layer_norm_fn = layer_norm_fn

        # Sub-layers
        self.ffn: FeedForward = self.feed_forward_fn(d_model, d_hidden_ff)
        self.norm1: LayerNorm = layer_norm_fn(d_model)  # after masked self-attention
        self.norm2: LayerNorm = layer_norm_fn(d_model)  # after cross attention
        self.norm3: LayerNorm = layer_norm_fn(d_model)  # after FFN

        self.causal_mask = causal_mask

    def forward(self, x: np.ndarray, encoder_output: np.ndarray) -> np.ndarray:
        """Forward pass through one encoder block.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (seq_len, d_model).

        encoder_output : np.ndarray
            Encoder output of shape (src_seq_len, d_model).
            Used as keys and values in cross-attention.

        Returns:
        -------
        np.ndarray
            Output tensor of shape (seq_len, d_model).
        """
        tgt_seq_len = x.shape[0]
        causal_mask = self.causal_mask(tgt_seq_len)
        # Sub-layer 1: Multi-Head Self-Attention + Add & Norm
        attn1_out: np.ndarray = multi_head_attention(
            x,
            w_q=self.w_q1,
            w_k=self.w_k1,
            w_v=self.w_v1,
            w_o=self.w_o1,
            n_heads=self.n_heads,
            mask=causal_mask,
        )
        x = self.norm1(x + attn1_out)
        # Sub-layer 2: Cross-Attention + Add & Norm
        attn2_out: np.ndarray = multi_head_attention(
            x,
            w_q=self.w_q2,
            w_k=self.w_k2,
            w_v=self.w_v2,
            w_o=self.w_o2,
            n_heads=self.n_heads,
            x_cross=encoder_output,
        )
        x = self.norm2(x + attn2_out)
        # Sub-layer 3: Feed-Forward + Add & Norm
        ffn_out: np.ndarray = self.ffn.forward(x)
        return self.norm3(x + ffn_out)
