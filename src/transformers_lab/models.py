"""Full Transformer model (Encoder + Decoder + linear projection)."""

from collections.abc import Callable
from typing import cast

import numpy as np

from transformers_lab.decoder import Decoder
from transformers_lab.embedding import Embedding
from transformers_lab.encoder import Encoder
from transformers_lab.feed_forward import FeedForward
from transformers_lab.layer_norm import LayerNorm
from transformers_lab.positional_encoding import sinusoidal_positional_encoding
from transformers_lab.weight import xavier_init


class Transformer:
    """Full transformers.

    Parameters
    ----------
    n_layers : int
        Number of encoder and decoder blocks. Paper default: 6.
    d_model : int
        Model dimensionality. Paper default: 512.
    n_heads : int
        Number of attention heads. Paper default: 8.
    d_hidden_ff : int
        Hidden dimensionality of the feed-forward network. Paper default: 2048.
    src_vocab_size : int
        Size of the source vocabulary.
    tgt_vocab_size : int
        Size of the target vocabulary.
    max_seq_len : int
        Maximum sequence length for positional encoding. Default: 5000.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_hidden_ff: int,
        src_vocab_size: int,  # source
        tgt_vocab_size: int,  # target
        max_seq_len: int = 5000,
        src_embed_fn: Callable[[int, int, Callable[..., np.ndarray]], Embedding] = Embedding,
        tgt_embed_fn: Callable[[int, int, Callable[..., np.ndarray]], Embedding] = Embedding,
        weight_fn: Callable[..., np.ndarray] = xavier_init,
        feed_forward_fn: Callable[[int, int], FeedForward] = FeedForward,
        layer_norm_fn: Callable[[int], LayerNorm] = LayerNorm,
    ) -> None:
        """Initialize the Transformer."""
        self.d_model = d_model

        # Embeddings
        self.src_embedding = src_embed_fn(src_vocab_size, d_model, weight_fn)
        self.tgt_embedding = tgt_embed_fn(tgt_vocab_size, d_model, weight_fn)

        # Positional encoding — précomputé une fois
        self.pos_encoding = sinusoidal_positional_encoding(max_seq_len, d_model)

        # Encoder & Decoder
        self.encoder = Encoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_hidden_ff=d_hidden_ff,
            weight_fn=weight_fn,
            feed_forward_fn=feed_forward_fn,
            layer_norm_fn=layer_norm_fn,
        )
        self.decoder = Decoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_hidden_ff=d_hidden_ff,
            weight_fn=weight_fn,
            feed_forward_fn=feed_forward_fn,
            layer_norm_fn=layer_norm_fn,
        )

        self.w_proj: np.ndarray = weight_fn(d_model, tgt_vocab_size)

    def forward(
        self,
        src_ids: np.ndarray,
        tgt_ids: np.ndarray,
    ) -> np.ndarray:
        """Forward pass through the full Transformer.

        Parameters
        ----------
        src_ids : np.ndarray
            Source token indices of shape (src_seq_len,).
        tgt_ids : np.ndarray
            Target token indices of shape (tgt_seq_len,).

        Returns:
        -------
        np.ndarray
            Logits of shape (tgt_seq_len, tgt_vocab_size).
        """
        src_seq_len = src_ids.shape[0]
        tgt_seq_len = tgt_ids.shape[0]

        # Embedding + positional encoding
        src = self.src_embedding(src_ids) + self.pos_encoding[:src_seq_len]  # (src_seq_len, d_model)
        tgt = self.tgt_embedding(tgt_ids) + self.pos_encoding[:tgt_seq_len]  # (tgt_seq_len, d_model)

        # Encoder → Decoder → projection
        encoder_output = self.encoder.forward(src)  # (src_seq_len, d_model)
        decoder_output = self.decoder.forward(tgt, encoder_output)  # (tgt_seq_len, d_model)
        return cast(np.ndarray, decoder_output @ self.w_proj)  # (tgt_seq_len, tgt_vocab_size)
