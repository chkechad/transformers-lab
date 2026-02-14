"""Tests for the Encoder module."""

import numpy as np

from transformers_lab.encoder import Encoder


def test_encoder_output_shape() -> None:
    """Output shape must equal input shape."""
    encoder = Encoder(n_layers=6, d_model=64, n_heads=8, d_hidden_ff=128)
    x = np.random.randn(10, 64)
    out = encoder.forward(x)
    assert out.shape == x.shape


def test_encoder_single_layer() -> None:
    """Should work with n_layers=1."""
    encoder = Encoder(n_layers=1, d_model=64, n_heads=8, d_hidden_ff=128)
    x = np.random.randn(5, 64)
    out = encoder.forward(x)
    assert out.shape == (5, 64)


def test_encoder_output_is_finite() -> None:
    """Output must not contain NaN or Inf."""
    encoder = Encoder(n_layers=6, d_model=64, n_heads=8, d_hidden_ff=128)
    x = np.random.randn(10, 64)
    out = encoder.forward(x)
    assert np.all(np.isfinite(out))


def test_encoder_layers_count() -> None:
    """Encoder must have exactly n_layers blocks."""
    encoder = Encoder(n_layers=22, d_model=64, n_heads=8, d_hidden_ff=128)
    assert len(encoder.layers) == 22
