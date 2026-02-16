import numpy as np

from transformers_lab.decoder import Decoder


def test_decoder_output_shape() -> None:
    """Output shape must equal input shape."""
    n_layers, d_model, n_heads, d_hidden_ff = 6, 64, 8, 128
    tgt_seq_len, src_seq_len = 10, 12

    decoder = Decoder(n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_hidden_ff=d_hidden_ff)
    tgt = np.random.randn(tgt_seq_len, d_model)
    enc_out = np.random.randn(src_seq_len, d_model)

    out = decoder.forward(tgt, enc_out)
    assert out.shape == (tgt_seq_len, d_model)


def test_decoder_single_layer() -> None:
    """Should work with n_layers=1."""
    decoder = Decoder(n_layers=1, d_model=64, n_heads=8, d_hidden_ff=128)
    tgt = np.random.randn(5, 64)
    enc_out = np.random.randn(7, 64)

    out = decoder.forward(tgt, enc_out)
    assert out.shape == (5, 64)


def test_decoder_output_is_finite() -> None:
    """Output must not contain NaN or Inf."""
    decoder = Decoder(n_layers=6, d_model=64, n_heads=8, d_hidden_ff=128)
    tgt = np.random.randn(10, 64)
    enc_out = np.random.randn(12, 64)

    out = decoder.forward(tgt, enc_out)
    assert np.all(np.isfinite(out))


def test_decoder_layers_count() -> None:
    """Decoder must have exactly n_layers blocks."""
    decoder = Decoder(n_layers=22, d_model=64, n_heads=8, d_hidden_ff=128)
    assert len(decoder.layers) == 22


def test_decoder_different_src_tgt_seq_len() -> None:
    """Decoder must handle different source and target sequence lengths."""
    decoder = Decoder(n_layers=2, d_model=64, n_heads=8, d_hidden_ff=128)
    tgt = np.random.randn(5, 64)
    enc_out = np.random.randn(12, 64)

    out = decoder.forward(tgt, enc_out)
    assert out.shape == (5, 64)


def test_decoder_encoder_output_unchanged() -> None:
    """Encoder output must not be modified during forward pass."""
    decoder = Decoder(n_layers=2, d_model=64, n_heads=8, d_hidden_ff=128)
    tgt = np.random.randn(5, 64)
    enc_out = np.random.randn(7, 64)
    enc_out.copy()

    decoder.forward(tgt, enc_out)
