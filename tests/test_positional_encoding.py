import numpy as np

from transformers_lab.positional_encoding import sinusoidal_positional_encoding


def test_shape() -> None:
    seq_len = 10
    d_model = 16
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    assert pe.shape == (seq_len, d_model)


def test_position_zero_values() -> None:
    seq_len = 5
    d_model = 8
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    assert np.allclose(pe[0, 0::2], 0.0)
    assert np.allclose(pe[0, 1::2], 1.0)


def test_positions_are_different() -> None:
    seq_len = 10
    d_model = 16
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    assert not np.allclose(pe[1], pe[2])


def test_even_odd_structure() -> None:
    seq_len = 6
    d_model = 12
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    assert not np.allclose(pe[:, 0::2], pe[:, 1::2])


def test_large_dimensions_stability() -> None:
    seq_len = 50
    d_model = 128
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    assert not np.isnan(pe).any()
    assert not np.isinf(pe).any()


def test_manual_value_check() -> None:
    seq_len = 3
    d_model = 4
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    position = 1
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    expected_sin = np.sin(position * div_term)
    expected_cos = np.cos(position * div_term)
    assert np.allclose(pe[1, 0::2], expected_sin)
    assert np.allclose(pe[1, 1::2], expected_cos)
