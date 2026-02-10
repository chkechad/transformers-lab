import numpy as np

from self_attention import self_attention


def test_self_attention_shape() -> None:
    x = np.random.randn(5, 8)
    out = self_attention(x, d_k=4)

    assert out.shape == (5, 4)


def test_self_attention_no_nan_inf() -> None:
    x = np.full((3, 8), 1e9)
    out = self_attention(x, d_k=4)

    assert np.all(np.isfinite(out))
