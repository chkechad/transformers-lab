import numpy as np

from src.layer_norm import LayerNorm


def test_layer_normalization() -> None:
    x = np.array([[1, 2, 3], [3, 5, 7]], dtype=np.float32)
    ln = LayerNorm(d_model=3)
    got = ln(x)
    expected = np.array([[-1.224744, 0, 1.224744], [-1.2247447, 0.0, 1.2247447]], dtype=np.float32)
    assert np.allclose(got, expected)
