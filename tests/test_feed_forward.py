import numpy as np

from src.feed_forward import FeedForward


def test_exemple_forward() -> None:
    x = np.array(
        [
            [1.0, 0.5, -0.3, 2.0],
            [0.1, -1.2, 0.7, 0.3],
            [-0.5, 0.2, 1.5, -0.8],
        ]
    )
    d_model = 4
    d_ff = 8
    ffn = FeedForward(d_model, d_ff)
    got = ffn.forward(x)
    assert got.shape == (3, 4)
    assert not np.allclose(got[0], got[1])
    assert not np.allclose(got[1], got[2])
    assert not np.allclose(got, x)
