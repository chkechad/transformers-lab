import numpy as np
from torch import from_numpy
from torch import softmax as torch_softmax

from softmax import softmax


def test_softmax_sum_to_one() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = softmax(x)
    y_torch_softmax = torch_softmax(from_numpy(x), -1)
    assert np.allclose(y_torch_softmax, y, atol=1e-7)

    assert np.isclose(np.sum(y), 1.0)


def test_softmax_positive() -> None:
    x = np.array([-100.0, 0.0, 100.0])
    y = softmax(x)

    assert np.all(y >= 0)


def test_softmax_translation_invariance() -> None:
    x = np.array([1.0, 2.0, 3.0])

    y1 = softmax(x)
    y2 = softmax(x + 1_000_000)

    assert np.allclose(y1, y2)


def test_softmax_uniform() -> None:
    x = np.array([5.0, 5.0, 5.0, 5.0])
    y = softmax(x)

    expected = np.ones_like(x) / len(x)

    assert np.allclose(y, expected)


def test_softmax_dominant_value() -> None:
    x = np.array([0.0, 0.0, 100.0])
    y = softmax(x)

    assert y[2] > 0.99


def test_softmax_axis_2d() -> None:
    x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    y = softmax(x, axis=1)

    row_sums = np.sum(y, axis=1)

    assert np.allclose(row_sums, np.ones(x.shape[0]))


def test_softmax_no_nan_inf() -> None:
    x = np.array([1e9, 1e9 + 1, 1e9 + 2])
    y = softmax(x)

    assert np.all(np.isfinite(y))
