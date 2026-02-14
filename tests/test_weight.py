import numpy as np
from pytest_mock import MockerFixture

from transformers_lab.weight import xavier_init


def test_xavier_init(mocker: MockerFixture) -> None:
    mock_values = np.array([[0.5, -1.2, 0.3], [1.0, 0.7, -0.8], [0.1, -0.4, 1.5], [-0.9, 0.6, -0.2], [1.1, -0.5, 0.4]])
    mocker.patch("numpy.random.randn", return_value=mock_values)
    n1 = 5
    n2 = 3
    weights = xavier_init(n1, n2)

    assert weights.shape == (n1, n2)
    expected = mock_values / np.sqrt(n2)
    np.testing.assert_allclose(weights, expected)
