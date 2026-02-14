"""Weight init."""

import numpy as np


def xavier_init(n1: int, n2: int) -> np.ndarray:
    """Xavier initialization for weight matrices."""
    return np.random.randn(n1, n2) / np.sqrt(n2)
