"""Shared test fixtures."""

from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture()  # type: ignore
def deterministic_weight_init() -> Callable[[int, int], np.ndarray]:
    """Weight init factory that produces reproducible weights. #TODO use helpers for this."""
    rng = np.random.default_rng(seed=42)

    def _init(n1: int, n2: int) -> np.ndarray:
        return rng.standard_normal((n1, n2)) / np.sqrt(n2)

    return _init
