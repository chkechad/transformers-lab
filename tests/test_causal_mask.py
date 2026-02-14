"""Tests for make_causal_mask against PyTorch reference."""

import numpy as np
import torch

from transformers_lab.causal_mask import causal_mask


def test_causal_mask_shape() -> None:
    mask = causal_mask(4)
    assert mask.shape == (4, 4)


def test_causal_mask_diagonal_is_zero() -> None:
    """Diagonal must be 0 — each token attends to itself."""
    mask = causal_mask(4)
    assert np.all(np.diag(mask) == 0.0)


def test_causal_mask_lower_triangle_is_zero() -> None:
    mask = causal_mask(4)
    rows, cols = np.tril_indices(4, k=0)
    assert np.all(mask[rows, cols] == 0.0)


def test_causal_mask_upper_triangle_is_neg_inf() -> None:
    mask = causal_mask(4)
    rows, cols = np.triu_indices(4, k=1)
    assert np.all(mask[rows, cols] == -np.inf)


def test_causal_mask_matches_pytorch() -> None:
    seq_len = 25
    mask_np = causal_mask(seq_len)
    mask_torch = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).numpy()
    np.testing.assert_array_equal(mask_np, mask_torch)
