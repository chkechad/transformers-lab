from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

from transformers_lab.embedding import Embedding


def test_embedding(deterministic_weight_init: Callable[[int, int], np.ndarray]) -> None:
    vocab_size = 10000
    d_model = 512

    embedding = Embedding(vocab_size=vocab_size, d_model=d_model, init_weight_fn=deterministic_weight_init)
    token_ids = np.array([11, 25, 455])
    vectors_np = embedding(token_ids)

    assert vectors_np.shape == (3, 512)

    torch_embedding = nn.Embedding(vocab_size, d_model)
    torch_embedding.weight.data = torch.tensor(embedding.weights, dtype=torch.float32)

    vectors_torch = torch_embedding(torch.tensor(token_ids)).detach().numpy() * np.sqrt(d_model)
    np.testing.assert_allclose(vectors_np, vectors_torch, atol=1e-5)
