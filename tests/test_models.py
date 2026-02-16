import numpy as np

from transformers_lab.models import Transformer


def test_transformer_smoke() -> None:
    np.random.seed(42)
    model = Transformer(
        n_layers=2,
        d_model=64,
        n_heads=8,
        d_hidden_ff=128,
        src_vocab_size=1000,
        tgt_vocab_size=1000,
    )

    src_ids = np.array([4, 12, 7, 3, 9])
    tgt_ids = np.array([9, 2, 5])

    logits = model.forward(src_ids, tgt_ids)

    assert logits.shape == (3, 1000)
    assert np.all(np.isfinite(logits))
