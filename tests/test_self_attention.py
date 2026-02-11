import numpy as np
import torch
from torch.nn.functional import (
    scaled_dot_product_attention as torch_scaled_dot_product_attention,
)

from src.self_attention import scaled_dot_product_attention, self_attention


def test_scaled_dot_product_attention_matches_torch() -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    seq_len = 5
    d_k = 4

    q = np.random.randn(seq_len, d_k).astype(np.float32)
    k = np.random.randn(seq_len, d_k).astype(np.float32)
    v = np.random.randn(seq_len, d_k).astype(np.float32)
    result_np = scaled_dot_product_attention(q, k, v)

    q_t = torch.tensor(q).unsqueeze(0).unsqueeze(0)
    k_t = torch.tensor(k).unsqueeze(0).unsqueeze(0)
    v_t = torch.tensor(v).unsqueeze(0).unsqueeze(0)

    result_torch = torch_scaled_dot_product_attention(q_t, k_t, v_t)
    result_torch_to_np = result_torch.squeeze(0).squeeze(0).detach().numpy()
    assert np.allclose(result_np, result_torch_to_np, atol=1e-5)


def test_self_attention_matches_torch() -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    seq_len = 6
    d_model = 8
    d_k = 4

    x_np = np.random.randn(seq_len, d_model).astype(np.float32)

    w_q_np = np.random.randn(d_model, d_k).astype(np.float32)
    w_k_np = np.random.randn(d_model, d_k).astype(np.float32)
    w_v_np = np.random.randn(d_model, d_k).astype(np.float32)

    result_np = self_attention(x_np, w_q_np, w_k_np, w_v_np)

    x_t = torch.tensor(x_np)
    w_q_t = torch.tensor(w_q_np)
    w_k_t = torch.tensor(w_k_np)
    w_v_t = torch.tensor(w_v_np)

    q_t = x_t @ w_q_t
    k_t = x_t @ w_k_t
    v_t = x_t @ w_v_t

    q_t = q_t.unsqueeze(0).unsqueeze(0)
    k_t = k_t.unsqueeze(0).unsqueeze(0)
    v_t = v_t.unsqueeze(0).unsqueeze(0)

    result_torch = torch_scaled_dot_product_attention(q_t, k_t, v_t)

    result_torch_to_np = result_torch.squeeze(0).squeeze(0).detach().numpy()

    assert np.allclose(result_np, result_torch_to_np, atol=1e-5)
