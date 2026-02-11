import numpy as np
import torch

from src.multihead_attention import multi_head_attention


def test_multi_head_matches_torch() -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    seq_len = 5
    d_model = 8
    n_heads = 2
    d_k = d_model // n_heads

    x_np = np.random.randn(seq_len, d_model).astype(np.float32)

    w_q_np = np.random.randn(n_heads, d_model, d_k).astype(np.float32)
    w_k_np = np.random.randn(n_heads, d_model, d_k).astype(np.float32)
    w_v_np = np.random.randn(n_heads, d_model, d_k).astype(np.float32)
    w_o_np = np.random.randn(d_model, d_model).astype(np.float32)

    result_np = multi_head_attention(x_np, w_q_np, w_k_np, w_v_np, w_o_np, n_heads)

    mha = torch.nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=n_heads,
        bias=False,
        batch_first=True,
    )

    w_q_flat = np.concatenate([w_q_np[h] for h in range(n_heads)], axis=1)
    w_k_flat = np.concatenate([w_k_np[h] for h in range(n_heads)], axis=1)
    w_v_flat = np.concatenate([w_v_np[h] for h in range(n_heads)], axis=1)

    in_proj_weight = np.concatenate([w_q_flat.T, w_k_flat.T, w_v_flat.T], axis=0)

    mha.in_proj_weight.data = torch.tensor(in_proj_weight)
    mha.out_proj.weight.data = torch.tensor(w_o_np.T)

    x_t = torch.tensor(x_np).unsqueeze(0)

    out_t, _ = mha(x_t, x_t, x_t)
    out_t_np = out_t.squeeze(0).detach().numpy()

    assert np.allclose(result_np, out_t_np, atol=1e-5)
