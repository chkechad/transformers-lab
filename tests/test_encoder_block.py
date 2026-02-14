import numpy as np
import torch
import torch.nn as nn

from transformers_lab.encoder_block import EncoderBlock


def copy_weights_to_pytorch(block: EncoderBlock, torch_block: nn.TransformerEncoderLayer) -> None:
    d_model = block.w_o.shape[0]
    w_q = block.w_q.transpose(0, 2, 1).reshape(d_model, d_model)
    w_k = block.w_k.transpose(0, 2, 1).reshape(d_model, d_model)
    w_v = block.w_v.transpose(0, 2, 1).reshape(d_model, d_model)

    w_in = np.concatenate([w_q, w_k, w_v], axis=0)

    torch_block.self_attn.in_proj_weight.data = torch.tensor(w_in, dtype=torch.float32)
    torch_block.self_attn.in_proj_bias.data = torch.zeros(3 * d_model)
    torch_block.self_attn.out_proj.weight.data = torch.tensor(block.w_o.T, dtype=torch.float32)
    torch_block.self_attn.out_proj.bias.data = torch.zeros(d_model)

    torch_block.linear1.weight.data = torch.tensor(block.ffn.w1.T, dtype=torch.float32)
    torch_block.linear1.bias.data = torch.tensor(block.ffn.b1, dtype=torch.float32)
    torch_block.linear2.weight.data = torch.tensor(block.ffn.w2.T, dtype=torch.float32)
    torch_block.linear2.bias.data = torch.tensor(block.ffn.b2, dtype=torch.float32)

    torch_block.norm1.weight.data = torch.tensor(block.norm1.gamma, dtype=torch.float32)
    torch_block.norm1.bias.data = torch.tensor(block.norm1.beta, dtype=torch.float32)
    torch_block.norm2.weight.data = torch.tensor(block.norm2.gamma, dtype=torch.float32)
    torch_block.norm2.bias.data = torch.tensor(block.norm2.beta, dtype=torch.float32)


def test_encoder_block() -> None:
    d_model = 64
    n_heads = 8
    d_hidden_ff = 128
    seq_len = 5

    np.random.seed(42)
    x_np = np.random.randn(seq_len, d_model).astype(np.float32)

    # Encoder numpy block
    encoder_block = EncoderBlock(d_model=d_model, n_heads=n_heads, d_hidden_ff=d_hidden_ff)

    # --- PyTorch Encoder ---
    torch_block = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_hidden_ff,
        dropout=0.0,
        activation="relu",
        batch_first=False,
        norm_first=False,
    )
    torch_block.eval()

    copy_weights_to_pytorch(encoder_block, torch_block)

    out_np = encoder_block.forward(x_np)

    x_torch = torch.tensor(x_np).unsqueeze(1)
    with torch.no_grad():
        out_torch = torch_block(x_torch).squeeze(1).numpy()
    assert np.allclose(out_np, out_torch, atol=1e-5), f"Outputs differ!\nNumpy:\n{out_np}\nPyTorch:\n{out_torch}"
