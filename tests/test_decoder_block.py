import numpy as np
import pytest
import torch
import torch.nn as nn

from transformers_lab.causal_mask import causal_mask as make_causal_mask
from transformers_lab.decoder_block import DecoderBlock


def copy_weights_to_pytorch(
    block: DecoderBlock,
    torch_block: nn.TransformerDecoderLayer,
) -> None:
    d_model = block.w_o1.shape[0]

    def reshape_attn_weights(w: np.ndarray) -> np.ndarray:
        return w.transpose(0, 2, 1).reshape(d_model, d_model)

    # --- Masked Self-Attention ---
    w_q1 = reshape_attn_weights(block.w_q1)
    w_k1 = reshape_attn_weights(block.w_k1)
    w_v1 = reshape_attn_weights(block.w_v1)
    w_in1 = np.concatenate([w_q1, w_k1, w_v1], axis=0)

    torch_block.self_attn.in_proj_weight.data = torch.tensor(w_in1, dtype=torch.float32)
    torch_block.self_attn.in_proj_bias.data = torch.zeros(3 * d_model)
    torch_block.self_attn.out_proj.weight.data = torch.tensor(block.w_o1.T, dtype=torch.float32)
    torch_block.self_attn.out_proj.bias.data = torch.zeros(d_model)

    # --- Cross-Attention ---
    w_q2 = reshape_attn_weights(block.w_q2)
    w_k2 = reshape_attn_weights(block.w_k2)
    w_v2 = reshape_attn_weights(block.w_v2)
    w_in2 = np.concatenate([w_q2, w_k2, w_v2], axis=0)

    torch_block.multihead_attn.in_proj_weight.data = torch.tensor(w_in2, dtype=torch.float32)
    torch_block.multihead_attn.in_proj_bias.data = torch.zeros(3 * d_model)
    torch_block.multihead_attn.out_proj.weight.data = torch.tensor(block.w_o2.T, dtype=torch.float32)
    torch_block.multihead_attn.out_proj.bias.data = torch.zeros(d_model)

    # --- Feed Forward ---
    torch_block.linear1.weight.data = torch.tensor(block.ffn.w1.T, dtype=torch.float32)
    torch_block.linear1.bias.data = torch.tensor(block.ffn.b1, dtype=torch.float32)
    torch_block.linear2.weight.data = torch.tensor(block.ffn.w2.T, dtype=torch.float32)
    torch_block.linear2.bias.data = torch.tensor(block.ffn.b2, dtype=torch.float32)

    # --- Layer Norms ---
    torch_block.norm1.weight.data = torch.tensor(block.norm1.gamma, dtype=torch.float32)
    torch_block.norm1.bias.data = torch.tensor(block.norm1.beta, dtype=torch.float32)
    torch_block.norm2.weight.data = torch.tensor(block.norm2.gamma, dtype=torch.float32)
    torch_block.norm2.bias.data = torch.tensor(block.norm2.beta, dtype=torch.float32)
    torch_block.norm3.weight.data = torch.tensor(block.norm3.gamma, dtype=torch.float32)
    torch_block.norm3.bias.data = torch.tensor(block.norm3.beta, dtype=torch.float32)


def test_decoder_block() -> None:
    """DecoderBlock numpy output must match PyTorch reference."""
    d_model = 64
    n_heads = 8
    d_hidden_ff = 128
    tgt_seq_len = 5
    src_seq_len = 7

    np.random.seed(42)
    tgt_np = np.random.randn(tgt_seq_len, d_model).astype(np.float32)
    enc_np = np.random.randn(src_seq_len, d_model).astype(np.float32)

    block = DecoderBlock(d_model=d_model, n_heads=n_heads, d_hidden_ff=d_hidden_ff)

    torch_block = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_hidden_ff,
        dropout=0.0,
        activation="relu",
        batch_first=False,
        norm_first=False,
    )
    torch_block.eval()
    copy_weights_to_pytorch(block, torch_block)

    out_np = block.forward(tgt_np, enc_np)

    tgt_torch = torch.tensor(tgt_np).unsqueeze(1)
    enc_torch = torch.tensor(enc_np).unsqueeze(1)
    mask = torch.tensor(make_causal_mask(tgt_seq_len), dtype=torch.float32)
    with torch.no_grad():
        out_torch = (
            torch_block(
                tgt_torch,
                enc_torch,
                tgt_mask=mask,
            )
            .squeeze(1)
            .numpy()
        )

    assert np.allclose(out_np, out_torch, atol=1e-5), f"Outputs differ!\nNumpy:\n{out_np}\nPyTorch:\n{out_torch}"


def test_decoder_block_raises_value_error_when_d_model_not_divisible_by_n_heads() -> None:
    d_model = 65
    n_heads = 8
    d_hidden_ff = 128
    with pytest.raises(ValueError):
        DecoderBlock(d_model=d_model, n_heads=n_heads, d_hidden_ff=d_hidden_ff)
