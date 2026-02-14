# Transformer From Scratch

## Context

Rewriting the transformer architecture from scratch in pure Python, without using any deep learning libraries, to
understand the inner workings of the model.

## Strict Rules

- NEVER import huggingface, transformers, or any high-level ML libraries
- Only allowed: `torch`, `torch.nn`, `math`, `numpy`
- Always comment tensor dimensions at each step: `# (batch_size, seq_len, d_model)`
- No copy-pasting from existing implementations — write everything from understanding

## Stack

- Python3.12
- numpy
- PyTorch2
- No Gpu required

## Implementation Roadmap

1. Scaled Dot-Product Attention
2. MultiHead Attention
3. Position Wise Feed-Forward Network
4. Positional Encoding
5. Add & Norm (Residual + LayerNorm)
6. Encoder Block
7. Decoder Block
8. Full Transformer

## Code Style

- Explicit variable names: `query`, `key`, `value` -not `x1`, `x2`, `x3`
- Step-by-step calculations with comments
- Type hint anywhere helpful for clarity
- Docstrings must include input/output shapes and a brief description of the operation
- Keep hyperparameters matching the original paper:
    - `d_model=512`, `num_heads=8`, `d_ff=2048`, `dropout=0.1`

## File Structure

```
transformer_lab/
├── self_attention.py
├── feed_forward.py
├── positional_encoding.py
├── multihead_attention.py
├── layer_norm.py
├── softmax.py
```

## Reference

- Paper: https://arxiv.org/abs/1706.03762
- All formulas must match exactly:
    - Attention: `softmax(QK^T / sqrt(d_k)) * V`
    - FFN: `max(0, xW1 + b1)W2 + b2`
