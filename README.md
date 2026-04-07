# Transformers

![CI](https://github.com/chkechad/transformers-lab/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/github/license/chkechad/transformers-lab)
![Last Commit](https://img.shields.io/github/last-commit/chkechad/transformers-lab)
![Coverage](https://codecov.io/gh/chkechad/transformers-lab/branch/main/graph/badge.svg)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-blue?logo=materialformkdocs)](https://chkechad.github.io/transformers-lab/)

Transformer architecture implemented from scratch — pure NumPy, no PyTorch, no HuggingFace.
Every forward pass validated numerically against PyTorch references.
The goal: understand every single operation, then push to modern architectures, GPU training, Rust kernels, and
production MLOps.

## Prerequisites

- Python 3.12
- NumPy
- Linear Algebra
    - Vectors, Matrices, Matrix multiplication
    - Softmax

---

## A Brief History of Language Modeling

| Year | Milestone                                   |
|------|---------------------------------------------|
| 2013 | Word2Vec / N-grams                          |
| 2014 | RNN / LSTM                                  |
| 2015 | Attention mechanism                         |
| 2017 | **Transformer** — Attention Is All You Need |
| 2018 | BERT (encoder-only)                         |
| 2019 | GPT-2, T5                                   |
| 2020 | GPT-3                                       |
| 2022 | PaLM, ChatGPT                               |
| 2023 | LLaMA, Mistral                              |
| 2024 | LLaMA 3, DeepSeek V2/V3                     |
| 2025 | DeepSeek R1                                 |

---

## Why Transformers Exist

### Limitations of RNNs / LSTMs

- Sequential computation → slow and hard to parallelize
- Long-range dependencies are difficult to capture

### Attention Mechanism

- Each word decides which other words to pay attention to
- Introduced [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

## Attention (Scaled Dot-Product Attention)

For each token:

- Query (Q): what I am looking for
- Key (K): what I offer
- Value (V): the information I provide

### Core Formula

Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V

- \(Q\) : Query
- \(K\) : Key
- \(V\) : Value
- \(d_k\) : dimension of the keys

## Implementation

### ✅ Implemented & Validated

#### Core Math

- **Softmax** — numerically stable via max subtraction
- **Xavier Initialization** — weight init for stable training
- **Scaled Dot-Product Attention** — `softmax(QKᵀ / sqrt(d_k)) V`
- **Causal Mask** — upper triangular `-inf` mask for autoregressive decoding

#### Modules

- **LayerNorm** — `gamma * (x - mean) / std + beta`
- **FeedForward** — `max(0, xW1 + b1)W2 + b2`
- **Embedding** — token lookup table scaled by `sqrt(d_model)`
- **Positional Encoding** — sinusoidal encoding from Vaswani et al.

#### Attention

- **MultiHead Attention** — self-attention + cross-attention
- **EncoderBlock** — MultiHead Attention → Add & Norm → FFN → Add & Norm
- **DecoderBlock** — Masked Attention → Add & Norm → Cross Attention → Add & Norm → FFN → Add & Norm

#### Full Architecture

- **Encoder** — stack of N EncoderBlocks
- **Decoder** — stack of N DecoderBlocks
- **Transformer** — Embedding + Positional Encoding + Encoder + Decoder + Linear Projection

#### Modern Architectures

- RoPE (Rotary Positional Embedding)
    - Position: Sinusoïdale additive positional encoding → RoPE (Rotary Positional Embedding)
        - Avantage : Cela permet au modèle de mieux capturer les distances relatives entre les tokens, contrairement à
          l'approche fixe et additive.
- RMSNorm
    - Norm: LayerNorm (Moyenne + Var) → RMSNorm (RMS uniquement)
        - Avantage : RMSNorm est plus rapide à calculer et peut offrir des performances similaires ou meilleures que
          LayerNorm dans certains cas, en particulier pour les modèles de grande taille.
- SwiGLU
    - Activation: ReLU → SwiGLU
        - Avantage : SwiGLU est une activation plus expressive que ReLU, permettant au modèle de mieux capturer les
          interactions complexes entre les caractéristiques.
- GQA (Grouped Query Attention)
    - Attention: MHA (1Q : 1K : 1V) → GQA (Gpe Q : 1K : 1V)
        - Avantage : GQA permet de réduire le coût de calcul de l'attention en regroupant les requêtes, tout en
          maintenant une bonne performance.
- LLaMA:  Decoder-only + RoPE, RMSNorm et SwiGLU
- MoE (Mixture of Experts)
- DeepSeek V3
    - Il améliore GQA multihead_attention.py avec la MLA (Multi-head Latent Attention). Au lieu de simplement grouper
      les
      têtes, il compresse les clés et valeurs
      dans un vecteur latent de basse dimension.
- DeepSeek R1
    - La différence n'est pas structurelle (elle utilise l'architecture de V3), mais réside dans l'entraînement. Il
      utilise
      l'apprentissage par renforcement (RL) pour forcer le modèle à générer une "chaîne de pensée" (Reasoning).
- Mamba (State Space Models)
    - Ce qui change : C'est une rupture totale. Mamba remplace complètement le mécanisme de scaled_dot_product_attention
      scaled_dot_prod_attention.py par un modèle d'espace d'états (SSM).

#### Résumé

| Composant      | Ton implémentation            | Modernisation                          | Modèle cible     |
|:---------------|:------------------------------|:---------------------------------------|:-----------------|
| **Position**   | Sinusoïdale additive [12]     | $\rightarrow$ RoPE (Rotation)          | LLaMA, Mistral   |
| **Norm**       | LayerNorm (Moyenne + Var) [9] | $\rightarrow$ RMSNorm (RMS uniquement) | LLaMA, PaLM      |
| **Activation** | ReLU [8]                      | $\rightarrow$ SwiGLU                   | LLaMA, DeepSeek  |
| **Attention**  | MHA (1Q : 1K : 1V) [11]       | $\rightarrow$ GQA (Gpe Q : 1K : 1V)    | LLaMA 3, Mistral |
| **Structure**  | Encoder-Decoder [10]          | $\rightarrow$ Decoder-only / MoE       | GPT-4, DeepSeek  |
| **Mécanisme**  | Attention $O(N^2)$ [13]       | $\rightarrow$ SSM (Selective Scan)     | Mamba            |

---

## Encoder / Decoder Architectures

| Architecture    | Models      | Use case                   |
|-----------------|-------------|----------------------------|
| Encoder-only    | BERT        | Classification, NER        |
| Encoder-Decoder | T5, Vaswani | Translation, summarization |
| Decoder-only    | GPT, LLaMA  | Generation, reasoning      |

---

## Stack

- Python 3.12
- NumPy
- PyTorch - validation reference only
- uv - package manager
- pytest - tests
- mypy - type checking
- ruff - linting
