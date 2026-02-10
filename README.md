# Transformers

## Prerequisites

- Python 3.12
- NumPy
- Linear Algebra
    - Vectors
    - Matrices
    - Matrix multiplication
    - Softmax

## A Brief History of Language Modeling

- 2013: Word2Vec / N-grams
- 2014: RNN / LSTM
- 2015: Attention mechanism
- 2017: Transformers – large pre-trained language models
- 2018: BERT
- 2019: T5
- 2020: GPT-3
- 2022: PaLM

## Why Transformers Exist

### Limitations of RNNs / LSTMs

- Sequential computation → slow and hard to parallelize
- Long-range dependencies are difficult to capture

### Attention Mechanism

- Each word decides which other words to pay attention to
- Introduced in *Attention Is All You Need*

## Attention (Scaled Dot-Product Attention)

For each token:

- Query (Q): what I am looking for
- Key (K): what I offer
- Value (V): the information I provide

### Core Formula

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

## Self-Attention

- Query, Key, and Value come from the same input
- Padding and causal masks
- Numerically stable softmax

## Multi-Head Attention

- Why multiple heads matter
- Splitting and concatenation
- Dimension consistency

## Positional Encoding

- Sinusoidal encoding
- Learned positional embeddings
- Impact on model performance

## Transformer Block

- Self-attention
- Residual connections
- Layer normalization
- Feed-forward network

## Encoder / Decoder Architectures

- Encoder-only (BERT)
- Encoder–Decoder (T5)
- Decoder-only (GPT)

## Training a Mini-Transformer

- Simple tasks (copy task, toy translation)
- Cross-entropy loss
- Backpropagation
- Limits of pure Python implementations

## Using the GPU

### CUDA (NVIDIA)

- Required for serious training
- CUDA Toolkit + cuDNN
- PyTorch
- Mixed precision (FP16 / BF16)

### Alternatives

- Apple Silicon (MPS)
- Cloud GPUs (AWS, GCP, Azure, RunPod)
