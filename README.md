# Transformers

## Prerequisite

- Python3.12
- Numpy
- Algèbre linéaire
    - Vecteurs
    - Matrices
    - Produits matriciel
    - Softmax

## Pourquoi les transformers existe :

- RNN / LSTM

- Attention: Chaque mot décide à quelles autres mots, il doit faire attention
- [Attention is all you need](https://arxiv.org/pdf/1706.03762)

## Attention (Scaled Dot-Product Attention)

Pour chaque token :

- **Query (Q)** : ce que je cherche
- **Key (K)** : ce que je propose
- **Value (V)** : l’information que je fournis

### Formule centrale

$$
\text{Attention}(Q, K, V)
=
\text{softmax}\left(
\frac{QK^{\top}}{\sqrt{d_k}}
\right)V
$$

### Où :

- $Q \in \mathbb{R}^{n \times d_k}$
- $K \in \mathbb{R}^{n \times d_k}$
- $V \in \mathbb{R}^{n \times d_v}$
- $d_k$ : dimension des clés
- $n$ : nombre de tokens

## Self attention

- Query,Key, Value
- Scaled Dot-Product Attention
- Softmax Stable
- Masque (Padding + Causal)

## Multi-Head attention

- Pourquoi plusieurs têtes
- Comment splitter / Concaténer
- Comment garder les dimensions propre

## Positional Encoding

- Sinus / Cosinus
- Encodage appris
- impact réel sur le modèle

## Bloc transformer

- attention
- residual connections
- layer normalization
- feed-forward network

## Encoder, Decoder, masques

- Encoder-only (BERT)
- Encoder-Decoder (T5)
- Encoder-only (GPT)

## Entraîner un mini-Transformer

- task simple (copy task, traduction toy)
- loss
- backprop (au moins conceptuellement)
- limitations du “Python pur”
