---
title: "Rotary Position Embedding"
date: 2025-11-29
draft: false
tags: ["LLM Fundamentals"]
categories: ["LLM"]
showToc: true
TocOpen: false
weight: 1
math: true
---

## 1. Introduction

Unlike RNN models, which have inherent order information, Transformer models process tokens without position information. For example, the sentence "*the cat sat on the mat*" would be indistinguishable from "*mat the on sat cat the*." Position encodings solve this problem by injecting order information into the model.

* **Absolute Position Encoding.** Each position in a sequence receives a unique representation that is added directly to the token embeddings. Learnable Position Embeddings use a learnable embedding matrix where each position index maps to a trainable vector, limited by the predefined length. Sinusoidal Position Embeddings, introduced in the original Transformer paper, use sinusoidal position encodings as a fixed, deterministic alternative. Several works have found that the performance gap between these two methods is not significant.
* **Relative Position Encoding.** Intuitively, what matters is not that a word is at position 67, but that it is 3 positions before the word we're currently processing. Relative position encoding directly encodes the distance between tokens rather than an absolute position within the given sequence. However, Figure 2 in [1] shows that relative position encoding is slow during inference, and modifying the attention mechanism adds implementation complexity.

## 2. RoPE: Rotary Position Embedding

RoPE encodes position by *rotating* the query and key vectors in attention through an angle perspective. Consider two tokens at positions $m$ and $n$. RoPE applies rotation matrices $R_m$ to the query $q$ and $R_n$ to the key. The attention score becomes:

$$
(R_mq)^T(R_nk) = q^TR_m^TR_nk = q^TR_{n-m}k.
$$

Proof:

$$
\begin{aligned}
R_\theta &= \begin{bmatrix}
\cos\theta & -\sin\theta \\\\\\
\sin\theta & \cos\theta
\end{bmatrix} \\\\\\
R^T_\theta &= \begin{bmatrix}
\cos\theta & \sin\theta \\\\\\
-\sin\theta & \cos\theta
\end{bmatrix} = \begin{bmatrix}
\cos(-\theta) & -\sin(-\theta) \\\\\\
\sin(-\theta) & \cos(-\theta)
\end{bmatrix} = R_{-\theta} \\\\\\
R_mR_n &= \begin{bmatrix}
\cos m & -\sin m \\\\\\
\sin m & \cos m
\end{bmatrix}\begin{bmatrix}
\cos n & -\sin n \\\\\\
\sin n & \cos n
\end{bmatrix} \\\\\\
&= \begin{bmatrix}
\cos m \cos n - \sin m \sin n & -\cos m \sin n - \sin m \cos n \\\\\\
\sin m \cos n + \cos m \sin n & -\sin m \sin n + \cos m \cos n
\end{bmatrix} \\\\\\
&= \begin{bmatrix}
\cos (m+n) & -\sin (m+n) \\\\\\
\sin (m+n) & \cos (m+n)
\end{bmatrix} \quad\text{//Trigonometric identities} \\\\\\
R_m^TR_n &= R_{-m}R_n = R_{n-m}.
\end{aligned}
$$

Unlike previous relative position encodings, RoPE can be applied **to the inputs before attention score calculation**, like absolute position encodings, significantly simplifying the implementation.

Since rotation is inherently a 2D operation, for a high-dimensional vector, RoPE decomposes the $d$-dimensional space into $d/2$ independent 2D planes. For dimensions $(2i, 2i+1)$ at position $m$, the rotation angle is:

$$
\theta_i = m \cdot \theta^{-2i/d}_{\text{base}},
$$

where $\theta_\text{base} = 10{,}000$ by default. Note that **each pair rotates at a different speed (frequency).** Early dimensions rotate fast, changing significantly with small position changes in a sequence. They are sensitive to *nearby* position differences and can capture local patterns. In contrast, later dimensions rotate slowly, making them insensitive to small changes. They are used to capture long-range patterns.

## 3. Implementation

Code is from [2].

```python
head_dim = 64
base = 10000
seq_len = 2048

# Pre-compute rotary embeddings
channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32) # (32,)
inv_freq = 1.0 / (base ** (channel_range / head_dim)) # \theta_{\text{base}}^{-2i/d}, (32,)
t = torch.arange(seq_len, dtype=torch.float32) # (2048,)
freqs = torch.outer(t, inv_freq) # \theta_i, (2048, 32)
cos, sin = freqs.cos(), freqs.sin()
cos, sin = cos[None, :, None, :], sin[None, :, None, :] # Add batch and head dims for later broadcasting, (batch, 2048, 64, 32)

# In transformer block computation
d = x.shape[3] // 2 # x: (batch, seq_len, head_dim)
x1, x2 = x[..., :d], x[..., d:] # Original RoPE uses Adjacent Pairing: (x0,x1), (x2,x3), ... LLaMA-style uses Half-Split Pairing: (x0, xd/2), (x1, xd/2+1), .... They are mathematically equivalent, but the LLaMA-style is easier to implement and more efficient
y1 = x1 * cos + x2 * sin
y2 = x1 * (-sin) + x2 * cos # This is a clockwise rotation while the original RoPE uses counterclockwise rotation. It does not matter since they are equivalent. If we use the original paper, y1 = x1 * cos + x2 * (-sin); y2 = x1 * sin + x2 * cos
out = torch.cat([y1, y2], dim=3)
```

---

## References

[1] Press, Ofir, Noah Smith, and Mike Lewis. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." ICLR 2022.

[2] https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
