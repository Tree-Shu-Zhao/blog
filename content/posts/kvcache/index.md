---
title: "KV Cache"
date: 2025-12-01
draft: false
tags: ["LLM Fundamentals"]
categories: ["LLM"]
showToc: true
TocOpen: false
weight: 1
math: true
---

## 1. KV Cache

Transformer models generate tokens **autoregressively,** producing one token at a time, where each new token depends on all previous tokens. In vanilla transformer layers, 
$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$
Assume a sequence "*The cat sat on the*." To generate a new token "*mat*" at time step t, we only need to use "*mat*" as the query to attend to all previous K and V of "*The cat sat on the*,"
$$
\operatorname{Attention}(Q_t, K_{1:t}, V_{1:t}) = \operatorname{softmax}\left(\frac{Q_tK_{1:t}^T}{\sqrt{d}}\right)V_{1:t}
$$
We observe that:

* $Q_t$, $K_t$, and $V_t$ depend on the current input token (i.e., the last generated token).
* $Q_{1:t-1}$ will never be **used** after the $t-1$ step.
* $K_{1:t-1}$ and $V_{1:t-1}$ will never be **changed** after the $t-1$ step.

Therefore, we can save $K$ and $V$ and reuse them in the next step, skipping the $K$ and $V$ calculations to accelerate inference time. It reduces total computation from $O(T^2)$ to $O(T)$, trading memory for speed.

Assume the batch size is $B$, sequence length is $T$, embedding dimension is $d$, the number of layers is $L$, and the precision is $P$ in bytes. The memory consumption is $2 * B * T * d * L * P,$ where $2$ denotes $K$ and $V$.

## 2. Implementation

Code is adapted from [1] and [2].

```python
class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0
    
    def insert_kv(self, layer_idx, k, v):
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
            
        B, H, T_add, D = k.size() # Prefilling: T_add = 100 (for example); Decoding: T_add = 1
        t0, t1 = self.pos, self.pos+T_add
        
        # Insert new K, V at current position
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        
        # Return full cache up to current position
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        
        # After LAST layer, we have processed all layers and need to update the pos
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        
        return key_view, value_view
    
    def prefill(self, other):
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        self.kv_cache = [:, :, :, :, :other.pos, :] = other.kv_cache[:, :, :, :, :other.pos, :]
        self.pos = other.pos
        

class Engine:
    def generate(self, tokens, max_tokens, ...):
        # Phase 1: Prefill - process entire prompt
        kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), ...)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        
        # Clone cache for decode phase (potentially with larger batch for multiple samples)
        kv_cache_decode = KVCache(batch_size=num_samples, seq_len=len(tokens)+max_tokens, ...)
        kv_cache_decode.prefill(kv_cache_prefill)
        
        # Phase 2: Decode - one token at a time
        while num_generated < max_tokens:
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)
            next_token = sample(logits[:, -1, :])
            
            yield next_token
            
            # Next iteration: forward only the new token
            ids = torch.tensor([[next_token]], dtype=torch.long, device=device)

            
class CausalSelfAttention(nn.Module):
    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()
        
        # Compute Q, K, V for new tokens only
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # Apply RoPE and QK-norm BEFORE caching
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Insert into cache, retrieve full K, V history
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        
        Tq = q.size(2)  # Number of new queries
        Tk = k.size(2)  # Total keys (cached + new)
        
        # Attention with appropriate masking
        if kv_cache is None or Tq == Tk:
            # Training or full prefill: standard causal attention
            # Queries:  q1  q2  q3  q4
            # Keys:     k1  k2  k3  k4
            # Causal Mask (1 = attend, 0 = mask):
            #       k1  k2  k3  k4
            # q1 [  1   0   0   0  ]
            # q2 [  1   1   0   0  ]
            # q3 [  1   1   1   0  ]
            # q4 [  1   1   1   1  ]
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            # Single token decode: attend to all cached keys
            # Query:    q5 (just one!)
            # Keys:     k1  k2  k3  k4  k5 (from cache + new)
            # No mask needed:
            #       k1  k2  k3  k4  k5
            # q5 [  1   1   1   1   1  ]
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # Chunk decode: custom mask for prefix + causal within chunk
            # Query:    q5  q6  q7
            # Keys:     k1  k2  k3  k4  k5  k6  k7
            #           ^^^^^^^^^^^^^^  ^^^^^^^^^^
            #           cached (prefix) new (causal within)
            # Custom Mask:
            #       k1  k2  k3  k4   k5  k6  k7
            # q5 [  1   1   1   1 |  1   0   0  ]
            # q6 [  1   1   1   1 |  1   1   0  ]
            # q7 [  1   1   1   1 |  1   1   1  ]
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, -1))

```

---

## References

[1] https://github.com/karpathy/nanochat/blob/master/nanochat/engine.py

[2] https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py