"""Module 5 (OPTIONAL) — two modern attention upgrades: QK-norm + sliding-window.

QK-norm: RMS-normalize q and k (per head) BEFORE the dot product. Stabilizes attention logits
(keeps them from exploding), lets you drop the 1/sqrt(d) or use it alongside. Used in nanochat.

Sliding-window: each query attends only to the last `window` keys (plus causal). Cheaper attention;
mix a few full-context layers with many local ones (the SSSL/TTTL patterns you saw).

You implement:
  1) qk_norm_attention(q,k,v)                 — rms-norm q,k then causal attention
  2) sliding_window_causal_attention(q,k,v,w) — causal AND within a window of w
Gate: shapes ok; sliding window with w>=T reproduces full causal attention.

Run:  uv run nano_self_learn/05_optional_qknorm_swa.py
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def rms_norm(x, eps=1e-6):  # GIVEN — normalize over last dim, no learnable scale
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _causal_attn(q, k, v, extra_mask=None):  # GIVEN helper (full causal + optional extra mask)
    T = q.size(-2)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    if extra_mask is not None:
        causal = causal & extra_mask
    att = att.masked_fill(~causal, float("-inf"))
    return torch.softmax(att, dim=-1) @ v


def qk_norm_attention(q, k, v):
    """q,k,v: (B,nh,T,hd). RMS-norm q and k over the head dim, then causal attention."""
    # TODO(you): q = rms_norm(q); k = rms_norm(k); return _causal_attn(q,k,v)
    raise NotImplementedError


def sliding_window_causal_attention(q, k, v, window: int):
    """Each position i attends to keys j with  i-window < j <= i  (causal + local window)."""
    # TODO(you): build a (T,T) bool mask that is True where (i - j) < window, pass as extra_mask.
    #   hint: idx = torch.arange(T); dist = idx[:,None] - idx[None,:]; local = (dist >= 0) & (dist < window)
    raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    B, nh, T, hd = 2, 4, 16, 8
    q, k, v = (torch.randn(B, nh, T, hd) for _ in range(3))

    y1 = qk_norm_attention(q, k, v)
    assert y1.shape == (B, nh, T, hd), f"qk-norm bad shape {y1.shape}"

    full = _causal_attn(q, k, v)
    yw = sliding_window_causal_attention(q, k, v, window=T)   # w>=T == full causal
    same = torch.allclose(yw, full, atol=1e-5)
    small = sliding_window_causal_attention(q, k, v, window=4)  # should differ
    diff = not torch.allclose(small, full, atol=1e-4)
    print(f"qk-norm shape ok | window=T matches full: {same} | window=4 differs: {diff}")
    print("PASS ✅" if (same and diff) else "FAIL ❌ — check the window mask")
