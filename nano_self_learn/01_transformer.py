"""Module 1 — handwrite a GPT-2-style transformer.

You implement THREE cores (everything else is given):
  1) scaled_dot_product_causal_attention(q,k,v)  — the heart
  2) Block.forward                                — pre-norm residual wiring
  3) GPT.forward                                  — embeddings -> blocks -> head -> loss

Shapes: B=batch, T=time/seq, C=n_embd, nh=n_head, hd=C//nh.

Run:  uv run nano_self_learn/01_transformer.py
Gate: shape test passes AND overfit-one-batch drives loss < 0.1.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128


def scaled_dot_product_causal_attention(q, k, v):
    """q,k,v: (B, nh, T, hd). Return attention output (B, nh, T, hd).

    Steps:
      att = q @ k.transpose(-2,-1) / sqrt(hd)      # (B,nh,T,T) affinities
      mask future: att[:, :, i, j>i] = -inf         # causal (torch.tril helps)
      att = softmax(att, dim=-1)
      out = att @ v
    """
    # TODO(you): implement the four lines above.
    raise NotImplementedError


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # q,k,v in one matmul
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x):  # x: (B,T,C)  — GIVEN (uses your attention fn)
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B,T,C) -> (B,nh,T,hd)
        split = lambda t: t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = scaled_dot_product_causal_attention(split(q), split(k), split(v))
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # merge heads
        return self.c_proj(y)


class MLP(nn.Module):  # GIVEN
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # TODO(you): pre-norm residual block:
        #   x = x + self.attn(self.ln1(x))
        #   x = x + self.mlp(self.ln2(x))
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying

    def forward(self, idx, targets=None):  # idx: (B,T) long
        # TODO(you):
        #   pos = arange(T); x = tok_emb(idx) + pos_emb(pos)
        #   for block in self.blocks: x = block(x)
        #   x = self.ln_f(x); logits = self.head(x)      # (B,T,vocab)
        #   loss = cross_entropy(logits.view(-1,vocab), targets.view(-1)) if targets is not None else None
        #   return logits, loss
        raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GPTConfig()
    m = GPT(cfg).to(dev)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), device=dev)
    y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), device=dev)

    logits, loss = m(x, y)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size), f"bad shape {logits.shape}"
    print(f"shapes ok | init loss {loss.item():.3f} (expect ≈ ln(vocab)={math.log(cfg.vocab_size):.3f})")

    # overfit ONE batch
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for i in range(200):
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"overfit loss after 200 steps: {loss.item():.4f}")
    print("PASS ✅" if loss.item() < 0.1 else "FAIL ❌ — model can't overfit one batch; check attention/block/forward")
