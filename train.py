"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# The router adds several distinct AdamW param shapes (table, step-emb, fc1/fc2 + biases); the
# fused compiled optimizer steps specialize per shape, so raise the dynamo cache/recompile limit
# above the default 8 to fit them all (one-time compiles, then cached).
import torch._dynamo
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 512

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

import wandb
RUN_NAME = os.environ.get("RUN_NAME", "mcts-gpt")

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Routed-blocks model: block pool + learned router over (blocks + identity)
# ---------------------------------------------------------------------------

class Router(nn.Module):
    """Tiny policy over (POOL_K blocks + identity). Computes in fp32 (MoE practice).
    L1 "static": a learned logit table (n_steps, n_options) — a fixed program, ~150 params.
    L2 "seq": table + zero-init MLP on (mean-pooled h, step embedding) — input-conditioned."""

    def __init__(self, n_embd, n_steps, n_options, level):
        super().__init__()
        self.level = level
        self.table = nn.Parameter(torch.zeros(n_steps, n_options))
        if level == "seq":
            step_dim, hidden = 64, 256
            self.step_emb = nn.Embedding(n_steps, step_dim)
            self.fc1 = nn.Linear(n_embd + step_dim, hidden)
            self.fc2 = nn.Linear(hidden, n_options)

    def forward(self, t, x):
        with torch.autocast(device_type="cuda", enabled=False):
            logits = self.table[t].float()
            if self.level == "seq":
                h = x.float().mean(dim=1)  # (B, C)
                e = self.step_emb.weight[t].float().expand(h.size(0), -1)
                logits = logits + self.fc2(F.relu(self.fc1(torch.cat([h, e], dim=-1))))
        return logits  # (n_options,) for static, (B, n_options) for seq


class RoutedGPT(nn.Module):
    """Block-pool transformer: fixed input/output blocks around ROUTE_STEPS routed steps.
    Each step a router picks among POOL_K blocks + identity; an identity tail == early exit
    at inference. All K blocks execute every step and outputs are combined with router
    weights (soft early -> ST-Gumbel hard) so the compiled graph stays static."""

    def __init__(self, config, pool_k, route_steps, router_level):
        super().__init__()
        self.config = config
        self.pool_k = pool_k
        self.route_steps = route_steps
        # Per-pool-block attention span: L=full, S=half, T=quarter (rsi TTTL). Router picks span.
        long_w = (config.sequence_len, 0)
        char_to_window = {"L": long_w, "S": (config.sequence_len // 2, 0), "T": (config.sequence_len // 4, 0)}
        pattern = config.window_pattern.upper()
        self.pool_windows = [char_to_window[pattern[k % len(pattern)]] for k in range(pool_k)]
        self.win_long = long_w
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
        })
        # Block(config, k): has_ve(k, n_layer=8) gives odd-k blocks a ve_gate, mirroring baseline
        self.block_in = Block(config, 0)   # no VE
        self.block_out = Block(config, 7)  # VE (baseline: last layer always has VE)
        self.pool = nn.ModuleList([Block(config, k) for k in range(pool_k)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Per-STEP scalars (route_steps + in + out)
        n_steps_total = route_steps + 2
        self.resid_lambdas = nn.Parameter(torch.ones(n_steps_total))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_steps_total))
        # Value embeddings: odd pool blocks + output block
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            **{str(k): nn.Embedding(config.vocab_size, kv_dim) for k in range(pool_k) if k % 2 == 1},
            "out": nn.Embedding(config.vocab_size, kv_dim),
        })
        self.router = Router(config.n_embd, route_steps, pool_k + 1, router_level)
        # Schedule knobs written in-place by the train loop (avoids recompilation)
        self.register_buffer("router_temp", torch.ones(()), persistent=False)
        self.register_buffer("router_hard", torch.zeros(()), persistent=False)
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    _precompute_rotary_embeddings = GPT._precompute_rotary_embeddings

    def all_blocks(self):
        return [self.block_in, self.block_out, *self.pool]

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.all_blocks():
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Router init. "recursive": cycle blocks 0..K-1 across all steps (depth via reuse — the
        # bias that beats a param-matched sequential baseline). "sequential": step t -> block t,
        # identity beyond the pool (== shallow baseline; collapses in practice).
        self.router.table.zero_()
        for t in range(self.route_steps):
            if ROUTER_INIT == "recursive":
                j = t % self.pool_k
            else:
                j = t if t < self.pool_k else self.pool_k
            self.router.table[t, j] = SEQ_INIT_BIAS
        if self.router.level == "seq":
            torch.nn.init.normal_(self.router.step_emb.weight, mean=0.0, std=0.02)
            torch.nn.init.uniform_(self.router.fc1.weight, -s, s)
            torch.nn.init.zeros_(self.router.fc1.bias)
            torch.nn.init.zeros_(self.router.fc2.weight)  # logits == table at init
            torch.nn.init.zeros_(self.router.fc2.bias)
        # Rotary + bf16 casts (mirrors GPT.init_weights)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def estimate_flops(self):
        """FLOPs per token (fwd+bwd): pool blocks execute ROUTE_STEPS times each, in/out once."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        def block_matmul_flops(block):
            return 6 * sum(p.numel() for p in block.parameters() if p.ndim == 2)
        def attn_flops(window_size):
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            return 12 * h * q * effective_seq
        flops = 6 * self.lm_head.weight.numel()
        flops += block_matmul_flops(self.block_in) + attn_flops(self.win_long)
        flops += block_matmul_flops(self.block_out) + attn_flops(self.win_long)
        for k, block in enumerate(self.pool):
            flops += self.route_steps * (block_matmul_flops(block) + attn_flops(self.pool_windows[k]))
        return flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for b in self.all_blocks() for p in b.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        router = sum(p.numel() for p in self.router.parameters())
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + router
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars,
            'router': router, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = [p for b in self.all_blocks() for p in b.parameters()]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        router_params = list(self.router.parameters())
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) +
            len(router_params))
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=router_params, lr=ROUTER_LR, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        K = self.pool_k

        _dbg = os.environ.get("DBG_NAN") and self.training
        def _ck(tag, t):
            if _dbg and not torch.isfinite(t).all():
                print(f"[nan] first non-finite at {tag}", flush=True); raise SystemExit(3)

        x = norm(self.transformer.wte(idx))
        x0 = x
        # Value embeddings depend only on idx: compute once, reuse across all steps
        ve = {key: emb(idx) for key, emb in self.value_embeds.items()}
        _ck("wte", x)

        x = self.resid_lambdas[0] * x + self.x0_lambdas[0] * x0
        x = self.block_in(x, None, cos_sin, self.win_long)
        _ck("block_in", x)

        probs_sum = None  # noise-free router probs, accumulated for aux losses
        z_acc = None
        sample = self.training and not ROUTER_FROZEN
        for t in range(self.route_steps):
            x_in = self.resid_lambdas[t + 1] * x + self.x0_lambdas[t + 1] * x0
            logits = self.router(t, x_in)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0).expand(B, -1)
            if sample:
                if self.router_hard > 0.5:
                    # deterministic straight-through on CLEAN logits: train-time forward == the
                    # inference program (no Gumbel corruption); router still adapts via ST gradient
                    p = F.softmax(logits, dim=-1)
                    oh = F.one_hot(logits.argmax(dim=-1), K + 1).to(p.dtype)
                    w = (oh - p).detach() + p
                else:
                    # brief early explore: Gumbel-softmax soft mixture (all blocks get gradient)
                    u = torch.rand_like(logits).clamp_min(1e-9)
                    gumbel = -torch.log(-torch.log(u))
                    w = F.softmax((logits + gumbel) / self.router_temp, dim=-1)
            else:
                w = F.one_hot(logits.argmax(dim=-1), K + 1).float()  # deterministic program
            p_clean = F.softmax(logits, dim=-1)
            probs_sum = p_clean if probs_sum is None else probs_sum + p_clean
            z = torch.logsumexp(logits, dim=-1).square().mean()
            z_acc = z if z_acc is None else z_acc + z
            _ck(f"router_w_t{t}", w)
            w = w.to(x_in.dtype)
            outs = torch.stack([
                blk(x_in, ve.get(str(k)), cos_sin, self.pool_windows[k])
                for k, blk in enumerate(self.pool)
            ])  # (K, B, T, C)
            _ck(f"pool_outs_t{t}", outs)
            # identity option leaves x untouched (true no-op -> tail of identities == early exit)
            x = w[:, K].view(B, 1, 1) * x + torch.einsum('bk,kbtc->btc', w[:, :K], outs)
            _ck(f"x_after_t{t}", x)

        x = self.resid_lambdas[-1] * x + self.x0_lambdas[-1] * x0
        x = self.block_out(x, ve["out"], cos_sin, self.win_long)
        _ck("block_out", x)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            if sample and torch.is_grad_enabled() and reduction == 'mean':
                # Load balance (Switch-style, on marginal block usage; identity excluded so
                # depth stays unconstrained) + router z-loss (ST-MoE)
                usage = probs_sum.mean(dim=0)[:K]
                usage = usage / usage.sum().clamp_min(1e-9)
                lb = K * (usage * usage).sum()
                loss = loss + LB_COEF * lb + Z_COEF * (z_acc / self.route_steps)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 96       # model_dim = depth * ASPECT_RATIO -> n_embd=768, n_head=6 (matches rsi block shape)
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "L"    # full attention on every block (most expressive; no sliding-window approximation)

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
MAX_STEPS = 2000        # step-bounded run (replaces the 5-min time budget)

# MCTS-GPT: block pool + learned per-sequence router = amortized search over block-composition
# trees (branch K+1, depth D). Recursion (block reuse) reaches depth D>K; halt = adaptive depth.
# Param-matched to baseline DEPTH=8: POOL_K=6 + fixed in/out block = 8 unique blocks.
# CPU-validated (routed-blocks/RESEARCH_NOTES.md): beats param-matched sequential baseline.
ROUTED = True            # block-pool routing vs. baseline sequential GPT
ROUTER_LEVEL = "seq"     # "seq": per-sequence router (L2, MLP on pooled h) | "static": logit table (L1)
ROUTER_INIT = "recursive"  # cycle blocks for depth-via-reuse (the winning bias); router adapts on top
ROUTER_FROZEN = False    # freeze router at init (sanity/ablation run)
POOL_K = 6               # routable blocks in the pool (+ fixed in/out = 8 unique, matches baseline)
ROUTE_STEPS = 16         # max routed depth; option K = halt/identity (no-op -> early exit)
ROUTER_LR = 0.02         # AdamW LR for router params
ROUTER_TEMP_START = 4.0  # softmax temperature at start (soft mixing warmup)
ROUTER_ANNEAL_FRAC = 0.15 # progress fraction of soft warmup, then clean deterministic straight-through
SEQ_INIT_BIAS = 4.0      # prob mass on the recursive program at init
LB_COEF = 0.003          # load-balance aux loss (marginal block usage, halt excluded)
Z_COEF = 3e-4            # router z-loss (ST-MoE)
if ROUTED:
    # dense K-block x D-step unroll retains ~K*D block activations (vs DEPTH for baseline) -> big VRAM,
    # amplified by the 768-dim (rsi-shape) blocks. Start conservative; grad-accum preserves
    # TOTAL_BATCH_SIZE. Raise in the smoke run if headroom; lower / add checkpointing if OOM.
    DEVICE_BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

wandb.init(
    project="nanogpt",
    name=RUN_NAME,
    config={
        **asdict(config),
        "depth": DEPTH,
        "device_batch_size": DEVICE_BATCH_SIZE,
        "total_batch_size": TOTAL_BATCH_SIZE,
        "matrix_lr": MATRIX_LR,
        "embedding_lr": EMBEDDING_LR,
        "unembedding_lr": UNEMBEDDING_LR,
        "scalar_lr": SCALAR_LR,
        "weight_decay": WEIGHT_DECAY,
        "window_pattern": WINDOW_PATTERN,
        "max_steps": MAX_STEPS,
        "recipe": "mcts-gpt-routed-recursive",
        "routed": ROUTED,
        "router_level": ROUTER_LEVEL,
        "router_init": ROUTER_INIT,
        "pool_k": POOL_K,
        "route_steps": ROUTE_STEPS,
        "router_lr": ROUTER_LR,
        "router_anneal_frac": ROUTER_ANNEAL_FRAC,
        "lb_coef": LB_COEF,
        "z_coef": Z_COEF,
    },
)

with torch.device("meta"):
    model = RoutedGPT(config, POOL_K, ROUTE_STEPS, ROUTER_LEVEL) if ROUTED else GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

raw_model = model  # keep uncompiled handle for buffers/router diagnostics
if not os.environ.get("DBG_NAN"):
    model = torch.compile(model, dynamic=False)

def set_router_schedule(progress):
    if not ROUTED:
        return
    frac = min(progress / ROUTER_ANNEAL_FRAC, 1.0) if ROUTER_ANNEAL_FRAC > 0 else 1.0
    temp = ROUTER_TEMP_START * (1 - frac) + 1.0 * frac
    raw_model.router_temp.fill_(temp)
    raw_model.router_hard.fill_(1.0 if frac >= 1.0 else 0.0)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Max steps: {MAX_STEPS}")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
# Running (global) loss stats for train and per-step validation
train_loss_sum = 0.0
train_loss_max = 0.0
val_loss_sum = 0.0
val_loss_max = 0.0
grad_global_norm_max = 0.0
# Cheap per-step validation: one held-out val batch per step
val_loader_stream = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules (step-based; time budget removed)
    progress = min(step / MAX_STEPS, 1.0)
    set_router_schedule(progress)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    # Gradient / parameter global norms (before the step consumes/zeros grads)
    with torch.no_grad():
        grad_global_norm = torch.norm(torch.stack([
            p.grad.detach().norm() for p in model.parameters() if p.grad is not None
        ])).item()
        param_global_norm = torch.norm(torch.stack([
            p.detach().norm() for p in model.parameters()
        ])).item()
    grad_global_norm_max = max(grad_global_norm_max, grad_global_norm)
    router_table_grad_norm = (raw_model.router.table.grad.norm().item()
                              if ROUTED and raw_model.router.table.grad is not None else 0.0)
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    if step <= 2:
        print(f"[dbg] step {step} loss={train_loss_f} grad_norm={grad_global_norm:.3e} "
              f"param_norm={param_global_norm:.3e} router_tbl_grad={router_table_grad_norm:.3e}", flush=True)

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print(f"FAIL loss={train_loss_f} grad_norm={grad_global_norm:.3e}")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Cheap per-step validation loss on one held-out batch (not counted in dt/mfu)
    torch.cuda.synchronize()
    _tv0 = time.time()
    try:
        xv, yv, _ = next(val_loader_stream)
    except StopIteration:
        val_loader_stream = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")
        xv, yv, _ = next(val_loader_stream)
    with torch.no_grad(), autocast_ctx:
        val_loss_f = model(xv, yv).item()
    torch.cuda.synchronize()
    val_ms = (time.time() - _tv0) * 1000

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    steps_left = MAX_STEPS - (step + 1)

    # Running global stats (train + val)
    train_loss_sum += train_loss_f
    train_loss_max = max(train_loss_max, train_loss_f)
    val_loss_sum += val_loss_f
    val_loss_max = max(val_loss_max, val_loss_f)
    n_logged = step + 1
    cur_lr = next(g["lr"] for g in optimizer.param_groups if g["kind"] == "muon")
    mem_active_gib = torch.cuda.max_memory_allocated() / 1024**3
    mem_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    _mem_stats = torch.cuda.memory_stats()
    mem_num_ooms = _mem_stats.get("num_ooms", 0)
    mem_num_alloc_retries = _mem_stats.get("num_alloc_retries", 0)
    update_param_ratio = (cur_lr * grad_global_norm) / (param_global_norm + 1e-12)

    router_stats = {}
    if ROUTED:
        with torch.no_grad():
            tbl = raw_model.router.table.float()
            p = F.softmax(tbl, dim=-1)
            ent = -(p * (p + 1e-9).log()).sum(-1).mean().item()
            prog = tbl.argmax(-1).tolist()
            usage = p[:, :POOL_K].mean(0)
            router_stats = {
                "router/temp": raw_model.router_temp.item(),
                "router/hard": raw_model.router_hard.item(),
                "router/entropy": ent,
                "router/identity_frac": sum(1 for j in prog if j == POOL_K) / len(prog),
                "router/usage_cv": (usage.std() / usage.mean().clamp_min(1e-9)).item(),
                "router/table_grad_norm": router_table_grad_norm,
            }
        if step % 50 == 0:
            print(f"\n  router program (argmax): {prog}  ent={ent:.3f}", flush=True)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | train: {train_loss_f:.4f} | val: {val_loss_f:.4f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | steps_left: {steps_left}    ", end="", flush=True)

    wandb.log({
        "train/loss": train_loss_f,
        "train/loss_smooth": debiased_smooth_loss,
        "val/loss": val_loss_f,
        "loss_metrics/global_avg_loss": train_loss_sum / n_logged,
        "loss_metrics/global_max_loss": train_loss_max,
        "val/global_avg_loss": val_loss_sum / n_logged,
        "val/global_max_loss": val_loss_max,
        "lr": cur_lr,
        "lr_mult": lrm,
        "mfu": mfu,
        "throughput(tps)": tok_per_sec,
        "n_tokens_seen": n_logged * TOTAL_BATCH_SIZE,
        "memory/max_active(GiB)": mem_active_gib,
        "memory/max_reserved(GiB)": mem_reserved_gib,
        "memory/num_ooms": mem_num_ooms,
        "memory/num_alloc_retries": mem_num_alloc_retries,
        "grad/global_norm": grad_global_norm,
        "grad/global_norm_max": grad_global_norm_max,
        "weights/global_param_norm": param_global_norm,
        "grad/update_param_ratio": update_param_ratio,
        "train_val_gap": debiased_smooth_loss - val_loss_f,
        "opt/muon_momentum": muon_momentum,
        "opt/muon_weight_decay": muon_weight_decay,
        "time/step_ms": dt * 1000,
        "time/val_ms": val_ms,
        "progress": progress,
        **router_stats,
    }, step=step)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 1000 == 0:
        gc.collect()

    step += 1

    # Stop after a fixed number of steps
    if step >= MAX_STEPS:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval (canonical val_bpb over EVAL_TOKENS)
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")

wandb.log({
    "val_bpb": val_bpb,
    "val/final_avg_loss": val_loss_sum / max(step, 1),
    "val/final_max_loss": val_loss_max,
    "training_seconds": total_training_time,
    "total_seconds": t_end - t_start,
    "memory/max_active(GiB)": peak_vram_mb / 1024,
    "memory/max_reserved(GiB)": torch.cuda.max_memory_reserved() / 1024**3,
    "mfu_percent": steady_state_mfu,
    "total_tokens_M": total_tokens / 1e6,
    "num_steps": step,
    "num_params_M": num_params / 1e6,
})
wandb.summary["val_bpb"] = val_bpb
wandb.finish()
