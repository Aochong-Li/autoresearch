# Copyright 2026 Recursive
# Copyright 2025 Andrej Karpathy
# SPDX-License-Identifier: Apache-2.0
"""
Nanochat pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Mean validation BPB: 0.9109 (10 seeds).
Usage: uv run train.py
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch._inductor.config as inductor_config

# Keep default inductor settings for this compile-capture ablation.
import torch.nn as nn
import torch.nn.functional as F

cap = torch.cuda.get_device_capability()

if cap[0] >= 10:
    # Blackwell (B200, SM100): wrap flash-attn-4 as a custom op so torch.compile
    # treats it as opaque (no tracing into cutlass DSL, no recompile-cache thrash,
    # no per-call Python kernel build).
    from flash_attn.cute import flash_attn_func as _fa4_raw
    from flash_attn.cute.interface import _flash_attn_bwd as _fa4_bwd_raw

    @torch.library.custom_op("fa4::fa4_causal", mutates_args=())
    def _fa4_causal_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       window_left: int) -> tuple[torch.Tensor, torch.Tensor]:
        ws = (window_left, 0) if window_left > 0 else (None, None)
        out, lse = _fa4_raw(q, k, v, causal=True, window_size=ws, return_lse=True)
        return out, lse

    @_fa4_causal_op.register_fake
    def _fa4_causal_fake(q, k, v, window_left):
        B, T, H, D = q.shape
        return torch.empty_like(q), torch.empty(B, H, T, device=q.device, dtype=torch.float32)

    def _fa4_setup_context(ctx, inputs, output):
        q, k, v, window_left = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.window_left = window_left

    @torch.library.custom_op("fa4::fa4_bwd", mutates_args=())
    def _fa4_bwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor, grad_output: torch.Tensor, lse: torch.Tensor,
                    window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wl = window_left if window_left > 0 else None
        dq, dk, dv = _fa4_bwd_raw(
            q, k, v, out, grad_output, lse,
            causal=True, window_size_left=wl, window_size_right=0,
        )
        return dq, dk, dv

    @_fa4_bwd_op.register_fake
    def _fa4_bwd_fake(q, k, v, out, grad_output, lse, window_left):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _fa4_backward(ctx, grad_output, grad_lse):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = torch.ops.fa4.fa4_bwd(q, k, v, out, grad_output, lse, ctx.window_left)
        return dq, dk, dv, None

    _fa4_causal_op.register_autograd(_fa4_backward, setup_context=_fa4_setup_context)

    def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
        wl = window_size[0] if isinstance(window_size, tuple) else window_size
        if wl is None or wl <= 0 or wl >= q.shape[1]:
            wl = -1
        out, _lse = torch.ops.fa4.fa4_causal(q, k, v, wl)
        return out

    print(f"Using flash-attn-4 as custom op (GPU capability {cap})")
else:
    # Hopper/Ampere (H100, A100): use flash-attn-3 via kernels package
    from kernels import get_kernel

    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    flash_attn_func = get_kernel(repo).flash_attn_interface.flash_attn_func
    print(f"Using flash-attn-3 from {repo} (GPU capability {cap})")

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader  # noqa: E402

import wandb
RUN_NAME = os.environ.get("RUN_NAME", "rsi")

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
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        # Separate gate for bigram VE on ALL VE layers reading decorrelated channels (32:64)
        self.bigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        # Trigram gate on layers 1, 5, and 7 for late/full-context coverage, reads channels 64:96
        ve_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        trigram_layers = (
            {ve_layers[0], ve_layers[-2], ve_layers[-1]}
            if len(ve_layers) >= 2
            else {ve_layers[-1]}
        )
        self.trigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if layer_idx in trigram_layers
            else None
        )
        # Head-level MoE gate on ALL layers for attention output routing
        self.head_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size, bigram_ve=None, trigram_ve=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Bigram VE with its own independent gate reading from decorrelated channels (32:64)
        if bigram_ve is not None:
            bigram_ve = bigram_ve.view(B, T, self.n_kv_head, self.head_dim)
            bg_gate = 2 * torch.sigmoid(self.bigram_gate(x[..., self.ve_gate_channels:2*self.ve_gate_channels]))
            v = v + bg_gate.unsqueeze(-1) * bigram_ve

        # Trigram VE with its own gate reading from channels 64:96
        if trigram_ve is not None:
            trigram_ve = trigram_ve.view(B, T, self.n_kv_head, self.head_dim)
            tg_gate = 2 * torch.sigmoid(self.trigram_gate(x[..., 2*self.ve_gate_channels:3*self.ve_gate_channels]))
            v = v + tg_gate.unsqueeze(-1) * trigram_ve

        cos, sin = cos_sin
        # QK-norm refinement: normalize BEFORE rotary instead of after
        q, k = norm(q), norm(k)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # Per-head RMSNorm on attention output (DiffTransformer-inspired sub-layer normalization)
        y = norm(y)

        # Head-level MoE: per-head routing gate on all layers
        head_gates = 2.0 * torch.sigmoid(self.head_gate(x[..., :self.ve_gate_channels]))
        y = y * head_gates.unsqueeze(-1)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        # Uniform tau=0.5: confirmed optimal threshold
        self.tau = 0.5

    def forward(self, x):
        h = self.c_fc(x)
        h = F.relu(h - self.tau).square()
        h = self.c_proj(h)
        return h


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)

    def forward(self, x, ve, cos_sin, window_size, bigram_ve=None, trigram_ve=None):
        # Simplified attention residual: per-head norm + head gate inside CSA already sufficient
        x = x + self.attn(norm(x), ve, cos_sin, window_size, bigram_ve=bigram_ve, trigram_ve=trigram_ve)
        x = x + norm(self.mlp(norm(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # JEPA MTP removed: multi-token prediction hurts step count in 5-min budget
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Input-dependent x0 gating: per-layer scale for sigmoid gate on x0 skip (layers 4+)
        # gate = 2*sigmoid(scale * x.mean(-1)) modulates x0_lambdas contribution
        # Zero-init so gate starts at 1.0 (neutral = same as current scalar behavior)
        self.x0_gate_scales = nn.Parameter(torch.zeros(config.n_layer))
        # Multi-layer output pooling: aggregate last-K intermediate layers as additive correction
        self.n_pool_layers = min(4, config.n_layer)  # layers [n-4, n-3, n-2] contribute (3 weights)
        self.layer_pool_weights = nn.Parameter(torch.zeros(self.n_pool_layers - 1))
        # Value embeddings (unigram)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        # Factored multi-hash bigram VE: K=2 half-dim tables concatenated per layer
        # Crossover: K=2 simplification recovers throughput
        ve_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        self.bigram_ve_layers = set(ve_layers)
        self.bigram_table_size = config.vocab_size * 64  # CROSSOVER B: 64x bigram tables
        self.bigram_K = 2
        half_kv_dim = kv_dim // 2
        # PER-LAYER DECORRELATED: completely disjoint hash prime pairs per bigram VE layer
        # Each layer uses entirely distinct multipliers -- zero prime reuse within bigram type
        # Constants from Murmur/FNV/golden-ratio family for good avalanche behavior
        _decorr_bigram_primes = [
            [(2654435761, 2246822519), (1013904223, 6291469)],   # layer 1: golden-ratio family
            [(374761393, 668265263), (3266489917, 104729)],      # layer 3: prime family
            [(1640531527, 97531), (48271, 40503)],               # layer 5: LCG/Knuth family
            [(16777619, 2166136261), (3432918353, 461845907)],   # layer 7: MurmurHash3 family
        ]
        self.bigram_hash_primes_per_layer = {}
        self.bigram_ves = nn.ModuleDict()
        for j, layer_i in enumerate(ve_layers):
            self.bigram_ves[str(layer_i)] = nn.ModuleList([
                nn.Embedding(self.bigram_table_size, half_kv_dim),
                nn.Embedding(self.bigram_table_size, half_kv_dim),
            ])
            self.bigram_hash_primes_per_layer[layer_i] = _decorr_bigram_primes[j]
        # Multi-layer factored trigram VE: K=2 half-dim tables at layers 1+5 plus layer 7.
        self.trigram_ve_layers = (
            {ve_layers[0], ve_layers[-2], ve_layers[-1]}
            if len(ve_layers) >= 2
            else {ve_layers[-1]}
        )
        self.trigram_table_size = config.vocab_size * 64  # CROSSOVER B: 64x trigram tables
        # PER-LAYER DECORRELATED: completely disjoint 6-prime tuples per trigram VE layer
        # Using disjoint constant families: each layer uses different multiplier sources
        _decorr_trigram_primes = [
            (16777619, 2166136261, 3432918353, 461845907, 2654435769, 1540483477),  # layer 1: FNV+Murmur family
            (3405403843, 2654435761, 2246822519, 1013904223, 6291469, 374761393),   # layer 5: golden-ratio family
            (668265263, 3266489917, 104729, 1640531527, 97531, 48271),              # layer 7: prime family
        ]
        self.trigram_hash_primes_per_layer = {}
        self.trigram_ves = nn.ModuleDict()
        for j, layer_i in enumerate(sorted(self.trigram_ve_layers)):
            self.trigram_ves[str(layer_i)] = nn.ModuleList([
                nn.Embedding(self.trigram_table_size, half_kv_dim),
                nn.Embedding(self.trigram_table_size, half_kv_dim),
            ])
            self.trigram_hash_primes_per_layer[layer_i] = _decorr_trigram_primes[j]
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
        self.x0_gate_scales.fill_(0.0)  # Zero-init: sigmoid(0)=0.5, 2*0.5=1.0 = neutral gate
        self.layer_pool_weights.fill_(0.0)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.attn.bigram_gate is not None:
                torch.nn.init.zeros_(block.attn.bigram_gate.weight)
            if block.attn.trigram_gate is not None:
                torch.nn.init.zeros_(block.attn.trigram_gate.weight)
            torch.nn.init.zeros_(block.attn.head_gate.weight)
        # Bigram VE: same init as regular VE (factored: two half-dim tables per layer)
        for layer_ves in self.bigram_ves.values():
            for bve in layer_ves:
                torch.nn.init.uniform_(bve.weight, -s, s)
                bve.to(dtype=torch.bfloat16)
        # Trigram VE init (factored: two half-dim tables per layer)
        for layer_tves in self.trigram_ves.values():
            for tve in layer_tves:
                torch.nn.init.uniform_(tve.weight, -s, s)
                tve.to(dtype=torch.bfloat16)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=1000000, device=None):
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
        assert all(c in "SLT" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        tiny_window = long_window // 4
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0), "T": (tiny_window, 0)}
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
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
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
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.layer_pool_weights.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
        ngram_ve_betas=None,  # if None, uses adam_betas
        ngram_ve_lr_scale=1.0,  # discriminative LR scale for n-gram VE (ULMFiT-inspired)
    ):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas, self.x0_gate_scales]  # gate scales grouped with x0 lambdas
        bigram_ve_params = list(self.bigram_ves.parameters())
        trigram_ve_params = list(self.trigram_ves.parameters())
        pool_params = [self.layer_pool_weights]
        assert len(list(self.parameters())) == (
            len(matrix_params)
            + len(embedding_params)
            + len(lm_head_params)
            + len(value_embeds_params)
            + len(resid_params)
            + len(x0_params)
            + len(bigram_ve_params)
            + len(trigram_ve_params)
            + len(pool_params)
        )
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if ngram_ve_betas is None:
            ngram_ve_betas = adam_betas
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            {
                "kind": "adamw",
                "params": lm_head_params,
                "lr": unembedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                "demon_beta1": True,  # Apply Demon beta1 scheduling
            },
            {
                "kind": "adamw",
                "params": embedding_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                "demon_beta1": True,
            },
            {
                "kind": "adamw",
                "params": value_embeds_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                "demon_beta1": True,
            },
            {
                "kind": "adamw",
                "params": resid_params,
                "lr": scalar_lr * 0.01,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                # No demon_beta1: scalar params keep fixed beta1
            },
            {
                "kind": "adamw",
                "params": x0_params,
                "lr": scalar_lr,
                "betas": (0.96, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.002,  # x0WD=0.002 (proven optimal)
                "is_x0_muon_warmdown": True,  # x0 Muon warmdown
            },
            {
                "kind": "rmsprop",
                "params": bigram_ve_params,
                "lr": embedding_lr * dmodel_lr_scale * ngram_ve_lr_scale,
                "beta2": ngram_ve_betas[1],
                "eps": 1e-10,
                "weight_decay": 0.0,
                "is_ngram_ve": True,
            },
            {
                "kind": "rmsprop",
                "params": trigram_ve_params,
                "lr": embedding_lr * dmodel_lr_scale * ngram_ve_lr_scale,
                "beta2": ngram_ve_betas[1],
                "eps": 1e-10,
                "weight_decay": 0.0,
                "is_ngram_ve": True,
            },
            {
                "kind": "adamw",
                "params": pool_params,
                "lr": scalar_lr * 0.15,  # revert to formula (0.75*0.15=0.1125)
                "betas": (0.96, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                {
                    "kind": "muon",
                    "params": group_params,
                    "lr": matrix_lr,
                    "momentum": 0.95,
                    "ns_steps": 5,
                    "beta2": 0.95,
                    "weight_decay": weight_decay,
                }
            )
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        # PER-LAYER DECORRELATED: precompute shifted indices (shared), compute per-layer hash indices inside loop
        prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        # Precompute per-layer bigram hash indices (different primes per layer for collision decorrelation)
        bigram_indices_per_layer = {}
        for layer_i in self.bigram_ve_layers:
            layer_bg_primes = self.bigram_hash_primes_per_layer[layer_i]
            bigram_indices_per_layer[layer_i] = [
                ((prev_idx * p1) ^ (idx * p2)) % self.bigram_table_size
                for p1, p2 in layer_bg_primes
            ]
        # Precompute per-layer trigram hash indices (different primes per layer for collision decorrelation)
        trigram_indices_per_layer = {}
        for layer_i in self.trigram_ve_layers:
            lp = self.trigram_hash_primes_per_layer[layer_i]
            trigram_indices_per_layer[layer_i] = (
                ((prev2_idx * lp[0]) ^ (prev_idx * lp[1]) ^ (idx * lp[2])) % self.trigram_table_size,
                ((prev2_idx * lp[3]) ^ (prev_idx * lp[4]) ^ (idx * lp[5])) % self.trigram_table_size,
            )
        n_layer = len(self.transformer.h)
        pool_start = n_layer - self.n_pool_layers
        pool_residual = None
        for i, block in enumerate(self.transformer.h):
            # Input-dependent x0 gate on ALL 8 layers: 2*sigmoid(scale*mean(x)) modulates x0 contribution
            # Starts at 1.0 (gate_scales=0 → sigmoid(0)=0.5 → 2*0.5=1.0)
            x0_gate = 2.0 * torch.sigmoid(self.x0_gate_scales[i] * x.float().mean(-1, keepdim=True)).to(x.dtype)
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0_gate * x0
            if str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx)
            else:
                ve = None
            # Factored multi-hash bigram VE: concat K=2 half-dim lookups from independent hashes (per-layer primes)
            if i in self.bigram_ve_layers:
                layer_ves = self.bigram_ves[str(i)]
                layer_indices = bigram_indices_per_layer[i]
                bgve = torch.cat([layer_ves[k](layer_indices[k]) for k in range(self.bigram_K)], dim=-1)
            else:
                bgve = None
            # Multi-layer factored trigram VE: concat two half-dim lookups per layer (per-layer primes)
            if i in self.trigram_ve_layers:
                tg_idx = trigram_indices_per_layer[i]
                layer_tves = self.trigram_ves[str(i)]
                tgve = torch.cat([layer_tves[0](tg_idx[0]), layer_tves[1](tg_idx[1])], dim=-1)
            else:
                tgve = None
            x = block(x, ve, cos_sin, self.window_sizes[i], bigram_ve=bgve, trigram_ve=tgve)
            if i == pool_start:
                pool_residual = self.layer_pool_weights[0] * x
            elif i == pool_start + 1:
                pool_residual = pool_residual + self.layer_pool_weights[1] * x
            elif i == pool_start + 2:
                pool_residual = pool_residual + self.layer_pool_weights[2] * x
        if pool_residual is not None:
            x = x + pool_residual
        x = norm(x)

        # Decoupled softcap in BF16: skip float() cast, halve logit tensor memory
        # Since model is natively BF16, softcap in BF16 should be numerically adequate
        logits = self.lm_head(x)
        logits = 16.5 * torch.tanh(logits / 15.0)

        if targets is not None:
            # Cast to float32 only for the CE loss computation (numerically sensitive)
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        # Eval path: need float32 logits
        return logits.float()


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
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


@torch.compile(dynamic=False, fullgraph=True)
def rmsprop_step_fused(p, grad, exp_avg_sq, step_t, lr_t, beta2_t, eps_t, wd_t):
    """RMSProp with bias correction -- no first moment, saves 50% optimizer VRAM."""
    p.mul_(1 - lr_t * wd_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(grad / denom, alpha=-lr_t)


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads,
    stacked_params,
    momentum_buffer,
    second_momentum_buffer,
    momentum_t,
    lr_t,
    wd_t,
    beta2_t,
    ns_steps,
    red_dim,
):
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
        # RMSProp CPU tensors (no beta1 -- saves first moment VRAM)
        self._rmsprop_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        step_fns = {
            "adamw": self._step_adamw,
            "rmsprop": self._step_rmsprop,
            "muon": self._step_muon,
        }
        self._step_dispatch = tuple((step_fns[group["kind"]], group) for group in self.param_groups)

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_rmsprop(self, group):
        """RMSProp: only second moment, no first moment -- 50% less optimizer VRAM for sparse tables."""
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg_sq"] = torch.zeros_like(p)
                # Note: NO exp_avg allocated -- this is the VRAM saving
            state["step"] += 1
            self._rmsprop_step_t.fill_(state["step"])
            self._rmsprop_lr_t.fill_(group["lr"])
            self._rmsprop_beta2_t.fill_(group["beta2"])
            self._rmsprop_eps_t.fill_(group["eps"])
            self._rmsprop_wd_t.fill_(group["weight_decay"])
            rmsprop_step_fused(
                p,
                grad,
                state["exp_avg_sq"],
                self._rmsprop_step_t,
                self._rmsprop_lr_t,
                self._rmsprop_beta2_t,
                self._rmsprop_eps_t,
                self._rmsprop_wd_t,
            )

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for step_fn, group in self._step_dispatch:
            step_fn(group)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 96  # model_dim = depth * ASPECT_RATIO (d8*96=768 -> dim=768, 6 heads)
HEAD_DIM = 128  # target head dimension for attention
WINDOW_PATTERN = "TTTL"  # 3 tiny + 1 long -- sandwich norm + warmdown=0.8 variant

# Optimization
TOTAL_BATCH_SIZE = 72 * 2048  # 147456 tokens per step (grad_accum=1 with devbatch=72 on B200)
EMBEDDING_LR = 0.6  # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04  # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.8  # x0 Muon warmdown SCALAR_LR=0.8
WEIGHT_DECAY = 0.1  # baseline WD
ADAM_BETAS = (0.8, 0.95)  # Adam beta1, beta2
DEMON_FINAL_BETA1 = 0.55  # baseline Demon
NGRAM_VE_BETAS = (0.5, 0.999)  # RMSProp only uses beta2=0.999; higher beta2 preserves gradient history for sparse tables
NGRAM_VE_LR_SCALE = 1.0  # RMSProp with full LR (no reduction)
WARMUP_RATIO = 0.0  # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.95  # extend warmdown trend (0.80->0.85->0.90->0.95), warmdown starts at 5%
ADAM_WARMDOWN_RATIO = 0.65  # slightly longer Adam warmdown to match extended Muon warmdown
NGRAM_WARMDOWN_RATIO = 0.0  # no warmdown for bigram/trigram VE (sparse tables benefit from full-rate training)
FINAL_LR_FRAC = 0.05  # restored FLR=0.05

# Model size
DEPTH = 8  # number of transformer layers
DEVICE_BATCH_SIZE = 72  # per-device batch size -- B200
MAX_STEPS = 2000        # step-bounded run (replaces the 5-min time budget)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
_SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(_SEED)
torch.cuda.manual_seed(_SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
# No autocast: model is natively BF16 -- eliminates FP32->BF16 cast overhead in compile graph
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False)
B200_BF16_PEAK_FLOPS = 989.5e12  # H100 bf16 peak (running on H100, not B200)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")


def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

wandb.init(
    project="nanogpt",
    name=RUN_NAME,
    config={
        **asdict(config),
        "recipe": "recursive-optimized-from-karpathy",
        "depth": DEPTH,
        "device_batch_size": DEVICE_BATCH_SIZE,
        "total_batch_size": TOTAL_BATCH_SIZE,
        "window_pattern": WINDOW_PATTERN,
        "max_steps": MAX_STEPS,
    },
)

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
# Cast entire model to BF16: enables removing autocast, simplifies compile graph
model.to(dtype=torch.bfloat16)

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts["total"]
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
    ngram_ve_betas=NGRAM_VE_BETAS,
    ngram_ve_lr_scale=NGRAM_VE_LR_SCALE,
)

muon_groups = []
ngram_groups = []
x0_warmdown_groups = []
adam_groups = []
adam_demon_groups = []
muon_group_lrs = []
x0_group_lrs = []
adam_group_lrs = []
for group in optimizer.param_groups:
    if group["kind"] == "muon":
        muon_groups.append(group)
        muon_group_lrs.append((group, group["initial_lr"]))
    elif group.get("is_ngram_ve", False):
        ngram_groups.append(group)
    elif group.get("is_x0_muon_warmdown", False):
        x0_warmdown_groups.append(group)
        x0_group_lrs.append((group, group["initial_lr"]))
    else:
        adam_groups.append(group)
        adam_group_lrs.append((group, group["initial_lr"]))
        if group.get("demon_beta1", False):
            adam_demon_groups.append((group, group["betas"][1]))

model = torch.compile(model, dynamic=False, mode="max-autotune", fullgraph=True)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)


def get_lr_multiplier(progress, warmdown_ratio=WARMDOWN_RATIO):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


MUON_PEAK_MOMENTUM = 0.95  # standard peak
MUON_WARMDOWN_MOMENTUM = 0.79  # testing VE beta2 ramp alone
# Reverse Demon for NorMuon beta2: INCREASE beta2 during warmdown for more stable variance normalization
MUON_BETA2_PEAK = 0.95  # standard beta2 during full-LR phase
MUON_BETA2_WARMDOWN = 0.97  # target beta2 at end of warmdown
MUON_LR_BOOST = 1.0  # no LR boost
# VE RMSProp reverse-Demon: increase VE beta2 during last 30% of Muon warmdown
# Analogous to Muon's 0.95->0.97, but for ngram VE tables (0.999->0.9995)
NGRAM_VE_BETA2_WARMDOWN = 0.9999  # STRONGER delayed VE beta2 ramp (0.999->0.9999 last 30% warmdown)
def get_muon_momentum(step, progress=None):
    # Warmup: 0.85 -> 0.95 over 300 steps
    frac = min(step / 300, 1)
    base = (1 - frac) * 0.85 + frac * MUON_PEAK_MOMENTUM
    # Quadratic Demon: back-loaded shape keeps peak momentum longer
    if progress is not None:
        warmdown_start = 1.0 - WARMDOWN_RATIO
        if progress > warmdown_start:
            wd_frac = (progress - warmdown_start) / WARMDOWN_RATIO
            base = MUON_PEAK_MOMENTUM + (wd_frac ** 2) * (MUON_WARMDOWN_MOMENTUM - MUON_PEAK_MOMENTUM)
    return base


def get_muon_beta2(progress):
    """Reverse beta2: increase beta2 during warmdown for more stable variance norm."""
    warmdown_start = 1.0 - WARMDOWN_RATIO
    if progress < warmdown_start:
        return MUON_BETA2_PEAK
    else:
        wd_frac = (progress - warmdown_start) / WARMDOWN_RATIO
        return MUON_BETA2_PEAK + wd_frac * (MUON_BETA2_WARMDOWN - MUON_BETA2_PEAK)


def get_muon_lr_boost(progress):
    """Boost Muon LR during warmdown to compensate for higher beta2 reducing step size."""
    warmdown_start = 1.0 - WARMDOWN_RATIO
    if progress < warmdown_start:
        return 1.0
    else:
        wd_frac = (progress - warmdown_start) / WARMDOWN_RATIO
        return 1.0 + wd_frac * (MUON_LR_BOOST - 1.0)


def get_adam_beta1(progress, warmdown_ratio=ADAM_WARMDOWN_RATIO):
    """Forward Demon: decrease beta1 during warmdown for more responsive gradient following."""
    initial_beta1 = ADAM_BETAS[0]
    final_beta1 = DEMON_FINAL_BETA1
    warmdown_start = 1.0 - warmdown_ratio
    if progress < warmdown_start:
        return initial_beta1
    else:
        warmdown_progress = (progress - warmdown_start) / warmdown_ratio
        return initial_beta1 + (final_beta1 - initial_beta1) * warmdown_progress


# WD pulse: RECTANGULAR shape -- with 95% warmdown (starts at 5%), pulses shifted earlier
# Main pulse at 3% center, 2% total duration (1% half-width): fires at 2-4%, before warmdown onset at 5%
# Early pulse at 1.5% center, 1% total duration: fires at 1-2% progress
# Both pulses fire in the full-LR phase (0-5%), maintaining the pre-warmdown regularization timing
WD_PULSE_CENTER = 0.03   # shift main pulse to 3% (fires before warmdown at 5%)
WD_PULSE_HALF_WIDTH = 0.01  # 1% half-width: 2% total duration (tighter for earlier firing)
WD_PULSE_MAGNITUDE = 5.0  # try 5x main pulse (vs 8x) -- 5x optimal WITH Muon Demon, 8x WITHOUT; current setup HAS Demon
WD_EARLY_PULSE_CENTER = 0.015  # shift early pulse to 1.5%
WD_EARLY_PULSE_HALF_WIDTH = 0.005  # 0.5% half-width: 1% total duration
WD_EARLY_PULSE_MAGNITUDE = 3.0  # 3x early pulse (gentler, to initialize regularization)
# Mid-warmdown triangular pulse: fires at 80% total progress (= ~79% through warmdown)
# This is WITHIN the VE beta2 ramp zone (which starts at 71.5% total = 70% through warmdown)
# Hypothesis: VE beta2 stabilization provides a safety net for a mid-warmdown WD perturbation
WD_MID_PULSE_CENTER = 0.80   # 80% total progress = ~79% through warmdown
WD_MID_PULSE_HALF_WIDTH = 0.025  # 2.5% half-width: 5% total triangular duration
WD_MID_PULSE_MAGNITUDE = 4.0  # 4x magnitude (triangular shape -- less harsh than rectangular)

def get_weight_decay(progress):
    base_wd = WEIGHT_DECAY * (1 - progress)
    # Early small pulse: 3x spike at 2% progress (step ~65), 2% total duration
    early_dist = abs(progress - WD_EARLY_PULSE_CENTER)
    if early_dist < WD_EARLY_PULSE_HALF_WIDTH:
        return base_wd * WD_EARLY_PULSE_MAGNITUDE  # RECTANGULAR early pulse
    # Main pulse: 8x rectangular spike at 5% progress (step ~163), 3% total duration
    dist = abs(progress - WD_PULSE_CENTER)
    if dist < WD_PULSE_HALF_WIDTH:
        return base_wd * WD_PULSE_MAGNITUDE  # RECTANGULAR main pulse
    # Mid-warmdown triangular pulse: fires within VE beta2 stabilization zone
    mid_dist = abs(progress - WD_MID_PULSE_CENTER)
    if mid_dist < WD_MID_PULSE_HALF_WIDTH:
        # Triangular: linear ramp up then down (proven optimal shape)
        local = (progress - (WD_MID_PULSE_CENTER - WD_MID_PULSE_HALF_WIDTH)) / (2 * WD_MID_PULSE_HALF_WIDTH)
        bump = 2 * local if local < 0.5 else 2 * (1 - local)
        return base_wd * (1.0 + bump * (WD_MID_PULSE_MAGNITUDE - 1.0))
    return base_wd


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
inv_time_budget = 1.0 / TIME_BUDGET
inv_muon_warmdown = 1.0 / WARMDOWN_RATIO
inv_adam_warmdown = 1.0 / ADAM_WARMDOWN_RATIO
muon_warmdown_start = 1.0 - WARMDOWN_RATIO
adam_warmdown_start = 1.0 - ADAM_WARMDOWN_RATIO
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
    for _micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules (step-based; time budget removed)
    progress = min(step / MAX_STEPS, 1.0)
    if progress < muon_warmdown_start:
        lrm_muon = 1.0
        muon_wd_frac = 0.0
    else:
        muon_wd_frac = (progress - muon_warmdown_start) * inv_muon_warmdown
        lrm_muon = ((1.0 - progress) * inv_muon_warmdown) * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC

    if progress < adam_warmdown_start:
        lrm_adam = 1.0
        adam_beta1 = ADAM_BETAS[0]
    else:
        adam_wd_frac = (progress - adam_warmdown_start) * inv_adam_warmdown
        lrm_adam = ((1.0 - progress) * inv_adam_warmdown) * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC
        adam_beta1 = ADAM_BETAS[0] + (DEMON_FINAL_BETA1 - ADAM_BETAS[0]) * adam_wd_frac

    frac = min(step / 300, 1)
    muon_momentum = (1 - frac) * 0.85 + frac * MUON_PEAK_MOMENTUM
    if progress > muon_warmdown_start:
        muon_momentum = MUON_PEAK_MOMENTUM + (muon_wd_frac ** 2) * (MUON_WARMDOWN_MOMENTUM - MUON_PEAK_MOMENTUM)
    muon_beta2 = MUON_BETA2_PEAK + muon_wd_frac * (MUON_BETA2_WARMDOWN - MUON_BETA2_PEAK)
    muon_lr_boost = 1.0 + muon_wd_frac * (MUON_LR_BOOST - 1.0)
    # VE RMSProp reverse-Demon: DELAYED ramp (only last 30% of Muon warmdown)
    late_frac = max(0.0, (muon_wd_frac - 0.7) / 0.3)
    ve_beta2 = NGRAM_VE_BETAS[1] + late_frac * (NGRAM_VE_BETA2_WARMDOWN - NGRAM_VE_BETAS[1])

    base_wd = WEIGHT_DECAY * (1 - progress)
    early_dist = abs(progress - WD_EARLY_PULSE_CENTER)
    if early_dist < WD_EARLY_PULSE_HALF_WIDTH:
        muon_weight_decay = base_wd * WD_EARLY_PULSE_MAGNITUDE
    else:
        dist = abs(progress - WD_PULSE_CENTER)
        if dist < WD_PULSE_HALF_WIDTH:
            muon_weight_decay = base_wd * WD_PULSE_MAGNITUDE
        else:
            mid_dist = abs(progress - WD_MID_PULSE_CENTER)
            if mid_dist < WD_MID_PULSE_HALF_WIDTH:
                local = (progress - (WD_MID_PULSE_CENTER - WD_MID_PULSE_HALF_WIDTH)) / (2 * WD_MID_PULSE_HALF_WIDTH)
                bump = 2 * local if local < 0.5 else 2 * (1 - local)
                muon_weight_decay = base_wd * (1.0 + bump * (WD_MID_PULSE_MAGNITUDE - 1.0))
            else:
                muon_weight_decay = base_wd

    muon_lr = lrm_muon * muon_lr_boost
    if progress < muon_warmdown_start:
        for group in muon_groups:
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            group["beta2"] = muon_beta2
    else:
        for group, initial_lr in muon_group_lrs:
            group["lr"] = initial_lr * muon_lr
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            group["beta2"] = muon_beta2
        for group, initial_lr in x0_group_lrs:
            group["lr"] = initial_lr * lrm_muon
    if progress >= adam_warmdown_start:
        for group, initial_lr in adam_group_lrs:
            group["lr"] = initial_lr * lrm_adam
        for group, beta2 in adam_demon_groups:
            group["betas"] = (adam_beta1, beta2)
    # Update ngram VE RMSProp beta2 during warmdown (delayed reverse-Demon for sparse tables)
    if progress >= muon_warmdown_start and late_frac > 0.0:
        for group in ngram_groups:
            group["beta2"] = ve_beta2
    # Gradient / parameter global norms (before the step consumes/zeros grads)
    with torch.no_grad():
        grad_global_norm = torch.norm(torch.stack([
            p.grad.detach().norm() for p in model.parameters() if p.grad is not None
        ])).item()
        param_global_norm = torch.norm(torch.stack([
            p.detach().norm() for p in model.parameters()
        ])).item()
    grad_global_norm_max = max(grad_global_norm_max, grad_global_norm)
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / B200_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm_muon: {lrm_muon:.2f} lrm_adam: {lrm_adam:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

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

    # Running global stats (train + val)
    train_loss_sum += train_loss_f
    train_loss_max = max(train_loss_max, train_loss_f)
    val_loss_sum += val_loss_f
    val_loss_max = max(val_loss_max, val_loss_f)
    n_logged = step + 1
    mem_active_gib = torch.cuda.max_memory_allocated() / 1024**3
    mem_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    _mem_stats = torch.cuda.memory_stats()
    mem_num_ooms = _mem_stats.get("num_ooms", 0)
    mem_num_alloc_retries = _mem_stats.get("num_alloc_retries", 0)
    cur_lr = max((g["lr"] for g in optimizer.param_groups), default=0.0)
    update_param_ratio = (cur_lr * grad_global_norm) / (param_global_norm + 1e-12)
    wandb.log({
        "train/loss": train_loss_f,
        "train/loss_smooth": debiased_smooth_loss,
        "val/loss": val_loss_f,
        "loss_metrics/global_avg_loss": train_loss_sum / n_logged,
        "loss_metrics/global_max_loss": train_loss_max,
        "val/global_avg_loss": val_loss_sum / n_logged,
        "val/global_max_loss": val_loss_max,
        "lr_mult_muon": lrm_muon,
        "lr_mult_adam": lrm_adam,
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
        "opt/muon_beta2": muon_beta2,
        "opt/muon_weight_decay": muon_weight_decay,
        "opt/adam_beta1": adam_beta1,
        "time/step_ms": dt * 1000,
        "time/val_ms": val_ms,
        "progress": progress,
    }, step=step)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Stop after a fixed number of steps
    if step >= MAX_STEPS:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = (
    100
    * num_flops_per_token
    * TOTAL_BATCH_SIZE
    * (step - 10)
    / total_training_time
    / B200_BF16_PEAK_FLOPS
    if total_training_time > 0
    else 0
)
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




