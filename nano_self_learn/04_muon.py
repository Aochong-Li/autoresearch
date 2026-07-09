"""Module 4 — Muon by hand (Newton–Schulz orthogonalization).

Idea: for a 2-D weight matrix, the raw momentum update points in a good direction but has a
skewed spectrum. Muon replaces it with its ORTHOGONALIZATION — same direction, singular values
pushed toward 1 — via a few Newton–Schulz iterations (no SVD needed).

You implement:
  1) newton_schulz(G, steps)  — the quintic iteration (coeffs given)
  2) Muon.step()              — momentum buffer -> orthogonalize -> update
Hybrid rule (given as guidance): Muon is for 2-D matrices only; embeddings/scalars/biases use AdamW.

Run:  uv run nano_self_learn/04_muon.py
Gate: NS output has singular values ≈ 1, AND Muon trains the tiny model (A/B vs AdamW printed).
"""
from __future__ import annotations
import importlib.util, pathlib
import torch


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize G (2-D) via the quintic Newton–Schulz iteration.
    a, b, c = 3.4445, -4.7750, 2.0315
    Normalize:  X = G / (||G||_F + 1e-7);  if rows>cols work on X.T (then transpose back).
    Each step:  A = X @ X.T ;  B = b*A + c*(A@A) ;  X = a*X + B @ X
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transpose = G.shape[0] > G.shape[1]
    if transpose:
        X = X.T
    # TODO(you): normalize X, run `steps` iterations of the quintic update above.
    raise NotImplementedError
    # if transpose: X = X.T
    # return X


class Muon(torch.optim.Optimizer):
    """Muon for 2-D params only. lr, momentum(beta), and ns_steps."""
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, ns = group["lr"], group["momentum"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, "Muon is for 2-D matrices; route others to AdamW"
                st = self.state[p]
                if not st:
                    st["buf"] = torch.zeros_like(p)
                # TODO(you):
                #   buf = st["buf"];  buf.mul_(mom).add_(p.grad)          # momentum
                #   upd = newton_schulz(buf, ns)                          # orthogonalize
                #   scale = max(1.0, p.size(0)/p.size(1)) ** 0.5          # rectangular correction
                #   p.add_(upd, alpha=-lr * scale)
                raise NotImplementedError


def _mlp():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(16, 64, bias=False), torch.nn.GELU(), torch.nn.Linear(64, 1, bias=False))


if __name__ == "__main__":
    torch.manual_seed(0)
    # Test 1: orthogonality — singular values should be ≈ 1
    G = torch.randn(64, 32)
    O = newton_schulz(G, 5)
    sv = torch.linalg.svdvals(O.float())
    print(f"NS singular values: min={sv.min():.3f} max={sv.max():.3f} (want ≈1)")
    ortho_ok = sv.min() > 0.7 and sv.max() < 1.3

    # Test 2: A/B — Muon vs AdamW on the same tiny regression
    xs = torch.randn(128, 16); ys = torch.randn(128, 1)
    def run(opt_factory):
        m = _mlp(); o = opt_factory(m)
        init = ((m(xs) - ys) ** 2).mean().item()
        for _ in range(100):
            loss = ((m(xs) - ys) ** 2).mean(); o.zero_grad(); loss.backward(); o.step()
        return init, loss.item()
    i_a, f_a = run(lambda m: torch.optim.AdamW(m.parameters(), lr=1e-2))
    i_m, f_m = run(lambda m: Muon(m.parameters(), lr=2e-2))
    print(f"AdamW: {i_a:.3f} -> {f_a:.3f}   |   Muon: {i_m:.3f} -> {f_m:.3f}")
    trains = f_m < 0.5 * i_m
    print("PASS ✅" if (ortho_ok and trains) else "FAIL ❌ — check the NS iteration / Muon step")
