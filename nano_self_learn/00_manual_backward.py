"""Module 0 — one backward pass BY HAND (ground the chain rule, then trust torch.autograd).

Graph:   d = a * b ;  e = d + c ;  f = tanh(e)
You compute df/da, df/db, df/dc analytically (chain rule), and we check vs torch.autograd.

Chain rule reminders:
    df/de = 1 - tanh(e)**2          (derivative of tanh)
    e = d + c   ->  de/dd = 1, de/dc = 1
    d = a * b   ->  dd/da = b, dd/db = a
Compose: df/da = df/de * de/dd * dd/da,  etc.

Run:  uv run nano_self_learn/00_manual_backward.py
"""
from __future__ import annotations
import math
import torch


def forward(a: float, b: float, c: float) -> float:
    return math.tanh(a * b + c)


def backward_by_hand(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Return (df/da, df/db, df/dc) using the chain rule above. No torch autograd here."""
    # TODO(you): compute the three partial derivatives by hand.
    raise NotImplementedError


def _torch_grads(a, b, c):
    ta, tb, tc = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (a, b, c))
    f = torch.tanh(ta * tb + tc)
    f.backward()
    return ta.grad.item(), tb.grad.item(), tc.grad.item()


if __name__ == "__main__":
    a, b, c = 1.5, -2.0, 0.5
    mine = backward_by_hand(a, b, c)
    ref = _torch_grads(a, b, c)
    ok = all(abs(m - r) < 1e-9 for m, r in zip(mine, ref))
    print(f"yours={tuple(round(x,6) for x in mine)}  torch={tuple(round(x,6) for x in ref)}")
    print("PASS ✅" if ok else "FAIL ❌ — check the chain-rule composition")
