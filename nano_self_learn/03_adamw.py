"""Module 3 — AdamW by hand.

You implement AdamW.step(). The update, per parameter p with grad g:
    m = b1*m + (1-b1)*g                      # 1st moment (momentum)
    v = b2*v + (1-b2)*g*g                     # 2nd moment (variance)
    m_hat = m / (1 - b1**t) ;  v_hat = v / (1 - b2**t)     # bias correction
    p <- p * (1 - lr*wd)                      # DECOUPLED weight decay (the "W")
    p <- p - lr * m_hat / (sqrt(v_hat) + eps)
Keep m, v in self.state[p]; step count t per-param.

Run:  uv run nano_self_learn/03_adamw.py
Gate: your optimizer matches torch.optim.AdamW to < 1e-5 after 50 steps.
"""
from __future__ import annotations
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:  # lazy init
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                # TODO(you): implement the 5 update lines from the docstring using st["m"], st["v"], st["t"].
                raise NotImplementedError


def _clone_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))


if __name__ == "__main__":
    torch.manual_seed(0)
    xs = torch.randn(64, 16); ys = torch.randn(64, 1)
    ma, mb = _clone_model(), _clone_model()
    mb.load_state_dict(ma.state_dict())
    oa = torch.optim.AdamW(ma.parameters(), lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    ob = AdamW(mb.parameters(), lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    for _ in range(50):
        for m, o in ((ma, oa), (mb, ob)):
            loss = ((m(xs) - ys) ** 2).mean()
            o.zero_grad(); loss.backward(); o.step()
    max_diff = max((pa - pb).abs().max().item() for pa, pb in zip(ma.parameters(), mb.parameters()))
    print(f"max param diff vs torch.optim.AdamW after 50 steps: {max_diff:.2e}")
    print("PASS ✅" if max_diff < 1e-5 else "FAIL ❌ — check bias-correction and decoupled weight decay")
