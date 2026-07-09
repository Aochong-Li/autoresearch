"""Module 0b — backward through a transformer BLOCK by hand (attention + MLP + residual).

You derive/implement TWO backwards and verify vs torch.autograd:
  1) attention_backward(dO, Q, K, V)  -> dQ, dK, dV
  2) mlp_backward(dm, z, W1, W2)       -> dz, dW1, dW2   (activation = relu(x)**2, like nanochat)
Then a GIVEN 3-layer residual demo shows why the *block index i* matters.

Simplifications (you said keep it simple): single head, no batch, no mask, no LayerNorm.
Shapes: T=seq, d=head/model dim, h=mlp hidden.  Q,K,V,O,z: (T,d).  W1:(h,d)  W2:(d,h).

--- The math you need ---
Attention:  S = Q Kᵀ / √d ;  A = softmax(S, dim=-1) ;  O = A V
  dV = Aᵀ dO
  dA = dO Vᵀ
  softmax backward (per row):  dS = A ⊙ ( dA − rowsum(dA ⊙ A) )      # the key formula
  dQ = (dS K) / √d ;   dK = (dSᵀ Q) / √d

MLP:  pre = z W1ᵀ ;  hid = relu(pre)**2 ;  m = hid W2ᵀ
  dW2 = dmᵀ hid ;  dhid = dm W2
  dpre = dhid ⊙ (2·relu(pre))      # d/dx relu(x)² = 2·relu(x)
  dW1 = dpreᵀ z ;  dz = dpre W1

Run:  uv run nano_self_learn/00b_block_backward.py
Gate: both backwards match torch.autograd to < 1e-4.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def attention_forward(Q, K, V):
    d = Q.size(-1)
    S = (Q @ K.transpose(-2, -1)) / math.sqrt(d)
    A = torch.softmax(S, dim=-1)
    return A @ V, A


def attention_backward(dO, Q, K, V):
    """Return dQ, dK, dV. Recompute A with attention_forward, then apply the formulas above."""
    _, A = attention_forward(Q, K, V)
    d = Q.size(-1)
    # TODO(you): dV, dA, dS (softmax backward), dQ, dK
    raise NotImplementedError


def mlp_forward(z, W1, W2):
    pre = z @ W1.T
    hid = F.relu(pre) ** 2
    return hid @ W2.T, pre, hid


def mlp_backward(dm, z, W1, W2):
    """Return dz, dW1, dW2 for m = relu(z W1ᵀ)² W2ᵀ."""
    _, pre, hid = mlp_forward(z, W1, W2)
    # TODO(you): dW2, dhid, dpre (use 2*relu(pre)), dW1, dz
    raise NotImplementedError


# ---- GIVEN: why block index i matters (residual gradient highway) ----
def residual_stack_demo():
    """3 residual blocks: x_{i+1} = x_i + F_i(x_i).  Show that dL/dx_0 carries a direct copy of
    dL/dx_3 (the identity path) PLUS cross terms — so block 0's grad = 'sum of everything downstream'.
    dL/dx_i = dL/dx_{i+1} @ (I + J_{F_i}).  Unroll i=2,1,0."""
    torch.manual_seed(0)
    T, d = 4, 8
    x0 = torch.randn(T, d, requires_grad=True)
    Ws = [torch.randn(d, d, requires_grad=True) * 0.1 for _ in range(3)]
    x = x0
    xs = [x]
    for W in Ws:
        x = x + torch.tanh(x @ W)   # F_i(x) = tanh(xW)
        xs.append(x)
    loss = xs[-1].sum()
    loss.backward()
    print("‖dL/dx0‖ =", round(x0.grad.norm().item(), 4),
          "  (identity path alone would give ‖1‖ =", round(math.sqrt(T * d), 4), ")")
    print("-> residual skip means grad reaches block 0 undiluted + cross-terms from blocks 1,2")


if __name__ == "__main__":
    torch.manual_seed(0)
    T, d, h = 5, 8, 16
    Q, K, V = (torch.randn(T, d, requires_grad=True) for _ in range(3))
    z = torch.randn(T, d, requires_grad=True)
    W1 = torch.randn(h, d, requires_grad=True)
    W2 = torch.randn(d, h, requires_grad=True)

    # attention check
    O, _ = attention_forward(Q, K, V)
    dO = torch.randn_like(O)
    O.backward(dO)
    dQ, dK, dV = attention_backward(dO, Q.detach(), K.detach(), V.detach())
    attn_ok = all(torch.allclose(a, b, atol=1e-4) for a, b in [(dQ, Q.grad), (dK, K.grad), (dV, V.grad)])
    print(f"attention backward matches torch: {attn_ok}")

    # mlp check
    m, _, _ = mlp_forward(z, W1, W2)
    dm = torch.randn_like(m)
    m.backward(dm)
    dz, dW1, dW2 = mlp_backward(dm, z.detach(), W1.detach(), W2.detach())
    mlp_ok = all(torch.allclose(a, b, atol=1e-4) for a, b in [(dz, z.grad), (dW1, W1.grad), (dW2, W2.grad)])
    print(f"mlp backward matches torch: {mlp_ok}")

    print("--- residual highway demo ---")
    residual_stack_demo()
    print("PASS ✅" if (attn_ok and mlp_ok) else "FAIL ❌ — check the softmax backward / relu² derivative")
