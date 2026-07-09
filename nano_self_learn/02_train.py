"""Module 2 — the training pipeline.

You implement:
  1) get_lr(step)   — linear warmup then cosine decay
  2) the training loop — forward -> loss.backward() -> clip -> opt.step() -> zero_grad
     (with bf16 autocast on GPU)

Everything else (data, model load, eval, sampling) is given.

Run:  uv run nano_self_learn/02_train.py
Gate: val loss drops well below the init (~ln(vocab)); the sample looks word-like.
"""
from __future__ import annotations
import math, importlib.util, pathlib
import torch
from data import CharTokenizer, get_batch, TEXT


def _load(name):
    p = pathlib.Path(__file__).parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("0", "m"), p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

gpt = _load("01_transformer")
GPT, GPTConfig = gpt.GPT, gpt.GPTConfig

# --- config ---
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, WARMUP, LR, BS, BLOCK = 800, 50, 3e-3, 32, 64


def get_lr(step: int) -> float:
    """Linear warmup for WARMUP steps up to LR, then cosine decay to ~0 over STEPS."""
    # TODO(you): warmup:  LR * step/WARMUP  ;  then cosine: 0.5*LR*(1+cos(pi*progress))
    raise NotImplementedError


@torch.no_grad()
def evaluate(model, data, n=50):  # GIVEN
    model.eval(); losses = []
    for _ in range(n):
        xb, yb = get_batch(data, BS, BLOCK, DEV)
        _, loss = model(xb, yb); losses.append(loss.item())
    model.train(); return sum(losses) / len(losses)


@torch.no_grad()
def sample(model, tok, n=200):  # GIVEN
    model.eval(); idx = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    for _ in range(n):
        logits, _ = model(idx[:, -BLOCK:])
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    model.train(); return tok.decode(idx[0].tolist())


def main():
    torch.manual_seed(0)
    tok = CharTokenizer(TEXT)
    data = tok.encode(TEXT).to(DEV)
    n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=BLOCK)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16) if DEV == "cuda" else torch.autocast("cpu", enabled=False)

    print(f"init val loss {evaluate(model, val_data):.3f}  (~ln(vocab)={math.log(tok.vocab_size):.3f})")
    for step in range(STEPS):
        # TODO(you): the training step —
        #   for g in opt.param_groups: g["lr"] = get_lr(step)
        #   xb, yb = get_batch(train_data, BS, BLOCK, DEV)
        #   with autocast: _, loss = model(xb, yb)
        #   opt.zero_grad(); loss.backward()
        #   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #   opt.step()
        raise NotImplementedError

    vloss = evaluate(model, val_data)
    print(f"final val loss {vloss:.3f}")
    print("--- sample ---\n" + sample(model, tok))
    print("PASS ✅" if vloss < 2.0 else "FAIL ❌ — val loss didn't drop enough; check loop/schedule")


if __name__ == "__main__":
    main()
