# nano-self-learn — handwrite a transformer + training + AdamW + Muon (2h)

You write the core; I scaffold + review. Each module is a **self-testing file**: fill the
`# TODO(you)` blocks, then run it — it prints `PASS`/`FAIL` for that module's gate.

```bash
uv run nano_self_learn/00_manual_backward.py   # etc.
```

The real nanochat implementation lives in `../train.py` (this repo's `master`) — peek there only
*after* you've attempted a piece, to compare.

## Agenda (time-boxed)

| # | File | You implement | Done gate |
|---|------|---------------|-----------|
| 0 | `00_manual_backward.py` | one backward pass by hand on a 3-node graph | matches `torch.autograd` grads |
| 1 | `01_transformer.py` | attention → MHA → block → GPT | shapes ok + **overfit 1 batch → loss≈0** |
| 2 | `02_train.py` | the training loop + CE loss + schedule | val loss drops, sample readable |
| 3 | `03_adamw.py` | AdamW update (moments, bias-corr, decoupled WD) | matches `torch.optim.AdamW` |
| 4 | `04_muon.py` | Newton–Schulz orthogonalize + hybrid | NS≈orthogonal + Muon-vs-AdamW A/B |
| 5* | `05_optional_qknorm_swa.py` | QK-norm + sliding-window (optional) | each A/B vs base block |

`data.py` is given (char tokenizer + tiny corpus) — don't edit it.

## The loop for each block
1. Read the file's docstring (concept + math).
2. Implement the `# TODO(you)` bodies.
3. Run the file → green gate.
4. I review your code and we go deeper.

## Concepts, in one line each
- **Attention**: `softmax(QKᵀ/√d + causal_mask) V` — a data-dependent weighted average of values.
- **Block (pre-norm)**: `x = x + attn(norm(x)); x = x + mlp(norm(x))`.
- **Loss**: cross-entropy of next-token logits vs targets (= −log p of the true token).
- **AdamW**: per-param adaptive step `m/(√v+ε)` with bias-correction; weight decay applied *separately*.
- **Muon**: for 2-D weight matrices, replace the raw momentum update with its *orthogonalization*
  (Newton–Schulz) — take the "direction" but normalize the singular values toward 1.

Log findings + A/B numbers in `LEARNING_LOG.md`.
