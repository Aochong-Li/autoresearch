"""Given helper (NOT a TODO): a 5-line char tokenizer + tiny data so we spend the
2 hours on the model/optimizers, not on data plumbing.

Usage:
    from data import CharTokenizer, get_batch, TEXT
    tok = CharTokenizer(TEXT)
    xb, yb = get_batch(tok.encode(TEXT), batch_size=8, block_size=64, device="cuda")
"""
from __future__ import annotations
import torch

# A tiny in-repo corpus. Swap TEXT for a real file later (open("input.txt").read()).
TEXT = (
    "to be, or not to be, that is the question:\n"
    "whether 'tis nobler in the mind to suffer\n"
    "the slings and arrows of outrageous fortune,\n"
    "or to take arms against a sea of troubles\n"
    "and by opposing end them.\n"
) * 200


class CharTokenizer:
    """Character-level tokenizer: vocab = sorted unique chars."""
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str = "cpu"):
    """Sample a batch of (x, y) where y is x shifted by one (next-token targets).
    Shapes: x,y -> (batch_size, block_size), both long tensors on `device`.
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
