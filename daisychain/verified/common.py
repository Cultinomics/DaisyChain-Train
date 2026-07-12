"""Shared helpers for verified neural units.

Same discipline as the neural-aarch64 / neural-photonic / neural-ddr units:
a small MLP is trained until it is *bit-identical to a golden reference over its
entire finite input domain* (N/N verification).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def bits_of(v: int, n: int) -> torch.Tensor:
    """LSB-first bit vector of length n."""
    return torch.tensor([(v >> i) & 1 for i in range(n)], dtype=torch.float32)


def int_of(bits: torch.Tensor) -> int:
    """Inverse of bits_of: LSB-first bit vector -> int."""
    v = 0
    for i, b in enumerate(bits.tolist()):
        if b >= 0.5:
            v |= (1 << i)
    return v


def pm(bits: torch.Tensor) -> torch.Tensor:
    """Map {0,1} bits to {-1,+1} for a friendlier input scale."""
    return bits * 2.0 - 1.0


def mlp(inp: int, out: int, h: int = 128, layers: int = 3) -> nn.Sequential:
    mods: list[nn.Module] = [nn.Linear(inp, h), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(h, h), nn.ReLU()]
    mods += [nn.Linear(h, out)]
    return nn.Sequential(*mods)


@torch.no_grad()
def verify(net: nn.Module, X: torch.Tensor, Ybits: torch.Tensor) -> tuple[int, int]:
    """Return (n_correct, n_total) over the full enumerated domain."""
    net.eval()
    pred = (net(X) > 0).float()
    ok = (pred == Ybits).all(dim=1).sum().item()
    return int(ok), X.shape[0]


def train(net: nn.Module, X: torch.Tensor, Ybits: torch.Tensor,
          steps: int = 4000, lr: float = 2e-3, tag: str = "") -> nn.Module:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    net.train()
    for s in range(steps):
        opt.zero_grad()
        loss = lossf(net(X), Ybits)
        loss.backward()
        opt.step()
        if tag and (s % 1000 == 0 or s == steps - 1):
            ok, tot = verify(net, X, Ybits)
            net.train()
            print(f"  [{tag}] step {s:5d}  loss {loss.item():.5f}  verify {ok}/{tot}")
    return net
