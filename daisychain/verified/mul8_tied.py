"""TIED neural multiplier -- one small cell, iterated (a la neural-raytracing).

Multiply is inherently iterative (shift-and-add), so instead of four separate
4x4 atoms we use a SINGLE weight-tied cell applied across the 8 steps -- exactly
the raytracing pattern of marching one shared cell rather than stacking many.

The tied cell is the conditional-add step of a shift-add multiplier:

    cell(acc, a, enable) = acc + (a if enable else 0)      # 8+8+1 -> 9 bits

Its domain is 2^8 * 2^8 * 2 = 131072 -- enumerable, so the cell is verified
BIT-EXACT over its whole domain (N/N). The shift and re-assembly between steps
are exact wiring, not neural. Looping the one tied cell 8x yields the full
unsigned 8x8 -> 16 product; signed uses the exact Baugh-Wooley correction.

One verified cell, reused 8 times -- the smallest possible learned core.
"""
from __future__ import annotations

import torch

from .common import bits_of, int_of, pm, mlp, verify, train


class NeuralMACStep:
    """N/N-verified tied cell: acc8 + (a8 if enable else 0) -> 9-bit sum."""

    def __init__(self, h: int = 128, layers: int = 3):
        self.net = mlp(17, 9, h=h, layers=layers)   # acc(8) + a(8) + enable(1) -> 9

    def dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        X, Y = [], []
        for acc in range(256):
            ab = bits_of(acc, 8)
            for a in range(256):
                aa = bits_of(a, 8)
                for en in (0, 1):
                    X.append(pm(torch.cat([ab, aa, torch.tensor([float(en)])])))
                    Y.append(bits_of(acc + (a if en else 0), 9))
        return torch.stack(X), torch.stack(Y)

    def fit(self, steps: int = 5000, lr: float = 2e-3, tag: str = "macstep"):
        X, Y = self.dataset()
        train(self.net, X, Y, steps=steps, lr=lr, tag=tag)
        return self

    def verify(self) -> tuple[int, int]:
        X, Y = self.dataset()
        return verify(self.net, X, Y)

    @torch.no_grad()
    def step(self, acc: int, a: int, enable: int) -> int:
        self.net.eval()
        x = pm(torch.cat([bits_of(acc & 0xFF, 8), bits_of(a & 0xFF, 8),
                          torch.tensor([float(enable)])])).unsqueeze(0)
        return int_of((self.net(x)[0] > 0).float())


class TiedMul8:
    """Signed 8x8 -> 16 multiply from ONE tied MAC-step cell, iterated 8x."""

    def __init__(self, h: int = 128, layers: int = 3):
        self.cell = NeuralMACStep(h=h, layers=layers)

    def fit(self, steps: int = 5000, lr: float = 2e-3, tag: str = "macstep"):
        self.cell.fit(steps=steps, lr=lr, tag=tag)
        return self

    def verify_cell(self) -> tuple[int, int]:
        return self.cell.verify()

    def _umul8(self, a_u: int, b_u: int) -> int:
        """Unsigned product via the tied shift-add loop (cell reused 8x)."""
        combined = b_u & 0xFF                     # low byte holds b, high byte acc
        for _ in range(8):
            enable = combined & 1                 # current LSB of b
            hi = (combined >> 8) & 0xFF
            s = self.cell.step(hi, a_u, enable)   # 9-bit: hi + (a if enable)
            combined = (combined & 0xFF) | (s << 8)
            combined >>= 1                        # shift right one place
        return combined & 0xFFFF

    def mul(self, a: int, b: int) -> int:
        a_u, b_u = a & 0xFF, b & 0xFF
        a7, b7 = (a_u >> 7) & 1, (b_u >> 7) & 1
        prod = self._umul8(a_u, b_u) - (a7 * b_u << 8) - (b7 * a_u << 8) + (a7 * b7 << 16)
        prod &= 0xFFFF
        return prod - 65536 if prod >= 32768 else prod

    @torch.no_grad()
    def verify_unsigned(self) -> tuple[int, int]:
        ok = 0
        for a in range(256):
            for b in range(256):
                if self._umul8(a, b) == a * b:
                    ok += 1
        return ok, 256 * 256

    @torch.no_grad()
    def verify(self) -> tuple[int, int]:
        ok = 0
        for a in range(-128, 128):
            for b in range(-128, 128):
                if self.mul(a, b) == a * b:
                    ok += 1
        return ok, 256 * 256
