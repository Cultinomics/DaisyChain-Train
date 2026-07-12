"""Verified INT8 activation / requantize units -- the rest of a quantized layer.

A quantized linear layer is:  int8 x int8 -> int32 accumulate -> requantize to
int8 -> activation. The multiply/accumulate is covered by mul8 + exact int32
sum. These two units cover the tail, each over a FINITE, enumerable domain so
they are verified BIT-EXACT (N/N):

  NeuralReLU8      int8 -> int8,  y = max(0, x)                 domain 256
  NeuralRequant16  int16 -> int8, y = sat_int8(x >> shift)      domain 65536

The int32 -> int16 narrowing that precedes requantize is an EXACT saturating
clamp (integer wiring, not neural). So a full qlinear is exact/verified through
every stage.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import bits_of, int_of, pm, mlp, verify, train
from . import instrument


def _s(v: int, bits: int) -> int:
    """two's-complement raw -> signed."""
    m = 1 << (bits - 1)
    return v - (1 << bits) if v >= m else v


def sat_int8(x: int) -> int:
    return -128 if x < -128 else (127 if x > 127 else x)


class NeuralReLU8:
    """N/N-verified INT8 ReLU: y = max(0, x)."""

    def __init__(self, h: int = 64, layers: int = 2):
        self.net = mlp(8, 8, h=h, layers=layers)

    def dataset(self):
        X, Y = [], []
        for b in range(256):
            X.append(pm(bits_of(b, 8)))
            Y.append(bits_of(max(0, _s(b, 8)) & 0xFF, 8))
        return torch.stack(X), torch.stack(Y)

    def fit(self, steps: int = 3000, lr: float = 2e-3, tag: str = "relu8"):
        X, Y = self.dataset(); train(self.net, X, Y, steps=steps, lr=lr, tag=tag); return self

    def verify(self):
        X, Y = self.dataset(); return verify(self.net, X, Y)

    @torch.no_grad()
    def relu(self, x: int) -> int:
        self.net.eval()
        raw = int_of((self.net(pm(bits_of(x & 0xFF, 8)).unsqueeze(0))[0] > 0).float())
        return _s(raw, 8)

    @torch.no_grad()
    def relu_array(self, arr: np.ndarray) -> np.ndarray:
        """Batched int8 ReLU over an array (one neural forward)."""
        self.net.eval()
        a = np.asarray(arr).astype(np.int64).ravel() & 0xFF
        bits = ((a[:, None] >> np.arange(8)) & 1).astype(np.float32) * 2.0 - 1.0
        out = (self.net(torch.from_numpy(bits)) > 0).to(torch.int64).numpy()
        raw = (out * (1 << np.arange(8))).sum(axis=1)
        instrument.bump("NeuralReLU8.forward_calls", 1)
        instrument.bump("NeuralReLU8.elements", a.shape[0])
        return np.where(raw >= 128, raw - 256, raw).reshape(np.asarray(arr).shape)


class NeuralRequant16:
    """N/N-verified requantize: int16 -> int8, y = sat_int8(x >> shift)."""

    def __init__(self, shift: int = 8, h: int = 256, layers: int = 3):
        self.shift = int(shift)
        self.net = mlp(16, 8, h=h, layers=layers)

    def _ref(self, x_signed: int) -> int:
        return sat_int8(x_signed >> self.shift)      # arithmetic shift, saturate

    def dataset(self):
        X, Y = [], []
        for b in range(65536):
            X.append(pm(bits_of(b, 16)))
            Y.append(bits_of(self._ref(_s(b, 16)) & 0xFF, 8))
        return torch.stack(X), torch.stack(Y)

    def fit(self, steps: int = 6000, lr: float = 2e-3, tag: str = "requant16"):
        X, Y = self.dataset(); train(self.net, X, Y, steps=steps, lr=lr, tag=tag); return self

    def verify(self):
        X, Y = self.dataset(); return verify(self.net, X, Y)

    @torch.no_grad()
    def requant(self, x: int) -> int:
        self.net.eval()
        raw = int_of((self.net(pm(bits_of(x & 0xFFFF, 16)).unsqueeze(0))[0] > 0).float())
        return _s(raw, 8)

    @torch.no_grad()
    def requant_array(self, arr: np.ndarray) -> np.ndarray:
        """Batched int16->int8 requantize over an array (one neural forward)."""
        self.net.eval()
        a = np.asarray(arr).astype(np.int64).ravel() & 0xFFFF
        bits = ((a[:, None] >> np.arange(16)) & 1).astype(np.float32) * 2.0 - 1.0
        out = (self.net(torch.from_numpy(bits)) > 0).to(torch.int64).numpy()
        raw = (out * (1 << np.arange(8))).sum(axis=1)
        instrument.bump("NeuralRequant16.forward_calls", 1)
        instrument.bump("NeuralRequant16.elements", a.shape[0])
        return np.where(raw >= 128, raw - 256, raw).reshape(np.asarray(arr).shape)
