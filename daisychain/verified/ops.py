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

from .common import bits_of, bits_matrix, int_of, pm, mlp, verify, train
from . import instrument


#: Cap on ELEMENTS handed to one batched neural forward, for the two elementwise
#: units below.
#:
#: `NeuralBackend.gemm` and `LUTBackend.gemm` both bound their temporaries; these
#: two did not, and they sit directly downstream of the GEMM in the proof path
#: (`_VerifiedQGEMM.forward` -> `requant_array(acc16)` -> `relu_array(yq)`), where
#: they receive the WHOLE activation matrix at once. `requant_array` widens each
#: element to 16 float32 inputs and pushes them through a 3-layer, 256-wide MLP,
#: so the hidden activations alone are N*256*4 bytes per layer.
#:
#: Measured (RSS delta, requant16): ~1.5-2 KB per element -- 65,536 elements cost
#: 144 MB and 262,144 cost 407 MB. A 512x768 activation is 393,216 elements. That
#: is the same failure the GEMM caps already guard against, on hardware this
#: project targets precisely because it is small.
#:
#: 32,768 elements ~= 64 MB, comparable to NeuralBackend's own product budget.
#:
#: Blocking is UNCONDITIONALLY exact here in a way it is not for a GEMM: these are
#: elementwise maps, so no accumulation order exists to perturb. Every output byte
#: is identical to the unblocked path.
MAX_UNIT_ELEMENTS = 1 << 15


def _blocked(fn, a: np.ndarray, cap: int = MAX_UNIT_ELEMENTS) -> np.ndarray:
    """Apply an elementwise unit to a flat array in bounded chunks."""
    n = a.shape[0]
    if n <= cap:
        return fn(a)
    out = np.empty(n, dtype=np.int64)
    for i in range(0, n, cap):
        out[i:i + cap] = fn(a[i:i + cap])
    return out


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
        b = np.arange(256)
        sgn = np.where(b >= 128, b - 256, b)
        return pm(bits_matrix(b, 8)), bits_matrix(np.maximum(0, sgn) & 0xFF, 8)

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

        def _run(chunk):
            bits = ((chunk[:, None] >> np.arange(8)) & 1).astype(np.float32) * 2.0 - 1.0
            out = (self.net(torch.from_numpy(bits)) > 0).to(torch.int64).numpy()
            instrument.bump("NeuralReLU8.forward_calls", 1)
            return (out * (1 << np.arange(8))).sum(axis=1)

        raw = _blocked(_run, a)                       # bounded; see MAX_UNIT_ELEMENTS
        instrument.bump("NeuralReLU8.elements", a.shape[0])
        return np.where(raw >= 128, raw - 256, raw).reshape(np.asarray(arr).shape)


class NeuralRequant16:
    """N/N-verified requantize: int16 -> int8, y = sat_int8(x >> shift)."""

    def __init__(self, shift: int = 8, h: int = 256, layers: int = 3):
        self.shift = int(shift)
        self.net = mlp(16, 8, h=h, layers=layers)

    def _ref(self, x_signed: int) -> int:
        """sat_int8(round_half_up(x / 2**shift)) -- rounding, not truncation.

        This was `x >> shift`, and Python's `>>` on negatives is an arithmetic
        FLOOR. Measured over the full int16 domain (unsaturated band): floor has a
        mean error of **-0.4981 LSB** -- a systematic negative bias on every
        requantized activation in the network -- against +0.002 for round-half-up.
        The two disagree on 32,640 of 65,536 inputs.

        Half-up via `(x + 2**(shift-1)) >> shift` is exact integer arithmetic and
        is the canonical hardware requant, which matters because this unit exists
        to model hardware. It removes the floor bias the Byrne/morpho work avoids with
        `sat_int8(round(acc / 128))` -- but note the TIE rule: Python/numpy/torch
        `round` is half-to-EVEN, this is half-UP. Measured over the int16 domain they
        disagree on exactly the 128 ties (x mod 256 == 128 with an even quotient),
        0.2% of inputs; both are unbiased to within 0.002 LSB. So "agree" holds for
        the bias, not bit-for-bit unless morpho also rounds half-up. (Same tie
        question: `qat.py` quantizes with np.round, half-even; the browser build uses
        floor(x + 0.5), half-up. The two fleets never co-train, so they cannot fork.)

        Changing this invalidates any previously trained requant16.pt: the unit is
        verified N/N against THIS reference, so the weights must be retrained and
        re-verified whenever it moves.
        """
        return sat_int8((x_signed + (1 << (self.shift - 1))) >> self.shift)

    def dataset(self):
        b = np.arange(65536)
        sgn = np.where(b >= 32768, b - 65536, b)
        y = np.clip((sgn + (1 << (self.shift - 1))) >> self.shift, -128, 127)
        return pm(bits_matrix(b, 16)), bits_matrix(y & 0xFF, 8)

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

        def _run(chunk):
            bits = ((chunk[:, None] >> np.arange(16)) & 1).astype(np.float32) * 2.0 - 1.0
            out = (self.net(torch.from_numpy(bits)) > 0).to(torch.int64).numpy()
            instrument.bump("NeuralRequant16.forward_calls", 1)
            return (out * (1 << np.arange(8))).sum(axis=1)

        raw = _blocked(_run, a)                       # bounded; see MAX_UNIT_ELEMENTS
        instrument.bump("NeuralRequant16.elements", a.shape[0])
        return np.where(raw >= 128, raw - 256, raw).reshape(np.asarray(arr).shape)
