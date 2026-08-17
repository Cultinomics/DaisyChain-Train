"""Compute backends for the bridge -- pure Python, no external toolchain.

  NumpyBackend  -- int32 GEMM on CPU SIMD (numpy). Real throughput, CPU-capped.
  NeuralBackend -- GEMM where every multiply is the N/N-verified neural unit and
                   accumulation is exact integer sum. The compute path is itself
                   a net: bit-exact, but functional (slow), not a speed path.

Whatever backend runs, the bridge's self-certify gate proves its output bit-exact
to the verified op before any result is trusted.

(An optional Go/GUDA backend lives in backends_go.py; it is not imported here and
not required -- this package stands alone without Go.)
"""
from __future__ import annotations

import numpy as np

from .kernel import gemm_int8


class NumpyBackend:
    name = "numpy-int8"

    def available(self) -> bool:
        return True

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return gemm_int8(A, B)


class NeuralBackend:
    """GEMM computed entirely by the verified neural multiply (+ exact sum).

    Pass a trained NeuralMul8 (or anything with `mul_array(a, b)`), e.g. loaded
    from mul8.pt. Products come from the net; accumulation is exact int64.
    """
    name = "neural-mul"

    def __init__(self, mul):
        self.mul = mul

    def available(self) -> bool:
        return hasattr(self.mul, "mul_array")

    #: Cap on PRODUCTS (m*n*k entries) handed to one batched multiply.
    #:
    #: Counted in products, not bytes, because bytes-per-product here is not the
    #: 8 of an int64 table lookup: `reshape(-1)` materializes the broadcast pair
    #: arrays, `mul_array` then splits each product into FOUR nibble pairs, and
    #: each pair goes through a 128-wide net whose activations dominate everything
    #: else. Measured on the numpy side alone: ~880 bytes per product (589,824
    #: products -> 519 MB peak), and torch's own allocations sit on top of that.
    #:
    #: Applying LUTBackend's m*n*k*8 accounting here -- which the first version of
    #: this patch did -- computes a block 110x too large and never splits at all.
    #: The contraction axis k is untouched either way, so blocking cannot change
    #: the sum or its order: the result stays bit-identical.
    max_products = 1 << 16

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A).astype(np.int64)
        B = np.asarray(B).astype(np.int64)
        m, k = A.shape
        _, n = B.shape
        Bt = B.T[None, :, :]
        out = np.empty((m, n), dtype=np.int64)
        rows = max(1, int(self.max_products // max(1, n * k)))
        for i in range(0, m, rows):
            r = A[i:i + rows]
            rn = r.shape[0]
            # all (A[i,t], B[t,j]) pairs in this block -> one batched neural multiply
            Ai = np.broadcast_to(r[:, None, :], (rn, n, k))
            Bj = np.broadcast_to(Bt, (rn, n, k))
            prod = self.mul.mul_array(Ai.reshape(-1), Bj.reshape(-1)).reshape(rn, n, k)
            out[i:i + rows] = prod.sum(axis=2)
        return out.astype(np.int64)


def pick_backend(neural=None):
    """NeuralBackend(mul) if a multiplier is given, else the numpy throughput path."""
    if neural is not None:
        return NeuralBackend(neural)
    return NumpyBackend()
