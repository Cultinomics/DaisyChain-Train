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

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A).astype(np.int64)
        B = np.asarray(B).astype(np.int64)
        m, k = A.shape
        _, n = B.shape
        # all (A[i,t], B[t,j]) pairs -> one batched neural multiply -> sum over k
        Ai = np.broadcast_to(A[:, None, :], (m, n, k))          # (m,n,k)
        Bj = np.broadcast_to(B.T[None, :, :], (m, n, k))        # (m,n,k)
        prod = self.mul.mul_array(Ai.reshape(-1), Bj.reshape(-1)).reshape(m, n, k)
        return prod.sum(axis=2).astype(np.int64)


def pick_backend(neural=None):
    """NeuralBackend(mul) if a multiplier is given, else the numpy throughput path."""
    if neural is not None:
        return NeuralBackend(neural)
    return NumpyBackend()
