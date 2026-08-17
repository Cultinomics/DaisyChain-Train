"""The GUDA-role path: a fast, vectorized INT8 GEMM on real CPU silicon.

This is where the *throughput* comes from -- real CPU SIMD (via numpy/torch
int32 accumulation), exactly GUDA's job. It is NOT a neural net and NOT
GPU-fast; its ceiling is the CPU. The neural verified op certifies that this
kernel is bit-faithful to the reference (verify_kernel.py) -- turning GUDA-style
"IEEE parity within tolerance" into "provably equal on this finite integer op".
"""
from __future__ import annotations

import numpy as np


#: Largest contraction length an int32 accumulator can take without overflow.
#: |a*b| <= 128*128 = 16384 for int8 operands, so K*16384 must stay inside int32.
#: The bound is real headroom (131071), but layer width is user-selectable from the
#: SpikeWhale panel, so it is checked rather than asserted in a comment -- an
#: overflow here would silently produce wrong numbers, not raise.
MAX_INT32_K = (2 ** 31 - 1) // (128 * 128)


def gemm_int8(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Signed INT8 GEMM with exact int32 accumulation (tensor-core semantics)."""
    k = A.shape[-1]
    if k > MAX_INT32_K:
        raise ValueError(
            "contraction length %d exceeds int32 accumulator headroom %d; "
            "accumulate in int64 for this size" % (k, MAX_INT32_K))
    a = A.astype(np.int32)
    b = B.astype(np.int32)
    return a @ b            # exact integer matmul within the checked bound


def random_int8(shape, rng) -> np.ndarray:
    return rng.integers(-128, 128, size=shape, dtype=np.int16).astype(np.int8)
