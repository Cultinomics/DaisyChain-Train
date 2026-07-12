"""The GUDA-role path: a fast, vectorized INT8 GEMM on real CPU silicon.

This is where the *throughput* comes from -- real CPU SIMD (via numpy/torch
int32 accumulation), exactly GUDA's job. It is NOT a neural net and NOT
GPU-fast; its ceiling is the CPU. The neural verified op certifies that this
kernel is bit-faithful to the reference (verify_kernel.py) -- turning GUDA-style
"IEEE parity within tolerance" into "provably equal on this finite integer op".
"""
from __future__ import annotations

import numpy as np


def gemm_int8(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Signed INT8 GEMM with exact int32 accumulation (tensor-core semantics)."""
    a = A.astype(np.int32)
    b = B.astype(np.int32)
    return a @ b            # exact integer matmul, no overflow for our sizes


def random_int8(shape, rng) -> np.ndarray:
    return rng.integers(-128, 128, size=shape, dtype=np.int16).astype(np.int8)
