"""Native-speed deployment of the verified units, without losing the guarantee.

A verified unit is a finite function. Its neural net is only needed to *prove*
correctness (N/N). For SPEED you materialize the proven function as a lookup
table -- run the net once over its whole (small) domain -- then every later call
is an array index at native memory speed. Because the net is N/N-verified, the
LUT is bit-identical to the net, which is bit-identical to the true op. So:

    neural forward  ==  LUT  ==  native integer op      (all bit-exact)

That's the "freeze the mesh to its matrix" lesson: verify once (slow, offline),
deploy native (fast). The LUTs are tiny: mul 256x256, requant 65536, relu 256.
"""
from __future__ import annotations

import numpy as np


def build_mul8_lut(mul) -> np.ndarray:
    """[256,256] signed-product table, indexed by unsigned bytes. Net runs once."""
    a = np.repeat(np.arange(256), 256)
    b = np.tile(np.arange(256), 256)
    prod = mul.mul_array(a, b)                     # verified neural multiply, ONCE
    return prod.reshape(256, 256).astype(np.int64)


def build_requant16_lut(rq) -> np.ndarray:
    """[65536] int16->int8 table, indexed by acc & 0xFFFF."""
    return rq.requant_array(np.arange(65536)).astype(np.int64)


def build_relu8_lut(relu) -> np.ndarray:
    """[256] int8 ReLU table, indexed by unsigned byte."""
    return relu.relu_array(np.arange(256)).astype(np.int64)


class LUTBackend:
    """GEMM via the materialized (verified) multiply table + integer accumulate."""
    name = "lut"

    def __init__(self, mul):
        self.mul_lut = build_mul8_lut(mul)

    def available(self):
        return True

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        au = (A.astype(np.int64) & 0xFF)
        bu = (B.astype(np.int64) & 0xFF)
        # products via table lookup, then sum over the contraction axis
        prod = self.mul_lut[au[:, None, :], bu.T[None, :, :]]   # (m, n, k)
        from . import instrument
        instrument.bump("VerifiedMul(LUT).gemms", 1)
        instrument.bump("VerifiedMul(LUT).products", prod.size)
        return prod.sum(axis=2)
