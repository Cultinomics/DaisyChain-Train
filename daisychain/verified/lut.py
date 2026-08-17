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


def certify_mul8_lut(lut: np.ndarray) -> tuple[int, int]:
    """Check the materialized multiply table against signed integer multiply.

    The table IS the entire finite domain, so certifying it is the exhaustive
    verification -- not a sample of it. Costs ~0.5 ms, which is why there is no
    reason to take the guarantee on trust at runtime.
    """
    a = np.arange(256)
    au, bu = np.repeat(a, 256), np.tile(a, 256)
    sa = np.where(au >= 128, au - 256, au)
    sb = np.where(bu >= 128, bu - 256, bu)
    got = lut[au, bu].ravel()
    return int((got == sa * sb).sum()), got.size


class LUTBackend:
    """GEMM via the materialized (verified) multiply table + integer accumulate."""
    name = "lut"

    #: Cap on the (rows, n, k) product block held at once, in bytes. The block is
    #: an int64 temporary of m*n*k entries, so an unchunked GEMM allocates
    #: m*n*k*8 -- cubic in layer width. At the panel's maximum settings
    #: (hidden 768, seqlen 512) that is 2.4 GB for ONE layer's forward, on
    #: machines this project targets precisely because they are small. Blocking
    #: the m axis bounds it; the contraction axis k is untouched, so the sum and
    #: its order are unchanged and the result is bit-identical.
    max_block_bytes = 64 << 20

    def __init__(self, mul, certify: bool = True):
        self.mul_lut = build_mul8_lut(mul)
        if certify:
            ok, tot = certify_mul8_lut(self.mul_lut)
            if ok != tot:
                raise ValueError(
                    "verified multiply LUT is not bit-exact: %d/%d entries match "
                    "signed integer multiply" % (ok, tot))
            self.certified = (ok, tot)

    def available(self):
        return True

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        au = (A.astype(np.int64) & 0xFF)
        bu = (B.astype(np.int64) & 0xFF)
        m, k = au.shape
        n = bu.shape[1]
        from . import instrument
        instrument.bump("VerifiedMul(LUT).gemms", 1)
        instrument.bump("VerifiedMul(LUT).products", m * n * k)
        # products via table lookup, then sum over the contraction axis
        bt = bu.T[None, :, :]
        rows = max(1, int(self.max_block_bytes // max(1, n * k * 8)))
        if rows >= m:                                   # small layer: one block
            return self.mul_lut[au[:, None, :], bt].sum(axis=2)
        out = np.empty((m, n), dtype=np.int64)
        for i in range(0, m, rows):
            out[i:i + rows] = self.mul_lut[au[i:i + rows, None, :], bt].sum(axis=2)
        return out
