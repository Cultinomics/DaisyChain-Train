"""Verified neural INT8 multiplier -- the atom of a GPU tensor core / VNNI lane.

A monolithic MLP cannot learn a bit-exact 8x8 multiply (the high-order product
bits are too nonlinear). So -- exactly like the byte-slice / ripple trick the
other projects use for hard functions -- we shrink the *verified atom* to a
4-bit unsigned multiply and compose everything exactly:

  atom:   NeuralMul4  -- unsigned 4x4 -> 8, domain 16*16 = 256, verified N/N.
  8x8:    a*b = ah*bh<<8 + (ah*bl + al*bh)<<4 + al*bl   (unsigned), exact.
  signed: a_s = a_u - 256*a7 ;  Baugh-Wooley correction, exact integer glue.

Only the 4x4 multiply is neural (and N/N-proven); the shifts, adds and sign
correction are exact composition. The parallel *throughput* of thousands of such
lanes is NOT here -- that needs real silicon (kernel.py, the GUDA-role path).
"""
from __future__ import annotations

import numpy as np
import torch

from .common import bits_of, int_of, pm, mlp, verify, train


class NeuralMul4:
    """N/N-verified unsigned 4x4 -> 8 multiplier (the finite atom)."""

    def __init__(self, h: int = 128, layers: int = 3):
        self.net = mlp(8, 8, h=h, layers=layers)

    def dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        X, Y = [], []
        for a in range(16):
            ab = bits_of(a, 4)
            for b in range(16):
                X.append(pm(torch.cat([ab, bits_of(b, 4)])))
                Y.append(bits_of(a * b, 8))
        return torch.stack(X), torch.stack(Y)

    def fit(self, steps: int = 4000, lr: float = 2e-3, tag: str = "mul4"):
        X, Y = self.dataset()
        train(self.net, X, Y, steps=steps, lr=lr, tag=tag)
        return self

    def verify(self) -> tuple[int, int]:
        X, Y = self.dataset()
        return verify(self.net, X, Y)

    @torch.no_grad()
    def mul(self, a: int, b: int) -> int:
        self.net.eval()
        x = pm(torch.cat([bits_of(a & 0xF, 4), bits_of(b & 0xF, 4)])).unsqueeze(0)
        return int_of((self.net(x)[0] > 0).float())

    @torch.no_grad()
    def mul_array(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Batched unsigned 4x4 -> 8 over arrays (one neural forward for all)."""
        self.net.eval()
        a = (np.asarray(a).astype(np.int64) & 0xF)
        b = (np.asarray(b).astype(np.int64) & 0xF)
        idx = np.arange(4)
        from . import instrument
        from .ops import _blocked

        def _run(sl):
            ca, cb = a[sl], b[sl]
            bits_a = (ca[:, None] >> idx) & 1
            bits_b = (cb[:, None] >> idx) & 1
            x = np.concatenate([bits_a, bits_b], axis=1).astype(np.float32) * 2.0 - 1.0
            out = (self.net(torch.from_numpy(x)) > 0).to(torch.int64).numpy()
            instrument.bump("NeuralMul4.forward_calls", 1)
            return (out * (1 << np.arange(8))).sum(axis=1)

        # BOUNDED. `NeuralBackend.gemm` blocks before it reaches here, but the LUT build does not:
        # `build_mul8_lut` hands NeuralMul8.mul_array all 65,536 pairs, which become 262,144 rows
        # through this net. Measured 277 MB RSS at import time, on every node, on hardware this
        # project targets for being small. Blocking is exact -- this is an elementwise map.
        return _blocked(_run, np.arange(a.shape[0]))


class NeuralMul8:
    """Signed 8x8 -> 16 multiply, composed exactly from the verified 4x4 atom."""

    def __init__(self, h: int = 128, layers: int = 3):
        self.atom = NeuralMul4(h=h, layers=layers)

    def fit(self, steps: int = 4000, lr: float = 2e-3, tag: str = "mul4"):
        self.atom.fit(steps=steps, lr=lr, tag=tag)
        return self

    def verify_atom(self) -> tuple[int, int]:
        return self.atom.verify()

    def _umul8(self, a_u: int, b_u: int) -> int:
        """Unsigned 8x8 -> 16 via four verified 4x4 sub-products (exact glue)."""
        al, ah = a_u & 0xF, (a_u >> 4) & 0xF
        bl, bh = b_u & 0xF, (b_u >> 4) & 0xF
        ll = self.atom.mul(al, bl)
        lh = self.atom.mul(al, bh)
        hl = self.atom.mul(ah, bl)
        hh = self.atom.mul(ah, bh)
        return ll + ((lh + hl) << 4) + (hh << 8)

    def mul(self, a: int, b: int) -> int:
        """Signed product via the verified atom + exact Baugh-Wooley correction."""
        a_u, b_u = a & 0xFF, b & 0xFF
        a7, b7 = (a_u >> 7) & 1, (b_u >> 7) & 1
        prod = self._umul8(a_u, b_u) - (a7 * b_u << 8) - (b7 * a_u << 8) + (a7 * b7 << 16)
        prod &= 0xFFFF                      # 16-bit two's complement
        return prod - 65536 if prod >= 32768 else prod

    @torch.no_grad()
    def mul_array(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Batched signed 8x8 -> 16 over arrays, via the verified 4x4 atom.

        Four nibble sub-products are batched into ONE neural forward, then the
        exact shift/add glue and Baugh-Wooley sign correction are applied in
        numpy. Result is bit-exact (the atom is N/N-verified)."""
        a = np.asarray(a).astype(np.int64).ravel()
        b = np.asarray(b).astype(np.int64).ravel()
        au, bu = a & 0xFF, b & 0xFF
        al, ah = au & 0xF, (au >> 4) & 0xF
        bl, bh = bu & 0xF, (bu >> 4) & 0xF
        n = au.shape[0]
        pa = np.concatenate([al, al, ah, ah])
        pb = np.concatenate([bl, bh, bl, bh])
        prod = self.atom.mul_array(pa, pb)
        ll, lh, hl, hh = prod[:n], prod[n:2*n], prod[2*n:3*n], prod[3*n:4*n]
        u = ll + ((lh + hl) << 4) + (hh << 8)
        a7, b7 = (au >> 7) & 1, (bu >> 7) & 1
        p = (u - (a7 * bu << 8) - (b7 * au << 8) + (a7 * b7 << 16)) & 0xFFFF
        return np.where(p >= 32768, p - 65536, p)

    @torch.no_grad()
    def verify(self, full: bool = True) -> tuple[int, int]:
        """Exhaustively check the composed signed multiply over all 65536 inputs."""
        ok = 0
        for a in range(-128, 128):
            for b in range(-128, 128):
                if self.mul(a, b) == a * b:
                    ok += 1
        return ok, 256 * 256
