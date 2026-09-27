"""Full-domain regression tests for the paths that actually ship.

The units already carry exhaustive verification of their SCALAR entry points
(`NeuralMul8.verify()` walks all 65536 signed pairs against `a*b`). Production
does not call those. It calls the BATCHED paths -- `mul_array`, `relu_array`,
`requant_array` -- and the lookup tables built from them. Those are different
code: nibble concatenation, one batched forward, vectorized sign correction.

A guarantee that covers a path nobody runs is not the guarantee anyone wanted,
so every check here goes against a golden reference (Python integer arithmetic),
over the complete finite domain, on the path that ships.

Run:  python test_verified_units.py
"""
from __future__ import annotations

import sys

import numpy as np

from daisychain.verified.qat import load_units, build_luts
from daisychain.verified import instrument
from daisychain.verified.kernel import gemm_int8, MAX_INT32_K
from daisychain.verified.backends import NeuralBackend
from daisychain.verified.lut import certify_mul8_lut

FAILURES = []


def ck(name, cond, detail=""):
    print("  %-4s %s%s" % ("ok" if cond else "FAIL", name,
                           "" if cond else "  <- " + str(detail)))
    if not cond:
        FAILURES.append(name)


def signed_domain():
    """Every ordered pair of signed bytes, and their true products."""
    a = np.repeat(np.arange(-128, 128), 256)
    b = np.tile(np.arange(-128, 128), 256)
    return a, b, a.astype(np.int64) * b.astype(np.int64)


def main():
    units = load_units()
    mul, requant, relu_unit = units
    luts = build_luts(*units)
    backend = luts["backend"]

    print("multiply -- the SHIPPED batched path, full domain")
    a, b, gold = signed_domain()
    got = mul.mul_array(a.astype(np.int8), b.astype(np.int8))
    ck("mul_array == a*b over all 65536 signed pairs", np.array_equal(got, gold))

    print("multiply -- the materialized table")
    ok, tot = certify_mul8_lut(backend.mul_lut)
    ck("mul LUT == a*b over all 65536 entries", ok == tot, "%d/%d" % (ok, tot))
    ck("LUTBackend self-certified at construction",
       getattr(backend, "certified", None) == (65536, 65536))

    print("requantize -- sat_int8(round_half_up(x / 256)), full int16 domain")
    x = np.arange(65536)
    xs = np.where(x >= 32768, x - 65536, x)
    # Golden is ROUND-HALF-UP, not the arithmetic shift this used to assert.
    # `x >> shift` is a floor, which carries a systematic -0.4981 LSB bias on every
    # requantized activation (measured over this same domain); half-up measures
    # +0.0020. The unit's reference was corrected and requant16.pt retrained
    # against it, so this golden moved with it -- see NeuralRequant16._ref.
    rq_gold = np.clip((xs + (1 << (requant.shift - 1))) >> requant.shift, -128, 127)
    ck("requant_array == sat_int8(round_half_up(x/2**shift)) over all 65536",
       np.array_equal(requant.requant_array(x), rq_gold))
    ck("requant LUT == golden over all 65536",
       np.array_equal(luts["requant"][x & 0xFFFF], rq_gold))

    print("relu -- max(0, x), full int8 domain")
    r = np.arange(256)
    rs = np.where(r >= 128, r - 256, r)
    relu_gold = np.maximum(0, rs)
    ck("relu_array == max(0,x) over all 256",
       np.array_equal(relu_unit.relu_array(rs.astype(np.int8)), relu_gold))
    ck("relu LUT == golden over all 256",
       np.array_equal(luts["relu"][r & 0xFF], relu_gold))

    print("GEMM -- blocked paths must be bit-identical to golden integer matmul")
    rng = np.random.default_rng(11)
    for (m, k, n) in [(1, 1, 1), (3, 5, 7), (64, 64, 64), (96, 128, 96), (129, 257, 65)]:
        A = rng.integers(-128, 128, size=(m, k), dtype=np.int16).astype(np.int8)
        B = rng.integers(-128, 128, size=(k, n), dtype=np.int16).astype(np.int8)
        g = A.astype(np.int64) @ B.astype(np.int64)
        ck("LUT gemm %dx%dx%d == int64 matmul" % (m, k, n),
           np.array_equal(backend.gemm(A, B), g))

    # The neural backend is the slow functional path; check it on a small shape
    # AND across a block boundary, since blocking is what this change introduced.
    # The block count is ASSERTED rather than assumed: mul_array bumps
    # NeuralMul4.forward_calls once per call, i.e. once per block, so the counter
    # is direct evidence the split actually happened. A test that says
    # "forced multi-block" while silently running one block proves nothing.
    nb = NeuralBackend(mul)
    nb.max_products = 16                          # force many blocks on a tiny GEMM
    A = rng.integers(-128, 128, size=(17, 9), dtype=np.int16).astype(np.int8)
    B = rng.integers(-128, 128, size=(9, 11), dtype=np.int16).astype(np.int8)
    instrument.enable()
    instrument.reset()
    got_nb = nb.gemm(A, B)
    blocks = instrument.report().get("NeuralMul4.forward_calls", 0)
    instrument.disable()
    ck("neural gemm actually split into >1 block", blocks > 1, "blocks=%d" % blocks)
    ck("neural gemm == int64 matmul (multi-block)",
       np.array_equal(got_nb, A.astype(np.int64) @ B.astype(np.int64)))

    print("GEMM -- blocking must not change results as the block size varies")
    A = rng.integers(-128, 128, size=(40, 24), dtype=np.int16).astype(np.int8)
    B = rng.integers(-128, 128, size=(24, 32), dtype=np.int16).astype(np.int8)
    ref = backend.gemm(A, B)
    same = True
    for cap in (1 << 10, 1 << 14, 1 << 18, 1 << 26):
        backend.max_block_bytes = cap
        same &= bool(np.array_equal(backend.gemm(A, B), ref))
    backend.max_block_bytes = 64 << 20
    ck("identical across block sizes 1 KB .. 64 MB", same)

    print("int32 accumulator bound is enforced, not assumed")
    ck("MAX_INT32_K == 131071", MAX_INT32_K == 131071, MAX_INT32_K)
    small = np.zeros((1, 4), dtype=np.int8)
    ck("gemm_int8 accepts K within bound",
       gemm_int8(small, np.zeros((4, 1), dtype=np.int8)).shape == (1, 1))
    try:
        gemm_int8(np.zeros((1, MAX_INT32_K + 1), dtype=np.int8),
                  np.zeros((MAX_INT32_K + 1, 1), dtype=np.int8))
        ck("gemm_int8 rejects K past the bound", False, "no error raised")
    except ValueError:
        ck("gemm_int8 rejects K past the bound", True)
    except MemoryError:
        ck("gemm_int8 rejects K past the bound", False,
           "allocated before checking -- bound must be checked first")

    print("instrument -- a zero must not be able to masquerade as evidence")
    instrument.disable()
    instrument.reset()
    ck("enabled() reports the probe state", instrument.enabled() is False)
    try:
        instrument.require(**{"NeuralMul4.forward_calls": 1})
        ck("require() refuses to report while disabled", False, "returned instead")
    except RuntimeError:
        ck("require() refuses to report while disabled", True)

    instrument.enable()
    instrument.reset()
    backend.gemm(np.ones((2, 2), dtype=np.int8), np.ones((2, 2), dtype=np.int8))
    try:
        instrument.require(**{"VerifiedMul(LUT).gemms": 1})
        ck("require() passes when the unit actually ran", True)
    except AssertionError as e:
        ck("require() passes when the unit actually ran", False, e)
    try:
        instrument.require(**{"VerifiedMul(LUT).gemms": 10 ** 9})
        ck("require() fails when a unit is under-invoked", False, "did not raise")
    except AssertionError:
        ck("require() fails when a unit is under-invoked", True)
    instrument.disable()

    print("quantize -- a non-finite input must fail loudly, not become int8 zeros")
    import torch
    from daisychain.verified.qat import VerifiedLinear
    layer = VerifiedLinear(3, 2, mul, requant, relu_unit, fast=True)
    ok_out = layer(torch.tensor([[0.5, -1.0, 3.0]]))
    ck("a finite batch still runs", bool(torch.isfinite(ok_out).all()))
    for bad in (float("nan"), float("inf")):
        try:
            layer(torch.tensor([[0.5, bad, 3.0]]))
            ck("%r in the activations raises" % bad, False,
               "quantized silently (the old path returned all-zero int8)")
        except FloatingPointError:
            ck("%r in the activations raises" % bad, True)

    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("   - %s" % f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
