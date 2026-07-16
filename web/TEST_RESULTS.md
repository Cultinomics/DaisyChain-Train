# Test results — updated 2026-07-16

All suites run with `npm test` (chains all ten). Every suite exits 0.
Hardware for GPU numbers: NVIDIA via WebGPU, DP4A int8 dot path, exact-gated
against the verified units at init.

| suite | what it proves | result |
|---|---|---|
| `test_core.js` | float trainer converges, replicas bit-identical | PASS — loss 33.71 → 0.000000, replica diff 0.000e+0 |
| `test_verified.js` | training THROUGH the int8 units converges, replicas bit-identical | PASS — loss 300.6 → 1.41, replica diff 0.000e+0 |
| `test_ieee.js` | the JS epilogue mirror is IEEE-754 spec-correct, not merely agreeable | PASS — 500k+ checks, 0 disagreements; rejects the old round-once mirror on 34% of inputs |
| `test_gates.js` | the exact kernel gate rejects real bugs, accepts the real kernel | PASS — 5/5 injected bugs rejected, clean kernel accepted |
| `test_metamorphic.js` | correctness properties that need no reference implementation | PASS — 6/6 properties hold; catches stride/swap/acc= mutants |
| `test_corpus.js` | mutation-scores the oracles with an external bug taxonomy | PASS — properties 2/4 (2/2 loop, 0/2 math), differential 4/4, control clean |
| `test_optimizer.js` | DaisyAdam beats SGD through the units, deterministic replicas | PASS — 1.59 vs 1.95, replica diff 0.000e+0 |
| `test_transformer.js` | the transformer LM trains through the units end to end | PASS — loss 4.75 → 1.26 (baseline 4.56), replica diff 0.000e+0 |
| `test_unit_backward.js` | int8 STE gradients do not damage convergence | PASS — units/float loss ratio 1.007 |

## The IEEE-754 oracle (`test_ieee.js`)

Built from the binary32 definition in exact BigInt arithmetic — no
`Math.fround` anywhere in the oracle, so neither side was tuned to the other.

```
i32ToF32Spec matches Math.fround on 200000 random int32 (incl. |s| > 2^24)
mulF32Spec matches the correctly-rounded product on 300000 draws (subnormal..overflow)
Verified.epi vs the oracle: 200000 random triples, 0 disagreements
tie-to-even ladder around 2^24: 378 cases, 0 disagreements
bgemmJS outputs rebuilt from the raw int32 accumulator via the oracle: 0 disagreements
the oracle REJECTS the round-once mirror shipped before: 68314/200000 inputs (34.16%)
```

That last line is the teeth: an oracle that never disagrees with anything
proves nothing. This one rejects the exact bug the old `1e-6` tolerance hid.

## Oracle mutation scores (`test_corpus.js`)

Bugs ported from an external taxonomy
([dipankarsarkar/gpuemu-corpus](https://huggingface.co/datasets/dipankarsarkar/gpuemu-corpus))
so the bug list has a different author than the checks.

| bug | lives in | properties | differential |
|---|---|---|---|
| `acc=` instead of `acc+=` | loop | CAUGHT (sensitivity) | CAUGHT |
| missing bounds guard (mult-of-8) | loop | CAUGHT (nonTriviality) | CAUGHT |
| dropped constant factor (2×) | math | missed | CAUGHT |
| wrong leaky-ReLU alpha | math | missed | CAUGHT |

The split is the finding: properties are a **loop-bug detector** (2/2), blind
to structure-preserving value bugs (0/2). Those are owned by the differential
gate against the spec-correct mirror. Each check class has an owner.

## Backward rework: bit-identity + GPU wall clock

The backward was reworked for dispatch efficiency: independent GEMMs
overlapped, the QKV weight-gradient and dln1in trios fused into single
batched (batch=3) GEMMs, and the `g.emb` operand quantized column-wise in one
pass instead of transpose-then-quantize. **None of this may change a bit** —
block scales are per-row/per-column per batch element, so fusion is exact,
and every fused sum keeps the original per-add f32 rounding schedule.

Bit-identity, old backward vs new (CPU mirrors, Node):

```
char-96,        float backward: loss + all  20480 gradient floats bit-identical
char-96,        unit  backward: loss + all  20480 gradient floats bit-identical
Spikewhale 16k, float backward: loss + all 545792 gradient floats bit-identical
Spikewhale 16k, unit  backward: loss + all 545792 gradient floats bit-identical
```

GPU (c=32 t=32 b=8 layers=2 heads=2, 16512-token vocab, 15 measured steps,
gradient FNV hash compared old vs new on-device):

| config | ms/step | grad hash |
|---|---|---|
| old backward, float | 286 | `7ff11308` |
| old backward, units | 466 | `62596547` |
| new backward, float | 286 | `7ff11308` (identical) |
| new backward, units | 346 | `62596547` (identical) |

- float path: unchanged speed, unchanged bits — this is what ships enabled.
- unit-backward path (dormant, `cfg.unitBackward`): cost drops **1.63× → 1.21×**
  vs float. Because the rework is bit-identical, the convergence curves from
  the unit-backward experiment stand unchanged; only the wall-clock exchange
  rate moved. At equal wall clock float still wins (~1.5% at the 200-step
  horizon), so `unitBackward` stays off by default.

## QKV dual-GEMM fusion (CUTLASS ex. 45)

The q/k/v projections share the same left operand (`ln1.y`), so the forward
now quantizes it ONCE and runs all three as one batched (batch=3) dispatch —
2 fewer dispatches and 2 fewer full quantize passes per layer per step.

Bit-identity (old three-GEMM forward vs fused), 5 full training steps with
weight updates in between so a single-ulp divergence anywhere compounds:

```
char-96,        float backward: 5 losses + 5×20480  grads + final weights bit-identical
char-96,        unit  backward: 5 losses + 5×20480  grads + final weights bit-identical
Spikewhale 16k, float backward: 5 losses + 5×545792 grads + final weights bit-identical
Spikewhale 16k, unit  backward: 5 losses + 5×545792 grads + final weights bit-identical
generate() output identical (the fused output's subarray views feed attention)
```

On GPU (DP4A): gradient FNV hashes match the pre-fusion values exactly in
both modes (`7ff11308` float / `62596547` units); ~2% wall-clock gain at
width 32 (the shared operand is only 32 KB there — the saved quantize work
scales quadratically with model width). Two-device live run: both replicas
at step 71/300 with identical loss to the last digit, no sync-guard trips.

## B2B MLP chain (CUTLASS ex. 13 two-GEMM fusion + ex. 23 epilogue reduction)

The MLP's two GEMMs now run back-to-back on the GPU: gemm1 (ReLU fused) and a
per-row |max| reduction share one command encoder, ~1 KB of absmax comes back
to JS (scale derivation needs division, which WGSL only guarantees to 2.5 ULP
— JS f64 division is exactly rounded and device-identical), then h1 is
quantized ON-DEVICE and fed straight to gemm2. h1 returns to JS only because
the STE backward needs it; it never goes up again.

This required respeccing the intermediate quantize from `round(x / scale)` to
`floor(f32(x * invScale) + 0.5)` — WGSL multiply/add are correctly rounded and
floor/clamp exact, so the GPU kernel and the fround-stepped JS mirror agree
bit-for-bit, and CPU-fallback devices run the mirror so mixed fleets stay
bit-identical. The respec is a real (bounded) math change: old and new builds
cannot co-train, and the per-step divergence guard stops such mixed groups.

`test_b2b.js` (Node):

```
scales from the fused absmax bit-identical to quantizeRows (3958 rows incl. zero rows)
respec moves an int8 by at most 1 step (1/112363 = 0.001% of values moved)
chain gemm1 (hence h1 and the ReLU mask) byte-identical to the un-chained GEMM
chain output equals the manual composition of its stages; deterministic
convergence unchanged: old 2.0500 vs new 2.0512 final loss (0.1% apart, 40 steps)
```

On GPU: both chain variants (LUT shader and DP4A) pass an exact `!==` init
gate against the mirror chain over ragged shapes including pack-tail padding.
Discriminating proof that the kernel implements the RESPEC and not the old
spec: a searched-for boundary input (int8 68→69 under the respec) run through
the GPU chain matches the new-spec mirror exactly and differs from the
old-spec composition. Two-device live run: step 141/300 with identical loss
(6.23958) on both replicas, per-step weight-hash divergence checks silent.

One bug was caught during the rework, by the bit-identity check itself: the
fused q+k+v sum initially ran in f64 and rounded once, where the old code
rounded to f32 after each add — a last-ulp fork that would have split
replicas. Fixed by matching the rounding schedule (`Math.fround` per add).
Same lesson as the epilogue mirror: match the rounding schedule, not just
the values.
