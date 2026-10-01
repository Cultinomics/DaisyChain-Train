# Test results — updated 2026-07-17

All suites run with `npm test` (chains all thirteen). Every suite exits 0.
Hardware for GPU numbers: NVIDIA via WebGPU, DP4A int8 dot path, exact-gated
against the verified units at init.

| suite | what it proves | result |
|---|---|---|
| `test_core.js` | float trainer converges, replicas bit-identical | PASS — loss 33.71 → 0.000000, replica diff 0.000e+0 |
| `test_verified.js` | training THROUGH the int8 units converges, replicas bit-identical | PASS — loss 300.6 → 1.41, replica diff 0.000e+0 |
| `test_ieee.js` | the JS epilogue mirror is IEEE-754 spec-correct, not merely agreeable | PASS — 1.4M+ checks (mul, add, fma), 0 disagreements; rejects the old round-once mirror on 34% of inputs; agrees bit-for-bit with an independent big-int fma golden on 8000 vectors |
| `test_gates.js` | the exact kernel gate rejects real bugs, accepts the real kernel | PASS — 5/5 injected bugs rejected, clean kernel accepted, audit sees the sign of zero |
| `test_metamorphic.js` | correctness properties that need no reference implementation | PASS — 6/6 properties hold; catches stride/swap/acc= mutants |
| `test_corpus.js` | mutation-scores the oracles with an external bug taxonomy | PASS — properties 4/4 (2/2 loop, 2/2 math), differential 4/4, control clean |
| `test_selfcorpus.js` | scores every instrument against MY OWN four bugs from July 2026 | PASS — properties 0/2, differential 2/2, metaTest 1/1, liveness 1/1 |
| `test_optimizer.js` | DaisyAdam beats SGD through the units, deterministic replicas | PASS — 1.59 vs 1.95, replica diff 0.000e+0 |
| `test_transformer.js` | the transformer LM trains through the units end to end | PASS — loss 4.75 → 1.26 (baseline 4.56), replica diff 0.000e+0 |
| `test_unit_backward.js` | int8 STE gradients do not damage convergence | PASS — units/float loss ratio 1.007 |
| `test_b2b.js` | the chained GPU MLP's CPU mirror: gemm1 byte-identical to the un-chained GEMM, deterministic, +0.5 quantize FMA-contraction-immune | PASS — chained transformer converges 4.7384 → 2.0512 |
| `test_nonfinite.js` | every quantizer refuses NaN/Inf instead of zeroing the layer | PASS — 13/13; the old quantizers fail 12 (added 2026-09-27) |
| `test_reductions.js` | the GPU row-max guard is complete, `averageGrads` refuses wrong-length / non-finite gradients, int32 accumulators refuse K > 131,071 | PASS — 19/19; the old code fails 13 (added 2026-09-30) |

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

The oracle now also covers **addition** and **fused multiply-add**:

- `addF32Spec` (exact BigInt sum, one rounding) certifies that `fround(a+b)`
  is the correctly-rounded f32 sum (Figueroa: 53 ≥ 2·24+2 makes the double
  rounding innocuous for add) — the fact every per-add mirror schedule in the
  codebase stands on. 300k draws including extreme exponent gaps, 0
  disagreements.
- `fmaF32Spec` (exact product, never rounded, plus addend, ONE rounding) —
  ported from the neural-rdna2 project's from-the-definition fma golden,
  which correctly rejects the float64 shortcut (it double-rounds on rare
  ties). Cross-checked **bit-for-bit against that independent Python
  implementation on 8000 generated vectors** (quantize-domain, catastrophic
  cancellation, raw finite bit patterns): two oracles, two codebases, two
  implementations of the same paragraph of the standard, zero disagreements.
  Fused really is different: it differs from the round-twice composition on
  100% of cancellation cases.
- **Mul and add carry the same cross-check** (`test_vectors_fp32.json`):
  6004 vectors computed by neural-rdna2's **LUT-backed verified fp32 core** —
  the RDNA2 wave interpreter's own arithmetic, every mantissa product through
  the exported mul4 atom — over epilogue-shaped ranges, subnormal/overflow
  regimes, raw finite bits, and signed zeros. `mulF32Spec` and `addF32Spec`
  agree bit-for-bit on all of them. Three independent implementations now
  concur on the epilogue's arithmetic: this BigInt oracle, V8's fround, and a
  verified-unit RDNA2 core.
- The fma-contraction immunity claim now stands on checked facts instead of
  arguments: on the quantize domain the f64 emulation used by `test_b2b.js`
  equals the true fma (300k draws), and the floor-invisibility result holds
  against the true fma at the binade edges (66k last-ulp diffs, 0
  floor-visible).

## Oracle mutation scores (`test_corpus.js`)

Bugs ported from an external taxonomy
([dipankarsarkar/gpuemu-corpus](https://huggingface.co/datasets/dipankarsarkar/gpuemu-corpus))
so the bug list has a different author than the checks.

| bug | lives in | properties | differential |
|---|---|---|---|
| `acc=` instead of `acc+=` | loop | CAUGHT (sensitivity) | CAUGHT |
| missing bounds guard (mult-of-8) | loop | CAUGHT (nonTriviality) | CAUGHT |
| dropped constant factor (2×) | math | CAUGHT (unitScaleAnchor) | CAUGHT |
| wrong leaky-ReLU alpha | math | CAUGHT (reluRange) | CAUGHT |

**4/4 both oracles** — but the road there is the finding. The first score was
0/4; adding non-triviality and sensitivity got the loop bugs (2/4). The two
math bugs are provably invisible to any RELATION — if `out` satisfies every
relation, so does `c·out` — so no cleverer relation exists. Closing them took
a different species of check: **definitional absolutes**. `reluRange` (ReLU
output cannot be negative — a range constraint from the definition) catches
the leaky alpha; `unitScaleAnchor` (at unit scales dequant is the identity, so
the output must equal the exact integer dot product, computed with plain
integer arithmetic — no LUT, no mirror) pins absolute scale and catches the
uniform 2×. Still no reference implementation anywhere in the property suite;
the suite is now relations for the loop plus spec-pinned absolutes for the
values, and the differential gate remains an independent second opinion.

## Scoring my own bugs (`test_selfcorpus.js`)

The external corpus measures the oracles against kernel bugs someone else
wrote down. But this month's four REAL bugs (each with a name and a fix) are
a different population:

| bug | lives in | properties | differential | metaTest | liveness |
|---|---|---|---|---|---|
| stripped binding (scales ignored) | data/scale | MISSED | CAUGHT | — | — |
| round-once sum (wrong rounding schedule) | data/rounding | MISSED | CAUGHT | — | — |
| dead gate (vacuous pass) | the checker | — | — | CAUGHT | — |
| roster-gradient stall | the protocol | — | — | — | CAUGHT |

Properties score **0/2** on the data-plane pair — the c·out theorem again:
one bug is a per-column scalar, the other a last-ulp rounding change, and the
unit-scale anchor sits exactly where both are invisible. The differential
gate catches both. But half the bugs did not live in the kernels at all: the
dead gate is a bug in a CHECKER (only mutation-testing the gate sees it), and
the stall is a bug in the PROTOCOL (every computed value on every peer was
correct, so no data oracle can fire — a liveness simulation with an
asymmetric gradient drop catches it, and verifies the repair protocol
finishes with identical weights). The instruments that caught this month's
bugs are not better oracles; they are different instruments. The population
defines the instrument, not the other way round.

## Shared-operand embedding GEMMs (profile-driven, zero math change)

Profiling a step (wrapping the engine methods, shipped code untouched) put the
two f32 backward GEMMs at **55% of the entire step** — 204.8 ms across 2 calls:

```
fgemm (f32 backward)   2 calls   204.8 ms   55.5% of step   <- the pair
bgemm                  5 calls    99.2 ms   26.9%
mlp (B2B chain)        2 calls    17.7 ms    4.8%
att.scores/ctx         4 calls    18.0 ms    4.9%
JS remainder                      29.2 ms    7.9%
```

Both GEMMs consume the SAME operand — `dlogits`, which at the 16512-token
vocab is 256x16512 f32 = **~17 MB** — so the old code uploaded 17 MB twice per
step and paid two submits and two map round trips. `gpuFgemm2` uploads it
once, runs both GEMM+reduce chains on one command encoder, submits once, and
maps both readbacks concurrently. The shader already reads A as either
`A[row*k+p]` or `A[p*m+row]` (transA), so one flat buffer serves both index
patterns and no arithmetic changes.

Controlled A/B (one engine init so the same backend serves both arms, same
session, same seed, arms interleaved, best-of-2):

```
old (2 separate fgemm calls) : 363.9 ms/step   grad 89b0f76a  loss b3afdb60
new (fgemm2 shared operand)  : 319.8 ms/step   grad 89b0f76a  loss b3afdb60
12.1% faster; gradient AND loss hashes bit-identical
```

The pair itself went 204.8 -> 90.4 ms (2.3x). Init gates the fusion the same
way everything else is gated: both legs must equal the two calls they replace
**bit-for-bit** (not a tolerance), exercising transA on one leg and the plain
path on the other; on any mismatch `fgemm2` is dropped and the old two-call
path runs. A note on method: the init backend race makes step totals vary
run-to-run (LUT vs DP4A can win), so uncontrolled before/after numbers are
not comparable — hence the same-session A/B.

## Two gaps found by auditing the verification itself

Reviewing the instruments (rather than the kernels) turned up two places where
the standard applied everywhere else had not been applied.

**1. The attention kernels had no live audit.** Every other GPU kernel is
recomputed against the units at *live* shapes; `att.scores` and `att.ctx` had
exact init gates and nothing after — the precise gap the audit exists to close,
left open on the kernels with the trickiest indexing (head-strided gather,
scatter write-back). `auditAttScores` / `auditAttCtx` now recompute individual
cells, stratified like the GEMM audit (last head, last token pair, last
channel first). Both reject a corrupted last-head/last-channel cell and accept
the clean kernel; a live 300-step run audits every attention dispatch with no
false positives.

**2. The f32 backward GEMM was gated by `allclose`** (`1e-3`) — the last
tolerance in a shipped path, and this suite's own mutation test shows allclose
waving through a real rounding bug. The reason was legitimate: split-K
accumulates in a different order than a naive reference, so bit-equality
against *that* reference is impossible. So the reference was rewritten instead:
`fgemmMirror` reproduces split-K's partition and accumulation order exactly,
and the gate now compares with `!==`.

That raised a question the code could not answer by reasoning: WGSL permits a
compiler to contract `s + a*b` into a fused multiply-add (one rounding instead
of two), which would change the result. Rather than assume, the gate tries both
schedules and reports which the device implements — and refuses the kernel if
**neither** matches, since a backward GEMM that is not bit-reproducible sets
weights the whole fleet is hashed against. Measured here:

```
split-K f32 GEMM is bit-exact (two-rounding multiply-add schedule)
```

So this driver does not contract, and the f32 backward is now exactly gated.
The two schedules are not a distinction without a difference — they disagree
on 22/24 cells at the gate shape, so the gate is choosing between real
alternatives. (The same measurement shows split-K disagreeing with a naive
matmul on 22/24 cells, which is exactly why the tolerance was there and why
the mirror had to be written.)

## Fixing the audit's coverage (the weak spot, named and closed)

The live audit's two constants — `cells: 6`, `due: 2% of GEMMs` — were chosen
to bound overhead and never justified against a bug class. That sampled ~1.2
cells per step, ~360 over a 300-step run: a coverage number sitting in exactly
the seat `allclose` used to occupy.

**The overhead premise was false.** An audit costs `k` multiply-adds per cell
in JS. Measured at the live logits shape:

```
 6 cells:  2.1 us/audit  ->  0.021 ms/step if EVERY GEMM is audited
16 cells:  2.8 us/audit  ->  0.028 ms/step
64 cells: 11.6 us/audit  ->  0.116 ms/step
```

Against a ~320 ms step that is under 0.01%. The 2% rate was throttling
coverage to save nothing, so the audit now runs on **every** GEMM.

**But the rate was the smaller half.** The bugs that actually occur here —
a bounds-guard off-by-one, a pack-tail padding bug — corrupt the **last
row/column only**. Uniform random sampling finds that with probability ~1/n
per cell, and at the 16512-wide unembed that is never. So sampling is now
**stratified**: the first 6 cells are the structural danger points (last
row, last column, all four corners, last batch) chosen deterministically, the
rest random interior cells for diffuse bugs.

Measured against that named bug class (last column zeroed, n=512, 300 audits):

| strategy | caught |
|---|---|
| uniform, 12 cells | **5 / 300** (predicted ~2.3%) |
| stratified, 12 cells | **300 / 300** |
| stratified, clean kernel | 0 false positives / 300 |

Note the cell count is identical in both rows: **the strategy changed, not the
budget**. Widen n and the uniform number goes to zero while the stratified one
stays at 100%. This is the same move as poisoning the buffer pool — construct
the dangerous case instead of waiting to land on it — applied to the sampler
instead of the gate.

## The dirty-buffer gate (and the assumption it falsified)

Buffer pooling introduced a bug class the gates were built before: a pooled
buffer is **not zero-initialized**, so a kernel that assumes zeros (the rowmax
`atomicMax` accumulator does) is correct on step one and wrong on step two.
That is a **state** bug — no single call is wrong, the *sequence* is — the
family no oracle can reach, because an oracle is a function of one call.

The premise was that the gate suite is structurally blind to this, since every
gate runs on first-acquisition (zeroed) memory. **Mutation-testing the gate
falsified that.** Deleting the `clearBuffer` and re-running:

```
B2B MLP chain (LUT) failed verification: out mismatch @0 (5x16x6x3)
                                                          ^ the SECOND shape
```

The plain gate catches it — not by design, but because the sweep's own four
shapes recycle each other's buffers (they land in one power-of-2 pool bucket)
and an uncleared max only grows. The suite had **incidental** dirty coverage
nobody designed and nobody documented.

Incidental is the problem. It depends on the sweep having ≥2 shapes, on those
shapes colliding in one bucket, and on residue exceeding the real value.
Shorten the shape list and the coverage silently evaporates — with the gate
still green. So the coverage is now deliberate: `poisonPool` runs the chain at
**1e4 magnitude** first, so the residue dominates any value the gate can
produce, releases it, then sweeps again. Detection no longer depends on
ordering luck, and it covers the first shape too.

Verified:

| check | result |
|---|---|
| correct kernel, dirty re-gate | admitted (no false positive) |
| mutant (`clearBuffer` deleted), plain gate | caught at shape 2 (incidental) |
| mutant, poisoned gate | caught at shape 1 (deliberate) |
| init cost | 1546 → 1636 ms, **~90 ms one-time** |
| eleven suites + live 300-step run | green, no console errors |

The re-gate deliberately skips the 800-trial respec hunt — that hunt is about
the quantize spec and has nothing to do with buffer state, and skipping it is
what keeps the added cost at 90 ms instead of doubling gate time.

The lesson worth more than the gate: an instrument can have coverage it was
never designed for, and coverage you did not design is coverage you cannot
rely on. The only way to find out was to mutate the gate and watch which
shape it failed on.

## Where the remaining GPU time goes (a negative result, kept on purpose)

After the shared-operand fix, `bgemm` was the biggest line (~95 ms/step).
Phase timers inside it found the cost is neither upload nor the JS copy:

```
upload 1.8 ms | encode 0.3 | submit 0.2 | copy 7.1 | map 108.5 ms  <- GPU wait
```

The hypothesis was poor memory coalescing: the kernels used `row = gid.x`, so
lanes in a wave wrote `O[row*n + col]` **n floats apart** — 66 KB at the 16512
vocab. Swapping so `col` is the fast-varying dim makes the wave's output write
contiguous, and is bit-identical by construction (same serial k loop, same
order; only which lane computes which cell moves).

**It bought nothing.** Like-for-like on the same backend, seeded, hashes equal:

```
row-fast (before): logits 256x32x16512   69.00 ms/call   fnv 3e9d38cb
col-fast (after) : logits 256x32x16512   69.36 ms/call   fnv 3e9d38cb
```

So the swap was reverted. Two probes explain why — hold the output at 17 MB
and cut the compute 32x, and almost nothing changes:

| shape | MACs | output | ms/call |
|---|---|---|---|
| logits `256x32x16512` | 135M | 17 MB | 76.8 |
| `256x1x16512` (1/32 the compute) | 4.2M | 17 MB | **66.9** |
| `256x16512x32` (same MACs, tiny out) | 135M | 32 KB | 98.3 |

**The logits GEMM is transfer-bound**: ~67 of its 77 ms is moving 17 MB from
GPU to CPU (~250 MB/s through WebGPU's staging-copy + fence + map path). No
kernel tuning can reach it. The third row is a separate pathology — a long
serial k with only 8192 threads starves parallelism, which is why the split-K
path exists for the backward.

The readback is not removable: softmax must run in JS because WGSL's `exp` is
not correctly rounded, and a per-vendor `exp` would fork replicas — the same
constraint that keeps division off the GPU. So the real lever here is not the
kernel but the **vocabulary**: at c=32 with a 16512-token vocab the unembed is
528K of ~550K parameters, and its activations (BT x vocab) dominate every
transfer in the step. A smaller tokenizer would beat any kernel work available.

This section is kept because a measured negative result is worth as much as
the win above it: it closes off a plausible-sounding direction with numbers
instead of leaving it to be re-attempted later.

## Buffer pooling (GPU wall clock, zero math change)

Every dispatch used to create and destroy its GPU buffers (~19 per MLP chain
call, per layer, per step). They are now recycled through a size-bucketed
pool. Two hazards were found and handled, one by review and one by the bench:
a pooled buffer is not zero-initialized (the rowmax atomicMax accumulator is
now cleared in the encoder), and a bucketed readback maps more bytes than the
logical result (all six readbacks now map exactly the logical range — the
bench caught this as an out-of-bounds on the first pooled run).

A/B bench, same pinned seed, prepool baseline vs pooled build (DP4A, NVIDIA):

| config | prepool | pooled | grad hash | loss-sequence hash |
|---|---|---|---|---|
| c32 t32, float | 269 ms | 253 ms | `7ff11308` = | `37c1cdec` = |
| c32 t32, units | 342 ms | 315 ms | `62596547` = | `37c1cdec` = |
| c64 t48, float | 472 ms | 428 ms | `4f8c378` = | `652498b6` = |
| c64 t48, units | 510 ms | 460 ms | `9add4284` = | `652498b6` = |

6–10% wall clock, every hash bit-identical to the baseline. All init gates
pass through the pooled paths, the eleven-suite chain is green, and a full
300-step live training run completed with a silent audit and no console
errors.

## LUT vs DP4A: measured, then made a per-device race

A forced-LUT build (the mul LUT is the exported unit's complete extensional
behavior; DP4A is admitted only because it exact-gates bit-identical to it)
was benchmarked against the DP4A build, pinned seed, all hashes identical:

| config | DP4A | LUT |
|---|---|---|
| c32 float | 221 ms | 222 ms |
| c32 units | 295 ms | 335 ms |
| c64 float | 376 ms | 378 ms |
| c64 units | 425 ms | 475 ms |

First-run numbers were warm-up-skewed (a cold DP4A run read 253/428 on the
float rows — the moral: never conclude from one run). Stable picture: the
shipped float path TIES (DP4A's JS packing overhead cancels its dot-hardware
advantage at these sizes); DP4A is ~12% faster in int8-backward mode. Since
the ordering is device-dependent in principle and the choice is free
(bit-identical either way), init now RACES both gated backends at a
training-like shape (~40ms) and keeps the winner, instead of trusting one
machine's benchmark. The backend label reports the race result.

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

## The sign of zero (RDNA2 ISA audit)

Reading the RDNA2 shader ISA against our determinism assumptions confirmed
three of them on real hardware and exposed one blind spot in our own gates:

- `V_DOT4_I32_I8` is an exact packed int8 dot with int32 accumulate — the
  DP4A path's exactness is an ISA guarantee, not a tested coincidence.
- f32 add/multiply are 0.5 ULP (correctly rounded) — the epilogue mirror's
  foundation. `V_RCP_F32` is 1 ULP — division stays off the GPU, as designed.
- Rounding mode and denorm flushing are **runtime MODE-register state**
  (`S_ROUND_MODE` / `S_DENORM_MODE`, per wave, driver-controlled) — which is
  why the exact gates re-run on every device at every init rather than
  trusting a device model. (Denorm flush is additionally unreachable in the
  shipped math: the 1e-8 scale floor keeps every epilogue product ≥ ~1e-16 in
  magnitude, far above the ~1.2e-38 subnormal threshold.)
- FMA contraction (`V_FMA_F32`: one rounding, not two) looked like a second
  hazard — WGSL permits contracting the quantize's `x*inv + 0.5` — but turned
  out to be an immunity: adding 0.5 is exact except at binade crossings, and
  there the double-rounding anomaly stays on the same side of every integer
  (RNE tie parity), so `floor()` — hence the int8 — is identical either way.
  `test_b2b.js` asserts both halves: last-ulp fused-vs-stepped differences DO
  occur (~175k per 2.8M edge-targeted draws), and zero survive floor. The
  `+0.5` respec is contraction-immune by construction; `round(x/scale)` was
  not. No gate can forbid the compiler an fma, so this had to be a theorem,
  not a check.
- The blind spot: RDNA2 has non-IEEE instruction variants a compiler may pick
  — output modifiers and DX9-legacy multiplies that **flush −0 to +0**. Our
  gates compared f32 outputs with JS `!==`, for which `-0 !== 0` is *false*:
  a −0-flushing kernel would pass every gate, then fork the fleet at the sync
  guard, which hashes raw bits. All gate and audit comparisons now compare
  **bit patterns** (`bitDiff`), i.e. exactly what the replica hash sees.
  `test_gates.js` proves the fix non-vacuously: a `-0` planted where the units
  produce `+0` is invisible to `!==` (sanity-checked in the test) and flagged
  by the repaired `auditTile`.

One bug was caught during the rework, by the bit-identity check itself: the
fused q+k+v sum initially ran in f64 and rounded once, where the old code
rounded to f32 after each add — a last-ulp fork that would have split
replicas. Fixed by matching the rounding schedule (`Math.fround` per add).
Same lesson as the epilogue mirror: match the rounding schedule, not just
the values.
