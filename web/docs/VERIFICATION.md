# Verification — why you can trust the numbers

DaisyChain-Web's core claim is strong: **every device — any GPU, any driver,
or plain CPU — computes bit-identical results**, so replicas can be compared
by hashing raw bytes. This document explains the layers that make that claim
*checked by things that run*, not argued. Full current results:
[TEST_RESULTS.md](../TEST_RESULTS.md).

## The verified INT8 units

All model multiplies run through one primitive: a **block-scaled int8 GEMM**.

1. Inputs are quantized per row / per column: `scale = max(|row|)/127`
   (floored at 1e-8), values to int8.
2. Products come from `mul_lut` — a 65536-entry table of **exact** int8
   products — accumulated in **int32** (exact; no overflow at these sizes).
3. The float epilogue is fixed to a bit-exact rounding schedule:
   `epi(s,a,b) = f32(f32(f32(s)·a)·b)` — round to f32 after the int→float
   conversion and after **each** multiply.

Steps 1–2 are integer-exact everywhere by construction. Step 3 is where
"bit-identical across devices" is usually lost — so it is pinned to WGSL's
guarantees (add/multiply are correctly rounded; division is not, so **no
division ever runs on the GPU** — scales are derived in JS f64, which is
exactly rounded and device-identical).

## Layer 1: exact init gates (every device, every boot)

No kernel computes a single training value before passing its gate: run the
kernel and the JS mirror on a sweep of shapes (ragged ones included) and
compare — int32 accumulators exactly, f32 outputs **at the bit level**. Any
mismatch demotes the device to the CPU mirror. Bit-level matters: JS `!==`
treats `-0 === 0`, but real ISAs have non-IEEE modes that flush −0 to +0, and
the replica hash *would* see that. The gates compare exactly what the hash
sees.

Gates re-run at **every** init because floating-point behavior is runtime
state on real hardware (rounding mode and denorm flushing are per-wave MODE
registers on RDNA2, set by the driver) — a device model can't be trusted
across boots; a fresh gate can.

Some gates additionally **gate the gate**: the B2B chain gate hunts for an
input where the old and new quantize specs actually disagree and requires the
GPU to match the new one — so a pass is something the old spec would fail,
not a vacuous agreement.

## Layer 2: continuous audit (every run, live shapes)

Init gates use test shapes; the **audit** samples random output cells of the
*live* GEMMs during training and recomputes them through the units. A kernel
that is correct at gate shapes but wrong at live shapes (stride bugs, padding
bugs) is caught while it trains.

## Layer 3: the kernel probe (every step, cross-device)

The weight hash cannot catch a device whose *kernel* is wrong — weights only
depend on the gradient bytes everyone receives. So each step every device
also publishes a **probe hash**: the same seeded int8 GEMM through its live
kernel. Same math ⇒ same hash, on every honest device, any backend.

## Layer 4: the referee — an IEEE-754 oracle

Who checks the JS mirror? `test_ieee.js` builds a binary32 oracle **from the
IEEE-754 definition in exact BigInt arithmetic** — no `Math.fround` anywhere
in it, round-to-nearest-even, subnormals, signed zero. The mirror's epilogue
agrees with the oracle on 500k+ checks, including a tie-to-even ladder around
2²⁴ — and the oracle **rejects** the older round-once mirror on 34% of
inputs, which is what makes the agreement meaningful.

## Layer 5: properties and mutation scores

`test_metamorphic.js` holds correctness properties that need **no reference
implementation**: relations (permuting rows permutes outputs; a zero row
yields zeros; batches decompose; single-cell sensitivity) plus two
**definitional absolutes** — fused-ReLU output can never be negative, and at
unit scales the output must equal the exact integer dot product. The split is
principled: if `out` satisfies every relation, so does `2·out` — relations are
provably blind to value bugs, absolutes are not.

`test_corpus.js` then **mutation-scores the checkers themselves** against an
externally authored bug taxonomy
([dipankarsarkar/gpuemu-corpus](https://huggingface.co/datasets/dipankarsarkar/gpuemu-corpus)):
each ported bug must be caught (4/4 properties, 4/4 differential), and a
control run must stay clean. A checker that has never rejected anything is
decoration; these have a scoreboard.

## Hardware ground truth (RDNA2 ISA audit)

Reading a real GPU ISA against the assumptions confirmed on silicon:
`V_DOT4_I32_I8` is an exact packed int8 dot (the DP4A path is exact by ISA
guarantee); f32 add/mul are 0.5 ULP; reciprocal is 1 ULP (division stays off
the GPU). It also produced two hardenings: the bit-level gate comparisons
above, and a proof that FMA contraction of the quantize's `x·inv + 0.5`
(one rounding instead of two — a choice WGSL leaves to the compiler) is
**floor-invisible by construction**: last-ulp anomalies occur at binade
edges, but RNE tie parity keeps both rounding schedules on the same side of
every integer, so the quantized int8 is identical either way. Since no gate
can forbid a compiler an fma, that one had to be a theorem, not a check —
`test_b2b.js` asserts both halves (anomalies exist; zero survive `floor`).

## What this does NOT protect against

A **malicious** peer that runs the correct math but *lies* — sends a crafted
gradient — is not caught by any of this; there is no gradient authentication.
The verification stack proves the *computation* is right on every honest
device. Trust in the *participants* is still yours to establish.
