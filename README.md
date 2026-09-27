---
license: mit
tags:
  - distributed-training
  - old-hardware
  - int8
  - webgpu
  - webrtc
---

# 🌼 DaisyChain-Train — Old Hardware Training Pipeline

**Part of DaisyChain on 🤗 Hugging Face → https://huggingface.co/DaisyChainAI**
Model page (weights + card): https://huggingface.co/DaisyChainAI/DaisyChain-Train

---

> **In plain terms:** DaisyChain-Train lets you use **old / spare machines** to
> train neural networks. The training runs through **emulated GPU logic** —
> verified INT8 units (GUDA-style) that stand in for a GPU's math — so machines
> *without* a modern GPU can still do the work. Chain several together and they
> train one shared model as a cluster.
> Before you rely on it, see what it **can't** do → [Limitations](docs/LIMITS.md).

**Use the hardware you already have to train.** Each machine runs the emulated
GPU logic (verified INT8 units — multiply / requantize / ReLU) to compute the
model, and DaisyChain pools the machines data-parallel: device selection,
capacity-weighted sharding, gradient sync, a P2P setup, and a live dashboard.
Two ways to run — **Docker** or **Python**.

---

## ⚠️ Read this first
DaisyChain-Train is for **small models on spare hardware**. It **pools compute,
not memory** (the model must fit on one machine), scaling is **sublinear**, and
it is **not** a substitute for a real GPU on large models. Full envelope in
**[docs/LIMITS.md](docs/LIMITS.md)** — please read it before relying on it.

---

## Feature list

### Python cluster trainer (`daisychain/`)
- **Data-parallel training across mixed machines** — each node trains its own
  shard; gradients combine into the exact full-batch gradient, replicas stay
  bit-identical.
- **Capacity-weighted sharding** — faster machines automatically take a
  bigger share of the batch.
- **Emulated GPU compute (verified INT8 units)** — `VerifiedLinear` layers
  run every forward multiply / requantize / ReLU through the bundled trained
  units; cluster-wide unit-invocation counts printed by rank 0.
- **Bring your own model** — any `Task` (`build_model` / `sample` / `loss`)
  via `DAISY_TASK`; template in `examples/my_task_template.py`.
- **Plain-float alternative task** — same cluster and pooling with ordinary
  float math.
- **Live dashboard** (`daisychain-dashboard`) — readiness banner, P2P
  connectivity scan, pooled cores/RAM, per-node capacity plan, live loss.
- **SpikeWhale control panel** (`spikewhale_panel`, `localhost:8899`) —
  sliders for model size / training settings, any HF dataset you can access
  (default streamed FineWeb-Edu), start/stop/re-adjust, live loss.
- **Docker demo cluster** — 3 nodes + dashboard in one command.
- **Windows helper** (`scripts\setup.bat`) and Tailscale mesh guide.

### DaisyChain-Web (`web/`) — browser P2P training
- **Zero-install nodes** — opening the page IS joining; devices on one
  network auto-group (Snapdrop-style, by public IP).
- **Private cross-network rooms** — `?room=CODE` with **host approval** for
  every join.
- **Full WebRTC mesh** — gradients travel peer-to-peer; the server only
  signals and serves static files, it never sees weights or gradients.
- **Leader-follower runs** — whoever presses Start sets width / sequence /
  batch-per-device / steps / learning rate for the whole group; config
  broadcast on the wire.
- **Mid-run join** — late devices are synced in (weights + step) and
  contribute from the next step.
- **Bit-identical replicas** — same seeded init, strict roster-order gradient
  averaging, deterministic Adam with identical state on every peer; verified
  live by per-step weight hashes.
- **Sync guard** — any weight-hash mismatch stops the run instead of
  training past a fork; the step roster forbids silent partial averages.
- **Gradient repair** — a follower missing a roster gradient re-requests it
  from the leader (8 steps retained), bit-exact, and the run continues.
- **Cross-device kernel probe** — every step, every device re-hashes a fixed
  seeded int8 GEMM through its live kernel; catches broken arithmetic that
  weight hashes cannot see.
- **Hardcoded FineWeb-Edu streaming** — the server reads random slices of
  the 10BT parquet shards straight off the HF CDN via HTTP range requests
  (pure-JS `hyparquet`); built-in corpus fallback offline.
- **Checkpoints** — download `.pt`, upload → **broadcast to the whole
  group**; validated (magic, dims, tokenizer vocab) before accepting.
- **Inference kit** — one self-contained HTML file with the trained weights
  baked in; generations offline, anywhere.
- **In-page generation** — prompt box on the trained model.
- **Old-hardware tier** — no WebGPU? The identical units run on CPU (same
  bits, so CPU and GPU devices co-train in one group). There is no
  plain-float path.
- **Large-message fragmentation** — multi-MB gradients/checkpoints chunked
  at 48 KB over the data channels.

### Verified compute & kernels (web)
- **Verified INT8 units everywhere** — block-scaled int8 GEMM: exact LUT
  products, exact int32 accumulation, bit-exact f32 epilogue with a pinned
  rounding schedule; scales derived in JS f64 (division never runs on GPU).
- **Backends, best-first** — DP4A hardware int8 dot → LUT compute shader →
  CPU mirror; every kernel **exact-gated at init** (bit-level compares) and
  demoted to the mirror on any mismatch.
- **Continuous random-cell audit** at live training shapes.
- **Fused attention kernels** — gather/scatter head-strided q·kᵀ and a·v
  straight from BT×C layout (CUTLASS ex. 36/52 style).
- **QKV dual-GEMM fusion** — shared left operand quantized once, one batch-3
  dispatch (ex. 45); bit-identical.
- **B2B MLP chain** — both MLP GEMMs back-to-back on GPU with fused per-row
  absmax reduction and on-device quantize (ex. 13 + 23); WGSL-exact respec
  with a fround-stepped JS mirror; fma-contraction-immune by construction.
- **Dispatch-optimized backward** — overlapped independent GEMMs, batch-3
  sibling fusions (ex. 05/24); bit-identical gradients; optional int8 STE
  backward path (dormant, 1.21× vs float).

### Verification stack (web)
- **Exact init gates** on every kernel, every device, every boot — including
  gates that "gate the gate" with discriminating boundary inputs.
- **IEEE-754 binary32 oracle** in exact BigInt arithmetic — proves the JS
  epilogue mirror is spec-correct (rejects the old mirror on 34% of inputs).
- **Metamorphic property suite** — reference-free relations + definitional
  absolutes; **4/4** on an externally-authored bug corpus, matching the
  differential gate.
- **RDNA2 ISA audit hardenings** — bit-level (−0-aware) gate comparisons;
  proof that FMA contraction cannot change the quantize.
- **Twelve-suite test chain** (`cd web && npm test`) — convergence, replicas,
  oracle, gates, properties, external corpus, **self-corpus** (the instruments
  scored against my own bugs), B2B, optimizer, transformer LM, int8 backward,
  non-finite quantize refusal;
  results in [web/TEST_RESULTS.md](web/TEST_RESULTS.md).
- **Dirty-buffer gate** — the pool is poisoned before a re-sweep so state bugs
  (a kernel assuming zeroed memory) are caught deterministically rather than
  by ordering luck.

### Documentation
- Python: [QUICKSTART](docs/QUICKSTART.md), [LIMITS](docs/LIMITS.md),
  [CUSTOM_TASK](docs/CUSTOM_TASK.md), [TAILSCALE](docs/TAILSCALE.md).
- Web: [Getting started](web/docs/GETTING_STARTED.md),
  [Architecture](web/docs/ARCHITECTURE.md),
  [Verification](web/docs/VERIFICATION.md),
  [Troubleshooting](web/docs/TROUBLESHOOTING.md).

---

## Quick start

### Docker (most reliable — one command)
```bash
docker compose -f docker/docker-compose.yml up --build
# open http://localhost:8080
```
Brings up a 3-node demo cluster + dashboard on one machine.

### Python (real machines)
On every machine (`pip install -e .`):
```bash
export MASTER_ADDR=100.101.102.10   # coordinator IP (Tailscale 100.x recommended)
export MASTER_PORT=29560
export WORLD_SIZE=3
export RANK=0                        # 1, 2, ... on the others
export GLOO_SOCKET_IFNAME=tailscale0 # your mesh / LAN NIC
daisychain-train
```

### Windows helper
```bat
scripts\setup.bat
```
An interactive menu: Docker, Python node, or just install deps.

Full walkthrough: **[docs/QUICKSTART.md](docs/QUICKSTART.md)**.

### 🐋 SpikeWhale control panel (sliders → real training)
```bash
python -m daisychain.spikewhale_panel
# open http://localhost:8899
```
A web control panel: pick model size / training settings with sliders, choose any
HuggingFace dataset you have access to (default: streamed FineWeb-Edu), hit
Start, and watch the live loss. Stop and re-adjust any time with
**← Back to settings**. Launches the real DaisyChain training underneath.

### 🌐 DaisyChain-Web (train by opening a browser tab)
```bash
cd web && npm install && node server.js
# open http://localhost:8787 on every device
```
Zero-install browser training: devices on the same network auto-group
(Snapdrop-style) and train a shared model **peer-to-peer over WebRTC**, computing
through the same verified INT8 units (WebGPU, with the identical units on CPU
for machines without it — there is no plain-float path). Private cross-network
rooms via `?room=CODE` with **host approval** — the room creator accepts each
device before it can join. Includes gradient averaging with a deterministic
Adam optimizer (identical state on every peer, nothing extra over the wire),
checkpoint **download** (.pt) and **upload → broadcast** so one device can
restore the whole group after a failure.

**Live demo:** https://huggingface.co/spaces/Quazim0t0/DaisyChain-Web

### Recent updates (July 2026) — DaisyChain-Web

**Verification stack** — the browser trainer's correctness is now *checked by
things that run*, not argued ([full results](web/TEST_RESULTS.md)):
- **IEEE-754 oracle** (`web/test_ieee.js`): a binary32 oracle built from the
  standard in exact BigInt arithmetic proves the JS epilogue mirror is
  spec-correct — and rejects the old round-once mirror on 34% of inputs.
- **Metamorphic properties + oracle mutation scoring** (`test_metamorphic.js`,
  `test_corpus.js`): properties needing no reference implementation, scored
  against an externally-authored bug taxonomy — **4/4**, matching the exact
  differential gate's 4/4. Relations own the loop bugs; two definitional
  absolutes (ReLU output range, a unit-scale integer anchor) own the value
  bugs no relation can see.
- **Exact kernel gates on every live kernel**, a continuous audit at live
  shapes, and a cross-device kernel probe (same seeded int8 GEMM, same hash on
  every honest device, any backend). The audit's sampling was rebuilt against
  a named bug class: its old constants (6 cells, 2% of GEMMs) bounded an
  overhead that had never been measured (auditing *every* GEMM costs <0.01% of
  a step), and uniform random cells cannot see a last-row/column bug at a
  16512-wide output. Sampling is now **stratified** — the first cells are the
  structural danger points, chosen deliberately. On a last-column bug, same
  cell budget: uniform caught **5/300** audits, stratified **300/300**, with
  zero false positives.
- **RDNA2 ISA audit**: reading a real GPU's shader ISA against our
  determinism assumptions confirmed three of them on silicon (exact packed
  int8 dot; correctly-rounded f32 add/mul; 1-ULP reciprocal — division stays
  off the GPU) and produced two hardenings. (1) Real ISAs have non-IEEE
  variants that **flush −0 to +0**; JS `!==` can't see that (`-0 !== 0` is
  false), so all gates and audits now compare **bit patterns** — exactly what
  the replica hash sees. (2) FMA contraction of the quantize's `x·inv + 0.5`
  (one rounding instead of two) turned out to be **floor-invisible by
  construction** — proven in `test_b2b.js` with 175k+ last-ulp anomalies at
  binade edges, zero surviving `floor()`. Rounding mode and denorm flushing
  are runtime driver state on real hardware, which is why every device
  re-runs the exact gates at every init.

**Training data** — FineWeb-Edu (10BT sample) is the hardcoded dataset. The
Space reads random slices of the parquet shards straight off the HF CDN with
range requests (pure-JS `hyparquet`, SNAPPY) and serves plain text at `/data`
— no dependency on the datasets-server rows API and its 503s.

**Resilience** — the sync guard now *repairs* instead of halting: a roster
gradient that reached the leader but not some follower (asymmetric WebRTC
mesh) is re-requested from the leader, bit-exact, and the run continues. The
guard still stops anything that would fork the weights.

**CUTLASS-style kernel work**, each step proven bit-identical or exact-gated:
- **Dispatch-optimized backward** (ex. 05/24): independent GEMMs overlapped,
  sibling trios fused into batch-3 dispatches — bit-identical gradients,
  dormant int8-backward path down from 1.63× to 1.21× vs float.
- **QKV dual-GEMM fusion** (ex. 45): q/k/v share one left operand — quantized
  once, one batched dispatch, zero changed bits.
- **B2B MLP chain** (ex. 13 + 23): both MLP GEMMs back-to-back on the GPU with
  a fused per-row absmax reduction; the intermediate is quantized on-device
  via a WGSL-exact respec (`floor(f32(x·invScale)+0.5)` — no GPU division)
  whose fround-stepped JS mirror keeps mixed GPU/CPU fleets bit-identical.

**Profile-driven speed work** — every change below is bit-identical (gradient
and loss hashes unchanged), so none of it trades correctness for wall clock:
- **Buffer pooling**: GPU buffers are recycled by size bucket instead of being
  created and destroyed per dispatch (~19 per MLP call, per layer, per step).
  **6–10% faster**, every hash unchanged.
- **Shared-operand embedding GEMMs**: profiling put two f32 backward GEMMs at
  **55% of the entire step**, and both consumed the same `dlogits` operand —
  ~17 MB at the 16512-token vocab, uploaded *twice*. One upload, one encoder,
  one submit: that pair went 205 → 90 ms and the step **12% faster**. The
  fusion is gated bit-for-bit against the two calls it replaces.
- **A negative result, kept on purpose**: the remaining hot kernel looked
  cache-hostile (adjacent lanes wrote 66 KB apart), but making the writes
  contiguous changed nothing. Two probes explain why — holding the output at
  17 MB while cutting compute 32× barely moved the time. The logits GEMM is
  **transfer-bound, not compute-bound**, and the readback cannot be removed
  because softmax must stay in JS (WGSL's `exp` is not correctly rounded, and
  a per-vendor `exp` would fork replicas). The real lever there is the
  vocabulary, not the kernel.
- **An init backend race was tried and removed**: it tied on the shipped path,
  cost ~430 ms of init, and made the backend vary between page loads, which
  silently invalidated three A/B comparisons before it was caught. A knob that
  changes what you are measuring is worse than a fixed choice.

**Dirty-buffer gate — and the assumption it falsified.** Pooling introduced a
bug class the gates predate: a pooled buffer is *not* zero-initialized, so a
kernel that assumes zeros is right on step one and wrong on step two. That is
a **state** bug, where no single call is wrong and the *sequence* is, which is
the family no oracle can reach. The assumption was that the gates were blind
to it. Mutation-testing the gate proved otherwise: deleting the zeroing made
the plain gate fail at its *second* shape, because the sweep's own shapes
recycle each other's buffers. The suite had **incidental** coverage nobody
designed, which is coverage nobody can rely on — shorten the shape list and it
evaporates with the gate still green. It is now deliberate: the pool is
poisoned with 1e4-magnitude residue before a re-sweep, so detection no longer
depends on ordering luck. ~90 ms one-time.

**Scoring the oracles against my OWN bugs** (`web/test_selfcorpus.js`) — the
external corpus measures kernel bugs someone else wrote down, so this suite
asks the harder question: what do the instruments score against the four real
bugs of the month? Properties **0/2** on the data-plane pair (the `c·out`
theorem again), differential **2/2** — but half the bugs were not in the
kernels at all. A dead gate is a bug in a *checker*, caught only by mutating
the gate; a stalled roster gradient is a bug in the *protocol*, where every
computed value on every peer was correct, so no data oracle could fire. Those
needed different instruments, not better oracles.

All twelve test suites (`cd web && npm test`) pass; results with methodology in
[`web/TEST_RESULTS.md`](web/TEST_RESULTS.md).

---

## How it works

Each machine runs the **same** command; they form a cluster and train one shared
model. Two things happen:

1. **The compute runs through the emulated GPU logic.** By default the model is
   built from `VerifiedLinear` layers, so every forward multiply / requantize /
   ReLU is done by the **bundled verified INT8 units** (`daisychain/verified/`) —
   the emulated GPU math. Rank 0 prints **cluster-wide unit-invocation counts**
   so you can see the emulated logic doing the work.
2. **The machines are pooled data-parallel.** Each node trains on its own shard;
   gradients are capacity-weighted and combined into the exact full-batch
   gradient, so replicas stay **bit-identical**. Faster machines automatically
   take a bigger share.

```
  old machine A ─┐
  old machine B ─┼─►  each runs the emulated GPU logic on its shard  ─►  one model
  old machine C ─┘        (gradients combined across the cluster)
```

## Bring your own model
DaisyChain-Train trains any **Task** (`build_model` / `sample` / `loss`). Copy
`examples/my_task_template.py`, set `DAISY_TASK=your_module:YourTask`. Use
`VerifiedLinear` (see `daisychain/verified_task.py`) to run your model's compute
through the emulated units. See **[docs/CUSTOM_TASK.md](docs/CUSTOM_TASK.md)**.

## Plain-float alternative
To skip the emulated units and train with normal float math on each machine, set
`DAISY_TASK=daisychain.example_task:ExampleTask`. Same cluster, same pooling —
the model math just runs as ordinary float instead of through the verified units.

## The dashboard
`daisychain-dashboard` (or the Docker service) serves a Tailwind page at `:8080`
— readiness banner, P2P connectivity scan, pooled cores/RAM + capacity plan
(per-node device, weight, batch), and live training loss.

## Networking
Use **Tailscale** for a P2P mesh so machines on different networks get stable IPs
on one interface — **[docs/TAILSCALE.md](docs/TAILSCALE.md)**.

---

## Layout
```
daisychain/cluster.py        capacity-weighted CPU/GPU data-parallel trainer
daisychain/train.py          entry point (daisychain-train)
daisychain/verified/         bundled trained N/N units + VerifiedLinear (train through them)
daisychain/verified_task.py  default task: forward runs on the verified units
daisychain/example_task.py   plain-float alternative task
daisychain/task.py           the Task interface + loader
daisychain/dashboard/        agent + P2P scanner + Tailwind server
docker/                      Dockerfile, dashboard image, compose (demo cluster)
scripts/setup.bat / setup.sh interactive setup helpers
config/                      nodes + cluster env examples
examples/my_task_template.py starting point for your own model
test_verified_units.py       full-domain checks of every shipped verified path (25)
docs/                        QUICKSTART, LIMITS, CUSTOM_TASK, TAILSCALE
daisychain/spikewhale_task.py   trains the real SpikeWhale on streamed HF datasets
daisychain/spikewhale_panel.py  slider control panel (localhost:8899)
web/                         DaisyChain-Web: P2P browser training (WebRTC + WebGPU)
export_luts_web.py           regenerates web/public LUTs from the trained units
```

## Recent updates (September 2026) — requant rounding, bounded units, non-finite guards

Everything below is measured, and `python test_verified_units.py` (25 checks) plus
`cd web && npm test` (twelve suites) reproduce it.

**The requant rounds instead of truncating.** `NeuralRequant16` was
`sat_int8(x >> 8)`, and Python's `>>` on negatives is an arithmetic floor, so every
requantized activation carried a systematic bias. Over the full int16 domain:

| requant | mean error | disagrees with the other on |
| --- | --- | --- |
| floor `x >> 8` (old) | **-0.4981 LSB** | 32,640 / 65,536 |
| round-half-up `(x + 128) >> 8` (now) | +0.0020 LSB | |

The shipped `requant16.pt` implemented the floor exactly, so this was real behaviour,
not a stale docstring. It was **retrained** against the new reference and saved only
at 65,536/65,536 (a 99.9% unit would silently break the LUT equivalence the browser
relies on), and `web/public/requant_lut.bin` was **re-exported** to match: the
invariant is *neural forward == LUT == native integer op*, so a table that lags its
unit is a fork. Tie rule, stated precisely: this is half-UP; Python/numpy/torch
`round` is half-to-even. The two differ on exactly the 128 exact ties (0.2% of
inputs) and both are unbiased to within 0.002 LSB. The Python fleet quantizes with
`np.round` and the browser with `floor(x + 0.5)`; the two fleets never co-train, so
that cannot fork a group.

**Bounded elementwise units.** The GEMM backends already capped their temporaries;
the units right after them did not. `relu_array` / `requant_array` received the
whole activation matrix in the proof path at ~1.5–2 KB per element (262,144
elements = **407 MB**), and the multiply-LUT build pushed 262,144 rows through the
atom net at startup (**277 MB RSS on every node**). Both are now blocked:
407 MB → **2 MB, flat with N**; 277 MB → **1 MB**. Elementwise blocking is exact
(no accumulation order), verified bit-identical at N = 1 / 999 / 32768 / 32769 /
100000. `dataset()` builders are vectorised (65,536 Python iterations → one numpy
pass, 0.015 s), which is what makes retraining a unit practical at all.

**A NaN or Inf no longer quantizes to zeros.** A float → int8 conversion has no
answer for NaN/Inf, and both trainers gave a silent wrong one:

| path | input | old result |
| --- | --- | --- |
| Python `qat.py` | `[0.5, nan, 3.0]` | scale = NaN → **`[0, 0, 0]`** (only a RuntimeWarning) |
| Python `qat.py` | `[0.5, inf, 3.0]` | scale = Inf → **`[0, 0, 0]`** |
| browser quantizers | `[0.5, Inf, 3]` | scale = Infinity → **`[0, 0, 0]`** |
| browser quantizers | `[0.5, NaN, 3]` | NaN skipped by the \|max\| scan, stored as **0** |

One bad value zeroed the whole layer's int8 input and training carried on. Now
`VerifiedLinear` raises `FloatingPointError`, and every browser quantizer
(`quantize`, `quantizeRows`, `quantizeCols`, `rowAbsMax`, `quantizeHeadCols`, and
the transformer's `quantizeColsAsRows`) throws `NonFiniteError`, naming how many
values were bad. Finite inputs are untouched, so builds with and without the guard
still co-train. Both new tests fail against the old code (Python: 2 checks; web:
`test_nonfinite.js`, 12) and pass now.

## Recent updates (August 2026) — verified compute path

Three changes to `daisychain/verified/`, each measured rather than asserted.
`python test_verified_units.py` (24 checks) reproduces all of it.

**Bounded GEMM memory.** `LUTBackend.gemm` and `NeuralBackend.gemm` materialized
the whole `(m, n, k)` product block before reducing it, so peak memory was cubic
in layer width. At the SpikeWhale panel's maximum (hidden 768, sequence 512) a
single layer's forward allocated **2.4 GB** — on the spare hardware this project
exists to use. Both now block the `m` axis. The contraction axis `k` is untouched,
so the sum and its order are unchanged and results stay bit-identical:

| GEMM | before | after |
| --- | --- | --- |
| 64x64x64 | 2.3 MB | 2.3 MB (unchanged path) |
| 256x256x256 | 135.8 MB | 68.9 MB |
| 512x768x768 (panel max) | 2426.9 MB | **77.2 MB** |
| neural backend, 64x96x96 | 519.2 MB | **57.5 MB** |

Caps are class attributes (`LUTBackend.max_block_bytes`,
`NeuralBackend.max_products`), tunable per instance. Below the cap the LUT path
takes the original single-block branch unchanged.

The two backends need *different* accounting, which is worth knowing before tuning
them: a LUT product costs 8 bytes, but a neural product is split into four nibble
pairs through a 128-wide net and costs ~880 bytes. Applying the LUT's byte budget
to the neural path computes a block 110x too large and never splits at all.

**The multiply LUT is certified, not trusted.** `build_luts()` returned its tables
unchecked while the docs described a self-certify gate. Because the table *is* the
complete finite domain, certifying it **is** the exhaustive verification:
`LUTBackend.__init__` now checks all 65536 entries against signed integer multiply
and raises on mismatch. Cost: **0.5 ms**.

**Full-domain tests for the paths that ship.** The units' exhaustive verification
covers the scalar entry points (`NeuralMul8.verify()` walks `mul()`); production
runs the batched ones — `mul_array`, `relu_array`, `requant_array` — and the LUTs
built from them, which are different code. `test_verified_units.py` checks every
shipped path against golden integer arithmetic over its complete domain. All pass
today; nothing would have caught a regression before.

**Two guards against silent no-ops.** `gemm_int8` enforces the int32 accumulator
bound (`MAX_INT32_K = 131071`) instead of asserting it in a comment, since layer
width is user-selectable and an overflow would produce wrong numbers rather than
raise. `instrument.require(**minimums)` fails when a verified unit was
under-invoked, and refuses to report at all while counting is disabled — a zero
from a probe that was switched off is not evidence that the units did not run.

---

## Install
```bash
pip install torch numpy psutil
pip install -e .          # exposes: daisychain-train, daisychain-agent, daisychain-dashboard
```
Requires Python ≥ 3.9, PyTorch ≥ 2.0. Multi-node is reliable on **Linux/macOS**;
on **Windows use Docker/WSL** (see [Limitations](docs/LIMITS.md)).

---

## Links
- **DaisyChain on Hugging Face:** https://huggingface.co/DaisyChainAI
- **This model:** https://huggingface.co/DaisyChainAI/DaisyChain-Train

**License:** MIT · **Author:** Dean Byrne (Quazim0t0) · **Org:** DaisyChainAI

## Citation
```bibtex
@misc{byrne2026daisychain,
  title        = {DaisyChain-Train: An Old Hardware Training Pipeline},
  author       = {Byrne, Dean (Quazim0t0)},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/DaisyChainAI/DaisyChain-Train}},
  note         = {Chain spare/old machines into a data-parallel training cluster}
}
```

**Dean Byrne (Quazim0t0)** · 2026
