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
  `test_corpus.js`): properties needing no reference implementation, then
  scored against an externally-authored bug taxonomy — properties catch 2/2
  loop bugs, the exact differential gate catches 4/4.
- **Exact kernel gates on every live kernel**, a continuous random-cell audit
  at live shapes, and a cross-device kernel probe (same seeded int8 GEMM,
  same hash on every honest device, any backend).

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

All ten test suites (`cd web && npm test`) pass; results with methodology in
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
docs/                        QUICKSTART, LIMITS, CUSTOM_TASK, TAILSCALE
daisychain/spikewhale_task.py   trains the real SpikeWhale on streamed HF datasets
daisychain/spikewhale_panel.py  slider control panel (localhost:8899)
web/                         DaisyChain-Web: P2P browser training (WebRTC + WebGPU)
export_luts_web.py           regenerates web/public LUTs from the trained units
```

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
