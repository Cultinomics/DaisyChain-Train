---
license: mit
tags:
  - distributed-training
  - old-hardware
  - cluster
  - verified-units
  - pytorch
---

# 🌼 DaisyChain — Old Hardware Training Pipeline

> **In plain terms:** DaisyChain lets you use **old / spare machines** to train
> neural networks. The training runs through **emulated GPU logic** — verified
> INT8 units (GUDA-style) that stand in for a GPU's math — so machines *without*
> a modern GPU can still do the work. Chain several together and they train one
> shared model as a cluster.
> Before you rely on it, see what it **can't** do → [Limitations](docs/LIMITS.md).

**Use the hardware you already have to train.** Each machine runs the emulated
GPU logic (verified INT8 units — multiply / requantize / ReLU) to compute the
model, and DaisyChain pools the machines data-parallel: device selection,
capacity-weighted sharding, gradient sync, a P2P setup, and a live dashboard.
Two ways to run — **Docker** or **Python**.

> Built by **DaisyChainAI**. Point it at your model + data and it trains across
> whatever old machines you have, through the emulated GPU logic.

**Repositories:** [GitHub](https://github.com/quzi93/DaisyChain-Train) · [🤗 HuggingFace](https://huggingface.co/DaisyChainAI/DaisyChain-Train)

---

## ⚠️ Read this first
DaisyChain is for **small models on spare hardware**. It **pools compute, not
memory** (the model must fit on one node), scaling is **sublinear**, and it is
**not** a substitute for a real GPU on real models. Full envelope in
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
On every machine (`pip install daisychain` or `pip install -e .`):
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

---

## How it works

Each machine runs the **same** command; they form a cluster and train one shared
model. Two things happen:

1. **The compute runs through the emulated GPU logic.** By default the model is
   built from `VerifiedLinear` layers, so every forward multiply / requantize /
   ReLU is done by the **bundled verified INT8 units** (`daisychain/verified/`)
   — the emulated GPU math. Rank 0 prints **cluster-wide unit-invocation counts**
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
DaisyChain trains any **Task** (`build_model` / `sample` / `loss`). Copy
`examples/my_task_template.py`, set `DAISY_TASK=your_module:YourTask`. Use
`VerifiedLinear` (see `daisychain/verified_task.py`) to run your model's compute
through the emulated units. See **[docs/CUSTOM_TASK.md](docs/CUSTOM_TASK.md)**.

## Plain-float alternative
If you'd rather skip the emulated units and just train with normal float math on
each machine, set `DAISY_TASK=daisychain.example_task:ExampleTask`. Same cluster,
same pooling — the model math just runs as ordinary float instead of through the
verified units.

## The dashboard
`daisychain-dashboard` (or the Docker service) serves a Tailwind page at
`:8080` — readiness banner, P2P connectivity scan, pooled cores/RAM + capacity
plan (per-node device, weight, batch), and live training loss.

## Networking
Use **Tailscale** for a P2P mesh so machines on different networks get stable
IPs on one interface — **[docs/TAILSCALE.md](docs/TAILSCALE.md)**.

---

## Layout
```
daisychain/cluster.py        capacity-weighted CPU/GPU data-parallel trainer
daisychain/task.py           the Task interface + loader
daisychain/train.py          entry point (daisychain-train)
daisychain/example_task.py   default runnable task (plain float)
daisychain/verified/         bundled trained N/N units + VerifiedLinear (train through them)
daisychain/verified_task.py  example task whose forward runs on the verified units
daisychain/dashboard/        agent + P2P scanner + Tailwind server
docker/                      Dockerfile, dashboard image, compose (demo cluster)
scripts/setup.bat / setup.sh interactive setup helpers
config/                      nodes + cluster env examples
examples/my_task_template.py starting point for your own model
docs/                        QUICKSTART, LIMITS, CUSTOM_TASK, TAILSCALE
```

## Install
```bash
pip install torch numpy psutil
pip install -e .          # exposes: daisychain-train, daisychain-agent, daisychain-dashboard
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0. Multi-node is reliable on **Linux/macOS**;
on **Windows use Docker/WSL** (see LIMITS).

---

**License:** MIT · **Author:** Dean Byrne (Quazim0t0) · **Org:** DaisyChainAI

## Citation

```bibtex
@misc{byrne2026daisychain,
  title        = {DaisyChain: An Old Hardware Training Pipeline},
  author       = {Byrne, Dean (Quazim0t0)},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/DaisyChainAI/DaisyChain-Train}},
  note         = {Chain spare/old machines into a data-parallel training cluster}
}
```

**Dean Byrne (Quazim0t0)** · 2026
