# Honest limits — read before you rely on DaisyChain

DaisyChain is a real tool for a **specific** job: chaining spare/old machines to
train **small** models faster by pooling their compute. It is not magic, and it
is easy to misapply. Here's the honest envelope.

## What it does
- **Data-parallel** training: every node holds a full copy of the model and
  trains on its slice of the data; gradients are averaged across the network.
- **Capacity-weighted**: each node measures its own speed (CPU or GPU) and takes
  a proportional batch, so a faster machine does more. GPU nodes auto-join and
  carry more load; CPU-only nodes still participate.

## What it does NOT do (expect these)
- ❌ **Not GPU-class training.** N pooled old CPUs/GPUs are still that class of
  hardware. A single modern GPU will beat the whole cluster for real models.
- ❌ **Pools compute, not memory.** Every node needs the *whole* model in RAM
  (or VRAM). You **cannot** train a model bigger than a single node can hold.
  Chaining 5 laptops does not give you one big machine.
- ❌ **Sublinear scaling.** Gradient sync over WiFi is bandwidth-bound, and the
  **slowest node paces every step** (synchronous). Each added node helps less
  than the last; a slow link can erase the benefit.
- ❌ **Not for huge/modern models.** Transformers/large CNNs at CPU (or old-GPU)
  speed are impractically slow.

## The sweet spot
Small models (small MLPs, tabular/classifier tasks, small fine-tunes),
a handful of machines, model fits on each, you want more throughput and don't
have a GPU handy. Education, research, retro/old-hardware hobby clusters.

## Hardware / OS
- **Multi-node reliably runs on Linux/macOS.** On **Windows**, gloo tensor
  collectives are unstable — use **Docker** or **WSL**. Single-machine use is
  fine on Windows.
- **Old GPUs** must be new enough for current PyTorch (compute capability ≳ 5.0,
  roughly GTX 900 / Maxwell and up). Kepler/Fermi-era cards won't work with
  modern torch.
- Put all nodes on the same **interface** (Tailscale `tailscale0`, or a real LAN
  NIC) via `GLOO_SOCKET_IFNAME` — not a VPN/Docker/WSL virtual NIC.

## The one-line version
You can chain old machines to train a **small** model together, faster by
throughput, with honest sublinear scaling — and DaisyChain won't pretend to be
more than that.
