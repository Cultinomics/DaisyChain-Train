---
title: DaisyChain-Web
emoji: 🌼
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Train a shared model P2P by opening a browser tab
---

# 🌼 DaisyChain-Web — train by opening a page

Part of **DaisyChain** → https://huggingface.co/DaisyChainAI

Open a link on two or more devices and they **train a shared model together,
peer-to-peer, right in the browser** — no install, no accounts. Devices connect
over **WebRTC** (like Snapdrop); each one computes **through the verified INT8
units** — the same emulated GPU logic as the rest of DaisyChain, run as a
**WebGPU** lookup-table matmul (with a CPU fallback for old machines) — and they
average gradients directly between peers.

> This is the browser-native version of DaisyChain: instead of setting up nodes,
> people join a training run by opening a URL.

**Docs:** [Getting started](docs/GETTING_STARTED.md) ·
[Architecture](docs/ARCHITECTURE.md) ·
[Verification (why the numbers are trustworthy)](docs/VERIFICATION.md) ·
[Troubleshooting](docs/TROUBLESHOOTING.md) ·
[Test results](TEST_RESULTS.md)

## Run it

```bash
npm install
npm start           # serves on http://localhost:8787
```

Open `http://localhost:8787` in two tabs (or two devices — see HTTPS note). They
find each other, connect P2P, and you click **Start training** on each. Watch the
shared loss fall on both.

Quick self-check of the training math (no browser):
```bash
npm test            # 2-peer gradient averaging: converges, replicas bit-identical
```

## How it works

| Piece | Role |
|-------|------|
| `server.js` | tiny **WebSocket signaling** server (introduces peers, relays WebRTC offers/ICE) + static host. It never sees the compute. |
| WebRTC data channels | **P2P** gradient exchange between browsers (STUN for NAT). |
| `public/verified_core.js` | the **verified INT8 units** in the browser — quantize → LUT multiply → dequant, STE backward. The emulated GPU logic doing the training compute. |
| `public/webgpu.js` | runs the verified multiply as a **WebGPU** LUT-matmul compute shader, with an automatic **CPU/JS fallback**. |
| `public/*.bin` | the units as lookup tables (`mul_lut`, `requant_lut`, `relu_lut`), exported from the trained weights by `daisychain/export_luts_web.py`. |
| `public/app.js` | the WebRTC mesh + the training loop + gradient averaging. |

Each peer starts from the **same** deterministically-seeded weights (no weight
broadcast needed), trains on its own data shard **through the verified units**,
and every step broadcasts its gradient and averages everyone's — so all peers
converge to the **same** model.

## What's verified
- **Through the units** (`node test_verified.js`): 2 peers training through the
  verified INT8 multiply converge and stay **bit-identical** (0.0 param diff).
- **Signaling** (`node -e ...`): peer discovery + relay works.
- **End-to-end in-browser**: two tabs connected over **real WebRTC**, both on
  **WebGPU running the verified INT8 units**, trained a shared model together —
  loss fell steadily, peers in sync. (`node test_core.js` also checks the plain
  float loop.)

## Regenerate the unit LUTs
The `.bin` tables are exported from the trained DaisyChain units:
```bash
cd ../daisychain && python export_luts_web.py     # writes mul/requant/relu LUTs into ../daisychain-web/public
```

## Who connects to whom (Snapdrop-style)

Peers are grouped by their **public IP**, so **only devices on the same network
auto-connect** — open the page on your phone and laptop at home and they find
each other, but a stranger viewing the same URL from another network does **not**
join your group. To connect across networks with people you invite, everyone
opens `?room=YOUR-CODE` (a shared private room). The server only relays WebRTC
handshakes; it never sees the training.

**Safety note:** WebRTC peers connect directly, so devices in your group can see
each other's IP address, and there's no gradient authentication — a malicious
peer could poison the shared model. Train only with devices/people you trust.
This is a proof of concept, not a hardened public service.

## Honest limits
- **Secure context required.** WebGPU and cross-device WebRTC need **localhost or
  HTTPS**. For real multi-device use, serve over HTTPS (a tunnel, a host, or a HF
  Space) — plain `http://192.168.x.x` won't get WebGPU.
- **No WebGPU? It falls back to CPU** — slower, but old machines (e.g. Windows XP/7
  via **Supermium**, old Macs via a compatibility browser) can still join at the
  CPU tier. WebGPU itself needs a GPU/driver with a modern backend, which very old
  hardware usually lacks.
- **Synchronous barrier**: the slowest peer paces each step. Peer-dropout handling
  is minimal (a timeout, then it proceeds) — this is a proof of concept, not
  hardened.
- **Small models only** — WebRTC bandwidth and browser compute cap the size. Same
  envelope as the rest of DaisyChain: pools compute, not memory.

---

**License:** MIT · **Author:** Dean Byrne (Quazim0t0) · **Org:** DaisyChainAI
