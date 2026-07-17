# Getting started with DaisyChain-Web

DaisyChain-Web trains a small transformer language model **peer-to-peer in the
browser**. Every device that opens the page becomes a training node: it
computes through the verified INT8 units (WebGPU, or the identical math on
CPU), exchanges gradients with the other devices over WebRTC, and all replicas
stay **bit-identical** — same weights, same loss, on every device, every step.

## The fastest way: the hosted Space

Open **https://huggingface.co/spaces/Quazim0t0/DaisyChain-Web** on two or more
devices. That's it — no install. Devices on the **same network** find each
other automatically (Snapdrop-style, grouped by public IP). To train with
devices on *different* networks, use a room code (below).

## Run it yourself

```bash
npm install
npm start            # serves on http://localhost:8787
```

Open `http://localhost:8787` in two browser tabs to try it on one machine.

> **HTTPS matters.** WebGPU and cross-device WebRTC require a *secure
> context*: `localhost` or HTTPS. Plain `http://192.168.x.x` from another
> device will NOT get WebGPU (and may not connect at all). For real
> multi-device runs, serve over HTTPS — a tunnel (`cloudflared`, `ngrok`), a
> host, or a Hugging Face Space.

## Rooms: training across networks

Everyone opens the same URL with a shared code:

```
https://<host>/?room=MY-SECRET-CODE
```

The first person in the room is the **host** and must **approve each device**
before it can join (an Accept button appears per request). The room code is
never guessable from the public page; pick something private.

## Starting a run

1. Wait until the devices appear in each other's peer list.
2. On **one** device, pick the settings (that device's choices apply to the
   whole group):

   | Setting | Range | Default | What it does |
   |---|---|---|---|
   | Model width | 16–128 | 32 | embedding/channel width of the transformer |
   | Sequence length | 16–128 | 32 | context window in tokens |
   | Batch per device | 2–32 | 8 | each device adds this much batch — more devices = bigger effective batch |
   | Steps | 100–10000 | 300 | training steps for the run |
   | Learning rate ×1000 | 5–50 | 10 | Adam learning rate (10 = 0.01) |

3. Press **Start training**. Every device begins the same run: the starter
   broadcasts the config, everyone builds the same seeded weights, and the
   loss falls in lockstep on all of them.

The **replica diff / weight hash** readout is the trust signal: every honest
device shows the same hash at the same step. Training data streams from
**FineWeb-Edu** (served by the Space from the HF CDN's parquet shards);
devices that can't reach it fall back to a built-in corpus.

Devices that join **mid-run** are synced in: the leader ships them the current
weights and step, and they participate from the next step.

## After training: generate, save, share

- **Generate** — type a prompt and sample from the trained model, right on the
  page.
- **Download checkpoint (.pt)** — saves weights + step + config. **Load** it
  later on any device; the loader validates the tokenizer and dimensions, and
  loading on one device **broadcasts the checkpoint to the whole group**, so
  one person can restore everyone after a failure.
- **Download inference kit** — a single self-contained HTML file with the
  weights baked in. Open it anywhere, even offline, to run generations. Handy
  for sharing what the group trained.

## Checking the math (no browser needed)

```bash
npm test
```

runs all ten verification suites — convergence, bit-identical replicas, the
IEEE-754 oracle, kernel gates, metamorphic properties, the external bug
corpus, the B2B chain, optimizer, transformer, and the int8 backward. Results
with methodology live in [TEST_RESULTS.md](../TEST_RESULTS.md).

## Safety notes

- WebRTC is direct: devices in a group can see each other's IP addresses.
- There is no gradient authentication — a malicious peer could poison the
  model. Train with devices and people you trust. (Dishonest *math* is a
  different story: a device whose kernel computes wrong values is caught by
  the kernel probe and audits — see [VERIFICATION.md](VERIFICATION.md).)
- This is a proof of concept, not a hardened public service.

## More

- [ARCHITECTURE.md](ARCHITECTURE.md) — how the mesh, the training step, and
  the wire protocol work.
- [VERIFICATION.md](VERIFICATION.md) — why you can trust the numbers.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — common failure modes and what
  the log messages mean.
