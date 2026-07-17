# Architecture

How a page load becomes a training node, and how a handful of browsers stay
bit-identical while training one model together.

```
 browser A ──┐                       ┌── browser B
  WebGPU/CPU │   WebRTC data channels │  WebGPU/CPU
  verified   ├──────── gradients ─────┤  verified
  INT8 units │                        │  INT8 units
             └──── server.js (signaling + static + /data) ────┘
                    never sees weights or gradients
```

## The pieces

| File | Role |
|---|---|
| `server.js` | WebSocket **signaling** (introduces peers, relays WebRTC offers/ICE), static hosting, and the `/data` endpoint that streams FineWeb-Edu text. It never sees weights or gradients. |
| `public/app.js` | the WebRTC mesh, the training loop, gradient averaging, checkpoints, and the sync guard. |
| `public/transformer.js` | the mini transformer LM (attention + MLP blocks, next-token prediction) — forward and backward, every multiply through the units. |
| `public/verified_core.js` | the **verified INT8 units**: block-scaled quantize → exact LUT multiply → int32 accumulate → bit-exact f32 epilogue; plus the JS mirrors, the audit, and the MLP chain reference. |
| `public/webgpu.js` | the WGSL kernels (LUT matmul, DP4A int8 dot, fused attention, B2B MLP chain) and the **exact init gates** that admit them. |
| `public/traincore.js` | float reference trainer + deterministic Adam. |
| `public/*.bin` | the units as lookup tables (`mul_lut` is a 65536-entry exact int8 product table). |

## Peers, groups, and the leader

Peers are grouped by **public IP** (same network ⇒ same group) or by an
explicit `?room=CODE` (with host approval per join). Inside a group every
pair gets a WebRTC data channel — a full mesh; the signaling server is only
used for the handshake.

Whoever presses **Start** becomes the **leader** for that run: their settings
are broadcast to everyone, they sync mid-run joiners (weights + step), and
they are the source of truth for gradient repair (below). Everything else is
symmetric — every device computes, sends, and averages the same way.

## One training step

1. Each device samples its own batch from its data shard and computes the
   loss and gradient **through the verified units**.
2. It broadcasts `[step | weight-hash | probe-hash | loss | gradient]` to
   every peer and collects everyone else's.
3. When it has a gradient from **every device in the step roster**, it
   averages them **in strict roster order** — float addition is not
   associative, so a different order would round differently and fork the
   weights — and applies one Adam update. Adam's state is identical
   everywhere, so nothing but the gradients ever crosses the wire.
4. Because every device starts from the same seeded weights and applies the
   same averaged update with the same rounding, the replicas are
   **bit-identical** — verified continuously by the weight hash.

### The sync guard

Before applying an update, each device checks every peer's **weight hash**
(FNV-1a of the full weight bytes before the step) against its own. Any
mismatch means the replicas have forked — the guard **stops the run** rather
than train past a divergence. The **probe hash** is different: it re-runs a
fixed seeded int8 GEMM through the device's live kernel each step, so a
device whose *arithmetic* has gone wrong is caught even while its weights
still match (weights only depend on the gradient bytes everyone receives).

### Gradient repair (asymmetric meshes)

WebRTC meshes can be asymmetric: a gradient can reach the leader but not some
follower. Instead of stalling, a follower missing a roster gradient asks the
**leader** to re-send that peer's gradient for that step (the leader retains
the last 8 steps of everyone's gradients). The repair is bit-exact — the
follower averages the identical bytes — so the run continues without a fork.
Only the leader may vouch for another peer's gradient.

## The wire protocol

All messages are binary on the data channel. Gradient messages start with a
non-negative `int32 step`; control messages use negative sentinels:

| Sentinel | Message |
|---|---|
| `-2` | checkpoint push (one device restores the whole group) |
| `-3` | run config from the starter (`c, t, b, steps, lr`) |
| `-4` | step roster (who is in this step) |
| `-5` | fragment — large messages are chunked at 48 KB |
| `-6` | mid-run resume (weights + step for a late joiner) |
| `-7` | repair request (step, peer) |
| `-8` | repair response (leader-only, original header + bytes) |

Gradient wire format: `[i32 step][u32 whash][u32 phash][f32 loss][f32 grad…]`.

## Compute backends

At init, `webgpu.js` tries the best available backend and **gates** each
kernel before use (see [VERIFICATION.md](VERIFICATION.md)):

1. **DP4A** (`packed_4x8_integer_dot_product`) — hardware int8 dot product.
2. **LUT shader** — the multiply table as a WebGPU storage buffer; also the
   oracle twin every other kernel is compared against.
3. **CPU (JS)** — the same units in plain JavaScript. Not an approximation:
   the CPU mirror and the GPU kernels produce the **same bits**, which is
   what lets GPU and CPU devices train in one group.

There is no plain-float forward path — every multiply in the model goes
through the units on every backend.

## Training data

`server.js` exposes `/data`: it picks a random slice from the FineWeb-Edu
10BT parquet shards and reads it **directly off the HF CDN with HTTP range
requests** (pure-JS `hyparquet`, SNAPPY decompression, results cached). No
datasets-server dependency. Browsers fetch `/data` per batch; devices that
cannot reach it fall back to a small built-in corpus. The dataset is
hardcoded — every group trains on FineWeb-Edu.

## Checkpoints and the inference kit

The `.pt` checkpoint stores magic/version, config dims, step, and the flat
f32 weights; the loader validates tokenizer vocab and dimensions before
accepting, and a load **broadcasts** to the group (sentinel `-2`). The
**inference kit** is a generated single-file HTML with the model code and the
current weights (base64) baked in — offline generation anywhere.
