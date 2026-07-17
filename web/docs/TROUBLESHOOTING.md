# Troubleshooting

Common failure modes, what the log messages mean, and what to do.

## Connection problems

**Devices don't see each other.**
- Same network? Auto-grouping is by **public IP** — devices on different
  networks (e.g. phone on cellular) will never auto-group. Use
  `?room=CODE` on every device instead.
- Room joins need **host approval** — the first device in the room must click
  Accept for each newcomer.
- Corporate/CGNAT networks can block WebRTC (only STUN is configured, no TURN
  relay). Try a room over the hosted Space, or another network.

**"WebGPU not available" / backend shows CPU.**
- WebGPU needs a **secure context**: `localhost` or HTTPS. Plain
  `http://192.168.x.x` never gets WebGPU — tunnel it or use the Space.
- Old hardware/drivers without a modern GPU backend fall back to CPU. That's
  supported — CPU devices compute the identical bits, just slower.

**A kernel warning at startup** (e.g. "failed verification — using CPU LUT
mirrors"). The device's GPU produced values that don't match the verified
units bit-for-bit, so that kernel was refused and a slower-but-exact path is
used. This is the system working as designed; the run stays correct.

## Training problems

**"SYNC GUARD: stopped — weight hash mismatch."** Replicas forked — the run
is stopped rather than trained past a divergence. The most common cause is
**mixed builds**: devices running different versions of the page whose math
specs differ (the quantize respec is a real, bounded spec change — old and
new builds must not co-train). Make sure every device has the same version
(hard-refresh: Ctrl+Shift+R), then restart the run or restore from a
checkpoint.

**"missing a roster gradient … requesting repair from the leader."** A
gradient reached some peers but not others (asymmetric mesh). The follower
re-requests it from the leader, who retains the last 8 steps — normally the
run continues by itself. If repair fails repeatedly, the sync guard stops the
run; check whether a device went offline.

**"probe hash mismatch."** A device's *kernel* is producing wrong values at
runtime (the seeded probe GEMM no longer matches the fleet). That device
should rejoin — its gates will re-run at init and demote it to CPU if its GPU
misbehaves.

**Loss looks stuck / falls very slowly.** Small widths and short runs are
modest models — try more steps, a wider model, or more devices (each adds
its batch to the effective batch). Remember the envelope: this pools
compute, not memory; it is a proof of concept for small models.

**A slow device paces everyone.** Steps are synchronous: the group advances
when every roster gradient arrives. Removing (or upgrading) the slowest
device speeds up everyone.

## Data problems

**"dataset unreachable — using built-in corpus."** The `/data` endpoint
couldn't read the FineWeb-Edu shards (offline, or the CDN hiccuped — the
server reads parquet slices straight off the HF CDN with range requests).
Training continues on the built-in fallback text; generation quality will
reflect the smaller corpus.

## Checkpoint problems

**"tokenizer mismatch."** The checkpoint was trained with a different vocab
than the tokenizer this page loaded — it cannot be resumed here.

**"bad dims in checkpoint."** Width/sequence outside the supported 16–128
range — likely a corrupted or foreign file.

Loading a valid checkpoint on **one** device broadcasts it to the whole
group; you don't need to load it everywhere.

## Verifying your install

```bash
npm test
```

All ten suites should pass on a clean clone (Node only, no browser/GPU
required). If a suite fails on unmodified code, please report it — the tests
are deterministic apart from bounded random sweeps that are designed not to
flake. See [TEST_RESULTS.md](../TEST_RESULTS.md) for what each suite proves.
