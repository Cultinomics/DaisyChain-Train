"""DaisyChain cluster core — data-parallel CPU/GPU training across spare machines.

Design: distribute the *parallel* axis (the batch) across nodes; keep each node's
work local. Every node holds a full model replica and trains on its shard; a
capacity-weighted gradient all-reduce combines them into the exact full-batch
gradient, so replicas stay bit-identical.

  - Each node uses ~90% of its cores (and its GPU if it has one).
  - Capacity is MEASURED (matmuls/sec) so a strong node auto-takes a bigger
    batch. Gradients are reduced on CPU copies, so CPU and GPU nodes mix.

Pools compute, not memory: the model must fit on one node. Honest limits are in
docs/LIMITS.md.
"""
from __future__ import annotations

import json
import os
import socket
import time

import torch
import torch.distributed as dist


# ---------------------------------------------------------------- resources ---
def configure_cpu(fraction: float = 0.9) -> int:
    cores = os.cpu_count() or 1
    n = max(1, int(round(cores * fraction)))
    torch.set_num_threads(n)
    pin_fp_mode()
    return n


def pin_fp_mode() -> None:
    """Pin the host floating-point mode: subnormals ON (no flush-to-zero / denormals-are-zero).

    FTZ/DAZ live in the thread's MXCSR, and any library built with -ffast-math can switch them
    on just by being loaded (the emulator found exactly this leaking into Python, 2026-09).
    MEASURED 2026-10-14: the same seed and gradients through 5 Adam steps give DIFFERENT weights
    with torch.set_flush_denormal(True) -- so one node with a leaked FTZ forks its replica."""
    torch.set_flush_denormal(False)


def fp_mode_ok() -> bool:
    """True if this thread keeps subnormals: FTZ off (a normal*normal product may be subnormal)
    and DAZ off (a subnormal input is read as itself)."""
    ftz = torch.tensor([1e-20], dtype=torch.float32) * torch.tensor([1e-20], dtype=torch.float32)
    daz = torch.tensor([1e-39], dtype=torch.float32) * 1.0
    return float(ftz) != 0.0 and float(daz) != 0.0


def repin_fp_mode(step: int, rank: int) -> bool:
    """Re-check the FP mode before an optimizer step; re-pin and say so if it was flipped.

    pin_fp_mode runs once at start, but a library imported mid-run (a dataloader worker's
    dependency, a fast-math extension) can switch FTZ/DAZ on afterwards. The emulator restores
    MXCSR around EVERY guest call for this reason; here the replica hash would catch the fork
    later with nothing naming the cause. Returns True if the mode had been flipped."""
    if fp_mode_ok():
        return False
    pin_fp_mode()
    print("  [fp] rank %d step %d: flush-to-zero/denormals-are-zero was switched ON since the "
          "last step (a library changed MXCSR); re-pinned subnormals ON" % (rank, step), flush=True)
    return True


def _cuda_usable() -> bool:
    """is_available() alone is not enough: with CUDA_VISIBLE_DEVICES set but EMPTY (a common way
    to say "CPU only") torch reports available=True and device_count()=0, and the node crashed
    at startup in get_device_properties(0). Require a device to actually exist (2026-09-30)."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def pick_device() -> "torch.device":
    if os.environ.get("DAISY_FORCE_CPU") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if _cuda_usable() else "cpu")


def _gpu_info():
    if not _cuda_usable():
        return None
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "vram_gb": round(p.total_memory / 1e9, 1),
            "capability": f"{p.major}.{p.minor}"}


def _available_ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().available / 1e9, 1)
    except Exception:
        return None


def capacity_score(device=None, secs: float = 0.3):
    """Measured throughput: fixed matmuls/sec on the local device. Self-calibrating
    (a GPU scores far higher), so capacity weighting hands it a bigger batch.

    Returns None if the measurement fails. It used to return the core count instead -- a number
    in a DIFFERENT UNIT that cluster_plan then weighed against other nodes' matmuls/sec, so a node
    whose benchmark threw got ~8 "capacity" next to a peer's ~200 and a sliver of the batch, with
    nothing in the plan saying why (2026-10-14: a fallback that looks like a result)."""
    dev = device or pick_device()
    try:
        a = torch.randn(512, 512, device=dev)
        b = torch.randn(512, 512, device=dev)
        _ = a @ b
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0, it = time.time(), 0
        while time.time() - t0 < secs:
            a = a @ b
            it += 1
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return it / (time.time() - t0)
    except Exception:
        return None


def survey_node(cpu_fraction: float = 0.9, measure: bool = True) -> dict:
    cores = int(os.environ.get("DAISY_CORES", os.cpu_count() or 1))
    dev = pick_device()
    gpu = _gpu_info() if dev.type == "cuda" else None
    if "DAISY_CAPACITY" in os.environ:
        cap, unit = float(os.environ["DAISY_CAPACITY"]), "user"
    elif measure:
        cap, unit = capacity_score(dev), "matmul/s"
        if cap is None:
            cap, unit = float(cores), "cores"
    else:
        cap, unit = float(cores), "cores"
    return {"host": socket.gethostname(), "cores": cores,
            "threads": max(1, int(round(cores * cpu_fraction))),
            "ram_gb": _available_ram_gb(), "device": dev.type,
            "gpu": gpu, "capacity": cap, "capacity_unit": unit}


def plan_capacities(nodes):
    """Per-node capacities in ONE unit, plus the unit used.

    Measured matmuls/sec and core counts are not comparable. If the measured nodes do not all
    share a unit, every node falls back to its core count (a coarser but consistent weighting)
    instead of mixing the two. A DAISY_CAPACITY override ("user") is taken as stated: the
    operator chose its scale. Nodes from an older survey without a unit count as "cores" if
    their capacity equals their core count, else as measured."""
    def unit_of(g):
        u = g.get("capacity_unit")
        if u:
            return u
        return "cores" if float(g.get("capacity") or 0) == float(g["cores"]) else "matmul/s"
    units = {unit_of(g) for g in nodes} - {"user"}
    if len(units) > 1:
        return [float(g["cores"]) for g in nodes], "cores (mixed units: %s)" % ", ".join(sorted(units))
    return [float(g.get("capacity") or g["cores"]) for g in nodes], (units.pop() if units else "user")


# ------------------------------------------------------------------ cluster ---
def init_cluster(backend: str = "gloo"):
    os.environ.setdefault("USE_LIBUV", "0")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size()


def cluster_plan(cpu_fraction: float = 0.9, base_batch: int = 32) -> dict:
    rank, world = dist.get_rank(), dist.get_world_size()
    me = survey_node(cpu_fraction)
    gathered = [None] * world
    dist.all_gather_object(gathered, me)

    caps, cap_unit = plan_capacities(gathered)
    if cap_unit.startswith("cores (mixed") and rank == 0:
        print("  [plan] capacity measured in different units across nodes -- weighting every node "
              "by core count instead (%s)" % cap_unit, flush=True)
    total_cap = sum(caps) or world
    global_batch = base_batch * world
    batches = [max(1, round(global_batch * c / total_cap)) for c in caps]
    total_batch = sum(batches)
    weights = [b / total_batch for b in batches]
    rams = [g["ram_gb"] for g in gathered if g["ram_gb"] is not None]
    return {"rank": rank, "world": world, "nodes": gathered,
            "weights": weights, "local_batches": batches,
            "my_weight": weights[rank], "my_local_batch": batches[rank],
            "total_cores": sum(g["cores"] for g in gathered),
            "total_ram_gb": (sum(rams) if rams else None),
            "global_batch": sum(batches),
            "devices": [g.get("device", "cpu") for g in gathered],
            "capacities": [round(c, 1) for c in caps], "capacity_unit": cap_unit,
            "gpus": [g.get("gpu") for g in gathered]}


@torch.no_grad()
def param_hash(params) -> int:
    """64-bit BLAKE2b digest of the exact parameter BITS (CPU copies), for replica checks."""
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    for p in params:
        h.update(p.detach().to("cpu").contiguous().view(-1).numpy().tobytes())
    return int.from_bytes(h.digest(), "little", signed=True)


@torch.no_grad()
def check_replicas(params, step: int) -> None:
    """Refuse to continue if any replica's parameter bits differ from rank 0's. A collective.

    replica_diff used to run once, AFTER training: a fork at step 1 surfaced only as a final
    number. The web client hashes weights every step; this is the same check for the cluster."""
    mine = torch.tensor([param_hash(params)], dtype=torch.int64)
    allh = [torch.zeros_like(mine) for _ in range(dist.get_world_size())]
    dist.all_gather(allh, mine)
    bad = [r for r, h in enumerate(allh) if int(h) != int(allh[0])]
    if bad:
        raise RuntimeError("step %d: replica(s) %s diverged from rank 0 (parameter bits differ); "
                           "refusing to continue on forked weights" % (step, bad))


def mixed_devices(plan) -> bool:
    return len(set(plan.get("devices") or [])) > 1


@torch.no_grad()
def broadcast_params(model, src: int = 0):
    for p in model.parameters():
        cpu = p.data.detach().to("cpu")
        dist.broadcast(cpu, src=src)
        p.data.copy_(cpu.to(p.data.device))


@torch.no_grad()
def check_finite_grads(model, step: int) -> None:
    """Refuse a non-finite REDUCED gradient before the optimizer applies it.

    After the all-reduce every rank holds the identical summed gradient, so every rank reaches
    the same verdict with no extra communication and they all stop at the same step. Before
    2026-09-30 one node's NaN was summed into every replica and opt.step() wrote it into the
    weights (and Adam's moments) everywhere -- the replicas still agreed, so replica_diff
    could not see it."""
    for name, p in model.named_parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            bad = int((~torch.isfinite(p.grad)).sum())
            raise FloatingPointError(
                "step %d: reduced gradient of %s has %d non-finite value(s); refusing to apply "
                "it (weights and optimizer state unchanged)" % (step, name, bad))


@torch.no_grad()
def cluster_mean_loss(loss, weight: float) -> float:
    """The full-batch mean loss: sum_i w_i * loss_i with the SAME capacity weights
    (w_i = n_i / sum n) the gradient all-reduce uses. A collective -- call it on every rank.

    It used to be sum(loss_i) / world: an equal-weight mean of UNEQUAL batches, so the logged
    cluster loss leaned toward the small (slow) nodes whenever capacities differed (2026-09-30)."""
    lt = loss.detach().to("cpu").clone().double() * weight
    dist.all_reduce(lt, op=dist.ReduceOp.SUM)
    return lt.item()


@torch.no_grad()
def capacity_weighted_allreduce_grads(model, weight: float):
    """Σ_i w_i g_i with w_i = n_i/Σn_j == the true full-batch mean gradient.
    Reduced on CPU copies so mixed CPU/GPU nodes interoperate over gloo."""
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        g = p.grad.detach().to("cpu").mul_(weight)
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        p.grad.copy_(g.to(p.grad.device))


class DaisyCluster:
    """One node's handle on the cluster. Same code runs on every machine."""

    def __init__(self, cpu_fraction: float = 0.9, base_batch: int = 32):
        self.threads = configure_cpu(cpu_fraction)
        self.device = pick_device()
        self.rank, self.world = init_cluster()
        self.plan = cluster_plan(cpu_fraction, base_batch)

    def is_master(self):
        return self.rank == 0

    def _write_status(self, path, **kw):
        payload = {"rank": self.rank, "world": self.world,
                   "plan": {"total_cores": self.plan["total_cores"],
                            "total_ram_gb": self.plan["total_ram_gb"],
                            "weights": self.plan["weights"],
                            "devices": self.plan["devices"],
                            "local_batches": self.plan["local_batches"]}, **kw}
        try:
            with open(path, "w") as f:
                json.dump(payload, f)
        except Exception:
            pass

    def fit(self, model, task, steps=500, lr=1e-2, optimizer="sgd",
            status_path=None, step_delay=0.0, check_every=None):
        """Train `model` on `task` (build_model already called). task.sample(n)
        draws this node's shard; task.loss(model, X, y) returns a scalar.

        Mixed CPU/GPU fleets: the optimizer runs on CPU MASTER copies on every rank. MEASURED
        2026-10-14 (RTX, torch 2.x): CUDA Adam vs CPU Adam on identical gradients differ in ~2% of
        weights after 20 steps (default, foreach=False and fused=True alike) -- the same update
        rule, two implementations, two roundings. Replicas forked at step 1 and nothing checked
        until replica_diff after training. One implementation everywhere, like an emulator that
        must give one semantics on two backends. Homogeneous fleets keep the on-device optimizer.
        Every `check_every` steps (default: the logging cadence) all ranks compare a hash of
        their parameter bits and stop on any difference (DAISY_CPU_OPT=0/1 overrides the choice)."""
        model.to(self.device)
        pin_fp_mode()                            # fit may run on a thread configure_cpu never saw
        broadcast_params(model)
        params = list(model.parameters())
        force = os.environ.get("DAISY_CPU_OPT")
        cpu_opt = (force == "1") if force in ("0", "1") else \
            (mixed_devices(self.plan) and self.device.type != "cpu")
        if cpu_opt:
            master = [torch.nn.Parameter(p.detach().to("cpu").clone()) for p in params]
            opt_params = master
        else:
            master, opt_params = None, params
        if optimizer == "adam":
            opt = torch.optim.Adam(opt_params, lr=lr)
        else:
            opt = torch.optim.SGD(opt_params, lr=lr)
        if self.is_master() and mixed_devices(self.plan):
            print("  [fit] mixed devices %s: optimizer on CPU master copies on GPU ranks"
                  % sorted(set(self.plan["devices"])), flush=True)
        every = check_every or max(1, steps // 20)
        w, nb = self.plan["my_weight"], self.plan["my_local_batch"]
        for s in range(steps):
            X, y = task.sample(nb)
            X, y = X.to(self.device), y.to(self.device)
            opt.zero_grad(set_to_none=False)
            loss = task.loss(model, X, y)
            loss.backward()
            capacity_weighted_allreduce_grads(model, w)
            check_finite_grads(model, s)
            repin_fp_mode(s, self.rank)                # before the step: the update is where FTZ forks
            if master is not None:
                with torch.no_grad():
                    for mp, p in zip(master, params):
                        mp.grad = p.grad.detach().to("cpu")
                    opt.step()
                    for mp, p in zip(master, params):
                        p.data.copy_(mp.data.to(p.data.device))
            else:
                opt.step()
            if s % every == 0 or s == steps - 1:
                check_replicas(params, s)              # collective: every rank calls it
            if step_delay:
                time.sleep(step_delay)
            if s % max(1, steps // 20) == 0 or s == steps - 1:
                cl = cluster_mean_loss(loss, w)             # collective: every rank calls it
                if self.is_master():
                    print(f"  step {s:5d}  cluster-avg loss {cl:.6f}", flush=True)
                    if status_path:
                        self._write_status(status_path, step=s, total_steps=steps,
                                            cluster_avg_loss=cl, done=(s == steps - 1))
        return model

    def replica_diff(self, model):
        vec = torch.cat([p.data.reshape(-1).to("cpu") for p in model.parameters()])
        bucket = [torch.zeros_like(vec) for _ in range(self.world)]
        dist.all_gather(bucket, vec)
        return max((bucket[i] - bucket[0]).abs().max().item() for i in range(self.world))

    def shutdown(self):
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
