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
    return n


def pick_device() -> "torch.device":
    if os.environ.get("DAISY_FORCE_CPU") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gpu_info():
    if not torch.cuda.is_available():
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


def capacity_score(device=None, secs: float = 0.3) -> float:
    """Measured throughput: fixed matmuls/sec on the local device. Self-calibrating
    (a GPU scores far higher), so capacity weighting hands it a bigger batch."""
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
        return float(os.cpu_count() or 1)


def survey_node(cpu_fraction: float = 0.9, measure: bool = True) -> dict:
    cores = int(os.environ.get("DAISY_CORES", os.cpu_count() or 1))
    dev = pick_device()
    gpu = _gpu_info() if dev.type == "cuda" else None
    if "DAISY_CAPACITY" in os.environ:
        cap = float(os.environ["DAISY_CAPACITY"])
    elif measure:
        cap = capacity_score(dev)
    else:
        cap = float(cores)
    return {"host": socket.gethostname(), "cores": cores,
            "threads": max(1, int(round(cores * cpu_fraction))),
            "ram_gb": _available_ram_gb(), "device": dev.type,
            "gpu": gpu, "capacity": cap}


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

    caps = [float(g.get("capacity") or g["cores"]) for g in gathered]
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
            "capacities": [round(c, 1) for c in caps],
            "gpus": [g.get("gpu") for g in gathered]}


@torch.no_grad()
def broadcast_params(model, src: int = 0):
    for p in model.parameters():
        cpu = p.data.detach().to("cpu")
        dist.broadcast(cpu, src=src)
        p.data.copy_(cpu.to(p.data.device))


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
            status_path=None, step_delay=0.0):
        """Train `model` on `task` (build_model already called). task.sample(n)
        draws this node's shard; task.loss(model, X, y) returns a scalar."""
        model.to(self.device)
        broadcast_params(model)
        if optimizer == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=lr)
        w, nb = self.plan["my_weight"], self.plan["my_local_batch"]
        for s in range(steps):
            X, y = task.sample(nb)
            X, y = X.to(self.device), y.to(self.device)
            opt.zero_grad(set_to_none=False)
            loss = task.loss(model, X, y)
            loss.backward()
            capacity_weighted_allreduce_grads(model, w)
            opt.step()
            if step_delay:
                time.sleep(step_delay)
            if s % max(1, steps // 20) == 0 or s == steps - 1:
                lt = loss.detach().to("cpu").clone()
                dist.all_reduce(lt, op=dist.ReduceOp.SUM)
                if self.is_master():
                    cl = lt.item() / self.world
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
