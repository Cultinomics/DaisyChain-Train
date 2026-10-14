"""Mixed CPU + GPU fleet: replicas must stay bit-identical (2026-10-14).

Real 2-rank gloo run: rank 0 trains on CUDA, rank 1 on CPU, with Adam.
  * MEASURED first: CUDA Adam and CPU Adam on identical gradients differ in ~2% of weights after
    20 steps, so with the optimizer on each rank's own device the replicas fork at step 1.
    With DAISY_CPU_OPT=0 (the old behaviour) the new per-interval replica check must STOP the run.
  * Default: GPU ranks step the optimizer on CPU master copies, so every rank runs the same
    implementation; the run completes and the final parameter bits agree exactly.
Also: pin_fp_mode turns flush-to-zero off even if something switched it on.
Needs a CUDA device; skips (exit 0, says so) without one.
"""
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daisychain import cluster as C          # noqa: E402

FAILURES = []


def ck(name, cond, detail=""):
    print(("  ok    " if cond else "  FAIL  ") + name + ("" if cond else "  [%s]" % (detail,)))
    if not cond:
        FAILURES.append(name)


class Task:
    def __init__(self, rank):
        self.g = torch.Generator().manual_seed(1000 + rank)

    def sample(self, n):
        X = torch.randn(n, 32, generator=self.g)
        return X, X[:, :1] * 0.5 - X[:, 1:2]

    def loss(self, model, X, y):
        return torch.nn.functional.mse_loss(model(X), y)


def worker(rank, world, init_file, out_dir, cpu_opt):
    if cpu_opt is not None:
        os.environ["DAISY_CPU_OPT"] = cpu_opt
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=world)
    cl = C.DaisyCluster.__new__(C.DaisyCluster)
    cl.threads, cl.rank, cl.world = 2, rank, world
    cl.device = torch.device("cuda" if rank == 0 else "cpu")
    cl.plan = {"devices": ["cuda", "cpu"], "my_weight": 0.5, "my_local_batch": 64,
               "weights": [0.5, 0.5], "local_batches": [64, 64], "total_cores": 2, "total_ram_gb": None}
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    err = None
    try:
        cl.fit(model, Task(rank), steps=40, lr=1e-3, optimizer="adam", check_every=5)
    except RuntimeError as e:
        err = str(e)
    torch.save({"err": err, "hash": C.param_hash(list(model.parameters()))},
               os.path.join(out_dir, "r%d.pt" % rank))
    dist.destroy_process_group()


def run(cpu_opt):
    with tempfile.TemporaryDirectory() as td:
        mp.spawn(worker, args=(2, os.path.join(td, "init"), td, cpu_opt), nprocs=2, join=True)
        return [torch.load(os.path.join(td, "r%d.pt" % r)) for r in range(2)]


def main():
    if not C._cuda_usable():
        print("SKIPPED: no CUDA device (this test needs one GPU rank and one CPU rank)")
        return 0
    print("old behaviour (DAISY_CPU_OPT=0): optimizer on each rank's own device")
    old = run("0")
    ck("the replica check stops the run on both ranks", all(r["err"] and "diverged" in r["err"] for r in old),
       [r["err"] for r in old])
    print("default: GPU ranks step on CPU master copies")
    new = run(None)
    ck("the run completes on both ranks", all(r["err"] is None for r in new), [r["err"] for r in new])
    ck("final parameter bits are identical across the CUDA and CPU ranks", new[0]["hash"] == new[1]["hash"])

    print("pin_fp_mode")
    torch.set_flush_denormal(True)
    C.pin_fp_mode()
    ck("subnormals survive after pin_fp_mode (FTZ off)",
       float(torch.tensor([1e-39], dtype=torch.float32) * 1.0) != 0.0)
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
