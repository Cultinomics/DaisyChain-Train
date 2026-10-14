"""Cluster reductions (2026-09-30). Run: python test_cluster_reductions.py  (CPU only, ~10 s).

1. cluster_mean_loss is the FULL-BATCH mean loss: sum_i w_i * loss_i with the capacity weights
   the gradient all-reduce uses. It was sum(loss_i) / world -- checked on a real 2-rank gloo
   group with unequal batches (3 and 13 samples), against the loss of the concatenated batch.
2. check_finite_grads refuses a non-finite reduced gradient BEFORE opt.step(), and leaves the
   weights untouched; a finite gradient passes.
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from daisychain.cluster import (capacity_weighted_allreduce_grads, check_finite_grads,
                                cluster_mean_loss)

FAILURES = []


def ck(name, cond, detail=""):
    print("  %s  %s%s" % ("ok  " if cond else "FAIL", name, ("  " + str(detail)) if detail else ""))
    if not cond:
        FAILURES.append(name)


SIZES = (3, 13)


def data(rank):
    g = torch.Generator().manual_seed(100 + rank)
    X = torch.randn(SIZES[rank], 4, generator=g)
    y = torch.randn(SIZES[rank], 1, generator=g)
    return X, y


def model0():
    torch.manual_seed(7)
    return torch.nn.Linear(4, 1)


def worker(rank, world, init_file, out_file):
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=world)
    w = SIZES[rank] / sum(SIZES)
    m = model0()
    X, y = data(rank)
    loss = torch.nn.functional.mse_loss(m(X), y)
    loss.backward()
    cl = cluster_mean_loss(loss, w)
    old = loss.detach().clone()
    dist.all_reduce(old, op=dist.ReduceOp.SUM)                 # the previous formula
    capacity_weighted_allreduce_grads(m, w)
    if rank == 0:
        torch.save({"cl": cl, "old": old.item() / world,
                    "grad": [p.grad.clone() for p in m.parameters()]}, out_file)
    dist.barrier()
    dist.destroy_process_group()


def main():
    print("cluster_mean_loss -- 2 ranks, batches of %d and %d samples" % SIZES)
    with tempfile.TemporaryDirectory() as td:
        init_file, out_file = os.path.join(td, "init"), os.path.join(td, "out.pt")
        mp.spawn(worker, args=(2, init_file, out_file), nprocs=2, join=True)
        got = torch.load(out_file)
    m = model0()
    Xa = torch.cat([data(0)[0], data(1)[0]])
    ya = torch.cat([data(0)[1], data(1)[1]])
    full = torch.nn.functional.mse_loss(m(Xa), ya)
    full.backward()
    ck("equals the loss of the concatenated batch (the full-batch mean)",
       abs(got["cl"] - full.item()) < 1e-6, (got["cl"], full.item()))
    ck("the old sum/world formula did NOT (it is the equal-weight mean of unequal batches)",
       abs(got["old"] - full.item()) > 1e-4, (got["old"], full.item()))
    ck("and the gradient all-reduce agrees with the same weighting (consistency check)",
       all(torch.allclose(g, p.grad, atol=1e-6) for g, p in zip(got["grad"], m.parameters())))

    print("check_finite_grads -- refuse a non-finite reduced gradient before opt.step()")
    lin = model0()
    before = [p.detach().clone() for p in lin.parameters()]
    lin(torch.ones(2, 4)).sum().backward()
    try:
        check_finite_grads(lin, 0)
        ck("a finite gradient passes", True)
    except FloatingPointError as e:
        ck("a finite gradient passes", False, e)
    for bad in (float("nan"), float("inf")):
        lin.weight.grad[0, 1] = bad
        try:
            check_finite_grads(lin, 5)
            ck("%r in a reduced gradient raises" % bad, False, "applied silently")
        except FloatingPointError as e:
            ck("%r in a reduced gradient raises, naming the step and parameter" % bad,
               "step 5" in str(e) and "weight" in str(e), str(e)[:80])
    ck("the weights were never touched",
       all(torch.equal(a, b) for a, b in zip(before, lin.parameters())))

    print("plan_capacities -- never weigh matmuls/sec against core counts (2026-10-14)")
    from daisychain.cluster import plan_capacities
    meas = {"cores": 8, "capacity": 210.0, "capacity_unit": "matmul/s"}
    fell = {"cores": 8, "capacity": 8.0, "capacity_unit": "cores"}      # its benchmark threw
    caps, unit = plan_capacities([meas, dict(meas, capacity=190.0)])
    ck("all measured: the measured capacities are used", caps == [210.0, 190.0] and unit == "matmul/s")
    caps, unit = plan_capacities([meas, fell])
    ck("one fallback: EVERY node weighs by cores, not 210 vs 8", caps == [8.0, 8.0] and "mixed" in unit, (caps, unit))
    caps, unit = plan_capacities([meas, {"cores": 4, "capacity": 999.0, "capacity_unit": "user"}])
    ck("a DAISY_CAPACITY override is taken as stated", caps == [210.0, 999.0], caps)
    caps, unit = plan_capacities([{"cores": 8, "capacity": 8.0}, {"cores": 8, "capacity": 150.0}])
    ck("old-format surveys (no unit) are classified, and mixed -> cores", caps == [8.0, 8.0], caps)

    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
