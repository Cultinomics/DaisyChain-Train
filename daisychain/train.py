"""DaisyChain training entry point (CLI: `daisychain-train`).

Reads cluster settings from env (set on each machine, changing only RANK):

  MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK   -- standard torch.distributed
  GLOO_SOCKET_IFNAME                            -- the NIC to use (e.g. tailscale0)
  DAISY_TASK        = "module:Class"            -- your task (default: example)
  DAISY_STEPS       = 300
  DAISY_LR          = 0.05
  DAISY_OPTIMIZER   = sgd | adam
  DAISY_BASE_BATCH  = 32
  DAISY_STATUS_FILE = status.json               -- rank 0 writes live status here
  DAISY_STEP_SLEEP  = 0                          -- demo pacing
  DAISY_SAVE        = daisychain_model.pt        -- rank 0 saves here
"""
import os

import torch

from .cluster import DaisyCluster
from .task import load_task


def _report_verified_counts(cluster):
    """All-reduce verified-unit invocation counts across nodes (if any fired)."""
    try:
        from .verified import instrument
        import torch.distributed as dist
        counts = instrument.report()
        if not counts:
            return
        keys = sorted(counts)
        t = torch.tensor([counts[k] for k in keys], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if cluster.is_master():
            print("[verified] CLUSTER-WIDE verified-unit invocations (trained through them):")
            for k, v in zip(keys, t.tolist()):
                print(f"[verified]   {k:34s} {int(v):,}")
    except Exception:
        pass


def main():
    # Default: train THROUGH the emulated GPU logic (verified INT8 units).
    # Set DAISY_TASK=daisychain.example_task:ExampleTask for a plain-float run.
    task_spec = os.environ.get("DAISY_TASK", "daisychain.verified_task:VerifiedTask")
    task = load_task(task_spec)

    cluster = DaisyCluster(
        cpu_fraction=float(os.environ.get("DAISY_CPU_FRACTION", "0.9")),
        base_batch=int(os.environ.get("DAISY_BASE_BATCH", "32")),
    )

    if cluster.is_master():
        p = cluster.plan
        print(f"[daisychain] task={task_spec}")
        print(f"[plan] world={p['world']} devices={p['devices']} "
              f"total_cores={p['total_cores']} total_ram_gb={p['total_ram_gb']}")
        print(f"[plan] capacities={p['capacities']} weights={[round(w,3) for w in p['weights']]}")
        print(f"[plan] local_batches={p['local_batches']} global_batch={p['global_batch']}")

    model = task.build_model()
    cluster.fit(
        model, task,
        steps=int(os.environ.get("DAISY_STEPS", "300")),
        lr=float(os.environ.get("DAISY_LR", "0.05")),
        optimizer=os.environ.get("DAISY_OPTIMIZER", "sgd"),
        status_path=os.environ.get("DAISY_STATUS_FILE", "status.json"),
        step_delay=float(os.environ.get("DAISY_STEP_SLEEP", "0")),
    )

    # if the task trained THROUGH the verified units, report cluster-wide counts
    _report_verified_counts(cluster)

    diff = cluster.replica_diff(model)
    if cluster.is_master():
        print(f"[check] replica max param diff across nodes: {diff:.2e}")
        save = os.environ.get("DAISY_SAVE", "daisychain_model.pt")
        torch.save({"state_dict": model.state_dict()}, save)
        print(f"[save] {save}")
    cluster.shutdown()


if __name__ == "__main__":
    main()
