# Training your own model

DaisyChain trains any **Task** — an object with three methods:

```python
import torch, torch.nn as nn

class MyTask:
    def build_model(self) -> nn.Module:
        torch.manual_seed(0)              # deterministic -> identical on every node
        return nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 10))

    def sample(self, n):                  # this node's data shard
        X = torch.randn(n, 16)
        y = torch.randint(0, 10, (n,))
        return X, y

    def loss(self, model, X, y):          # mean loss over the batch
        return nn.functional.cross_entropy(model(X), y)
```

## Point DaisyChain at it

```bash
export DAISY_TASK="my_task:MyTask"        # module:Class, must be importable
daisychain-train
```

Copy `examples/my_task_template.py` to start.

## Rules that matter

1. **`build_model` must be deterministic** (seed it). Every node builds the model
   independently, then rank 0's weights are broadcast — but seeding keeps shapes
   and buffers consistent.
2. **`sample(n)` should return this node's shard.** For real datasets, split by
   `RANK` (e.g. different files/row-ranges per rank) so nodes don't all train on
   the same rows. Read `os.environ["RANK"]` / `WORLD_SIZE`.
3. **The model must fit on one node.** DaisyChain pools compute, not memory.
4. Keep it **small.** See [LIMITS.md](LIMITS.md).

## Knobs (env)

| var | default | meaning |
|-----|---------|---------|
| `DAISY_TASK` | example | `module:Class` |
| `DAISY_STEPS` | 300 | training steps |
| `DAISY_LR` | 0.05 | learning rate |
| `DAISY_OPTIMIZER` | sgd | `sgd` or `adam` |
| `DAISY_BASE_BATCH` | 32 | per-node base batch (scaled by capacity) |
| `DAISY_SAVE` | daisychain_model.pt | where rank 0 saves |
| `DAISY_FORCE_CPU` | – | set `1` to ignore a local GPU |
