"""Template: copy this, fill in your model + data, then set

    DAISY_TASK=my_task_template:MyTask

(make sure the file is importable, e.g. run daisychain-train from this folder or
pip-install your package). Keep build_model deterministic so every node starts
from identical weights.
"""
import torch
import torch.nn as nn


class MyTask:
    def build_model(self) -> nn.Module:
        torch.manual_seed(0)                       # identical init on every node
        # TODO: return YOUR model
        return nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 10))

    def sample(self, n: int):
        # TODO: return n training samples from THIS node's data shard as (X, y).
        # For real data, shard by rank (e.g. different files/rows per RANK).
        X = torch.randn(n, 16)
        y = torch.randint(0, 10, (n,))
        return X, y

    def loss(self, model, X, y):
        # TODO: your loss (mean over the batch)
        return nn.functional.cross_entropy(model(X), y)
