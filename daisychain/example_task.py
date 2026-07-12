"""The default example task: fit a small MLP to a synthetic function.

It exists so `daisychain-train` runs out of the box and you can confirm the
cluster works end to end. Replace it with your own task (see docs/CUSTOM_TASK.md)
-- copy this file, change build_model / sample / loss, and set DAISY_TASK.
"""
import torch
import torch.nn as nn


class ExampleTask:
    def __init__(self):
        # fixed target so every node's shard is consistent
        g = torch.Generator().manual_seed(1234)
        self.W = torch.randn(8, 1, generator=g)

    def build_model(self):
        torch.manual_seed(0)                 # identical init on every node
        return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))

    def sample(self, n):
        X = torch.randn(n, 8)
        return X, X @ self.W + 0.05 * torch.randn(n, 1)

    def loss(self, model, X, y):
        return nn.functional.mse_loss(model(X), y)
