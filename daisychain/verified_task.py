"""Example task that trains THROUGH the bundled verified units.

Same shape as ExampleTask, but the model is built from VerifiedLinear layers, so
every forward multiply/requant/ReLU runs on the trained N/N-verified INT8 units
(materialized as lookup tables for speed). Backward uses a straight-through
estimator, so ordinary weights still learn.

Run it with:  DAISY_TASK=daisychain.verified_task:VerifiedTask  daisychain-train
"""
import torch
import torch.nn as nn

from .verified import VerifiedLinear, load_units, instrument


class VerifiedTask:
    def __init__(self, fast: bool = True):
        self.mul, self.rq, self.relu = load_units()   # bundled trained weights
        instrument.enable()                            # count unit invocations
        self._fast = fast
        g = torch.Generator().manual_seed(1234)
        self.W = torch.randn(8, 1, generator=g)

    def build_model(self):
        torch.manual_seed(0)
        return nn.Sequential(
            VerifiedLinear(8, 8, self.mul, self.rq, self.relu, use_relu=True, fast=self._fast),
            VerifiedLinear(8, 1, self.mul, self.rq, self.relu, use_relu=False, fast=self._fast),
        )

    def sample(self, n):
        X = torch.randn(n, 8)
        return X, X @ self.W

    def loss(self, model, X, y):
        return nn.functional.mse_loss(model(X), y)
