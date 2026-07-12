"""The Task interface — bring your own model + data.

A DaisyChain task is any object with three methods:

    build_model() -> torch.nn.Module          # the model to train (identical on every node)
    sample(n)     -> (X, y)                    # draw n training samples (this node's shard)
    loss(model, X, y) -> scalar tensor         # mean loss over the batch

Point DaisyChain at your task with  DAISY_TASK="my_module:MyTask"  (or --task).
An example lives in examples/example_task.py. Keep build_model deterministic
(seed it) so every node starts from the same weights.
"""
from __future__ import annotations

import importlib
from typing import Protocol, Tuple

import torch


class Task(Protocol):
    def build_model(self) -> torch.nn.Module: ...
    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def loss(self, model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


def load_task(spec: str):
    """spec = 'package.module:ClassName' -> instantiated task object."""
    if ":" not in spec:
        raise ValueError(f"task spec must be 'module:Class', got {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)()
