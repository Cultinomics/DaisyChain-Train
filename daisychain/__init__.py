"""DaisyChain — Old Hardware Training Pipeline.

Chain spare/old machines into a data-parallel training cluster in minutes.
Bring your own model+data as a Task; DaisyChain handles device selection,
capacity-weighted sharding, gradient sync, and a live dashboard.
"""
from .cluster import (DaisyCluster, survey_node, cluster_plan, pick_device,
                      capacity_score, configure_cpu)
from .task import Task, load_task

__version__ = "0.1.0"
__all__ = ["DaisyCluster", "survey_node", "cluster_plan", "pick_device",
           "capacity_score", "configure_cpu", "Task", "load_task"]
