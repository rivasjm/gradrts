"""Evaluation core.

    from frozen import harness, pool

    report = harness.evaluate()          # every technique registered in algorithms/
    report["techniques"]["pd"]["schedulable"]

Exposes the holistic local-EDF analysis (the judge), the model, the deterministic
example pool and the evaluation harness.
"""

from .model import Flow, LinearSystem, Processor, SchedulerType, Task, TaskType
from .analysis import HolisticLocalEDFAnalysis
from . import pool, harness
from .pool import (POPULATION, SEED, SIZES, UTILIZATIONS, get_systems,
                   set_utilization)
from .harness import evaluate, is_schedulable

__all__ = [
    "Flow", "LinearSystem", "Processor", "SchedulerType", "Task", "TaskType",
    "HolisticLocalEDFAnalysis",
    "pool", "harness",
    "POPULATION", "SEED", "SIZES", "UTILIZATIONS", "get_systems",
    "set_utilization", "evaluate", "is_schedulable",
]
