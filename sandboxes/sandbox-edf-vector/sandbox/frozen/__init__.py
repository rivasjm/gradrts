"""Frozen evaluation core for the vectorized-EDF-analysis sandbox.

Import from here (read-only):

    from frozen import HolisticLocalEDFAnalysis, get_systems, build_instances
    from frozen import UTILIZATIONS, SEED, SIZES, N_TASKS

Nothing in this package may be modified: it is the fixed referee used to check
the equivalence and the speedup of the vectorized analysis you write in
``algorithms/``.
"""

from .model import (Flow, LinearSystem, Processor, SchedulerType, Task,
                    TaskType)
from .analysis import HolisticLocalEDFAnalysis, LimitFactorReachedException
from . import pool
from .pool import (POPULATION, POOLS, SEED, SIZES, UTILIZATIONS,
                   N_TASKS, get_systems, set_utilization)
from . import harness
from .harness import (build_instances, reference_wcrts, compare, evaluate)

__all__ = [
    "Flow", "LinearSystem", "Processor", "SchedulerType", "Task", "TaskType",
    "HolisticLocalEDFAnalysis", "LimitFactorReachedException",
    "pool", "harness",
    "get_systems", "set_utilization", "UTILIZATIONS", "SEED", "POOLS",
    "POPULATION", "SIZES", "N_TASKS",
    "build_instances", "reference_wcrts", "compare", "evaluate",
]
