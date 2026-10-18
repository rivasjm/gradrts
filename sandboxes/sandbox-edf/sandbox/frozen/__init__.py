"""Frozen evaluation core for the local-EDF deadline-assignment sandbox.

Import from here (read-only):

    from frozen import HolisticLocalEDFAnalysis, get_systems, set_utilization
    from frozen import UTILIZATIONS, SEED, POOLS, REFERENCE_METHOD

Nothing in this package may be modified: it is the fixed referee used to score
every deadline-assignment algorithm you write in ``algorithms/``.
"""

from .model import (Flow, LinearSystem, Processor, SchedulerType, Task,
                    TaskType)
from .analysis import HolisticLocalEDFAnalysis, LimitFactorReachedException
from . import pool
from . import baselines
from . import harness
from . import results
from .pool import (POPULATION, POOLS, REFERENCE_METHOD, SEED, SIZES,
                   UTILIZATIONS, get_systems, set_utilization)
from .harness import evaluate, is_schedulable

__all__ = [
    "Flow", "LinearSystem", "Processor", "SchedulerType", "Task", "TaskType",
    "HolisticLocalEDFAnalysis", "LimitFactorReachedException",
    "pool", "baselines", "harness", "results",
    "get_systems", "set_utilization", "UTILIZATIONS", "SEED", "POOLS",
    "REFERENCE_METHOD", "POPULATION", "SIZES", "evaluate", "is_schedulable",
]
