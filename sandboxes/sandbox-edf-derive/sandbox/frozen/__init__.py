"""Evaluation core (shared with the sibling ``sandbox-edf-techniques``).

    from frozen import harness, pool, explain

    report = harness.evaluate()               # every technique in algorithms/
    ok, details = explain(system)             # per-task interference breakdown

Exposes the holistic local-EDF analysis (the judge), an **instrumented** version
that reports the interference breakdown, the model, the deterministic example
pool and the evaluation harness.
"""

from .model import Flow, LinearSystem, Processor, SchedulerType, Task, TaskType
from .analysis import HolisticLocalEDFAnalysis, LimitFactorReachedException
from .instrumented import (InstrumentedHolisticLocalEDFAnalysis, explain,
                           format_detail)
from . import pool, harness, instrumented
from .pool import (POPULATION, SEED, SIZES, UTILIZATIONS, get_systems,
                   set_utilization)
from .harness import evaluate, is_schedulable

__all__ = [
    "Flow", "LinearSystem", "Processor", "SchedulerType", "Task", "TaskType",
    "HolisticLocalEDFAnalysis", "LimitFactorReachedException",
    "InstrumentedHolisticLocalEDFAnalysis", "explain", "format_detail",
    "pool", "harness", "instrumented",
    "POPULATION", "SEED", "SIZES", "UTILIZATIONS", "get_systems",
    "set_utilization", "evaluate", "is_schedulable",
]
