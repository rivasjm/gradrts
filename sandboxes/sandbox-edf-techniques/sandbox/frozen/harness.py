"""Evaluation harness.

A *technique* is a callable ``technique(system) -> None`` that assigns each
task's local deadline (``task.deadline``); returning a sequence of deadlines in
``system.tasks`` order is also accepted.

``evaluate`` sweeps the example pool over the utilization levels, applies every
technique to a fresh copy of each instance and judges the result with the
holistic local-EDF analysis: an instance counts as schedulable iff every flow
meets its end-to-end deadline. With no explicit techniques it evaluates every
technique registered in the ``algorithms`` package (with ``@register``), which is
how the registered techniques are discovered automatically.
"""

import time
from copy import deepcopy

from . import pool as _pool
from .analysis import HolisticLocalEDFAnalysis


def is_schedulable(system) -> bool:
    """True iff every flow meets its deadline under the holistic local-EDF analysis."""
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def _apply_result(system, result):
    """Accept either an in-place mutation or a returned deadline sequence."""
    if result is None:
        return
    tasks = system.tasks
    if len(result) != len(tasks):
        raise ValueError(f"returned {len(result)} deadlines for {len(tasks)} tasks")
    for task, deadline in zip(tasks, result):
        task.deadline = deadline


def run_technique(technique, system):
    """Apply ``technique`` to ``system``; return its wall-clock time (seconds)."""
    start = time.perf_counter()
    result = technique(system)
    _apply_result(system, result)
    return time.perf_counter() - start


def evaluate(techniques=None, seed=_pool.SEED, size=15,
             population=_pool.POPULATION, utilizations=None, verbose=False):
    """Count how many instances each technique makes schedulable.

    Parameters
    ----------
    techniques:
        ``{name: technique}``; defaults to every technique registered in the
        ``algorithms`` package with ``@register``.
    seed, size, population:
        The pristine pool of example systems to draw from.
    utilizations:
        Utilization levels to sweep (default: ``pool.UTILIZATIONS``).
    verbose:
        Print the running counts as each level completes.

    Returns
    -------
    report : dict
        ``report["techniques"][name]`` holds ``schedulable`` (total),
        ``per_level`` (schedulable at each level), ``mean_time`` and
        ``total_time`` (seconds, over the technique itself) and ``errors``.
    """
    if techniques is None:
        from algorithms import load
        techniques = load()

    utilizations = list(utilizations) if utilizations is not None else list(_pool.UTILIZATIONS)
    systems = _pool.get_systems(seed=seed, size=size, population=population)
    names = list(techniques)
    n_levels = len(utilizations)
    instances = len(systems) * n_levels

    per_level = {name: [0] * n_levels for name in names}
    total_time = {name: 0.0 for name in names}
    errors = {name: [] for name in names}

    for ui, u in enumerate(utilizations):
        for si, base in enumerate(systems):
            work = deepcopy(base)
            _pool.set_utilization(work, u)
            for name, technique in techniques.items():
                system = deepcopy(work)
                try:
                    seconds = run_technique(technique, system)
                    schedulable = is_schedulable(system)
                except Exception as exc:  # keep the sweep alive and report it
                    seconds, schedulable = 0.0, False
                    if len(errors[name]) < 5:
                        errors[name].append(
                            f"u={u:.3f} sys={si}: {type(exc).__name__}: {exc}")
                if schedulable:
                    per_level[name][ui] += 1
                total_time[name] += seconds
        if verbose:
            counts = " ".join(f"{n}={sum(per_level[n][:ui + 1])}" for n in names)
            print(f"  u={u:.3f} ({ui + 1}/{n_levels}): {counts}", flush=True)

    return {
        "seed": seed,
        "size": size,
        "population": population,
        "utilizations": utilizations,
        "instances": instances,
        "techniques": {
            name: {
                "schedulable": sum(per_level[name]),
                "per_level": per_level[name],
                "mean_time": total_time[name] / instances if instances else 0.0,
                "total_time": total_time[name],
                "errors": errors[name],
            }
            for name in names
        },
    }
