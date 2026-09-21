"""Scenario runner for map-bf-scalability.

Runs a set of fixed-priority mapping tools on the nested map-bf-scalability
pool (fixed utilization, task count growing across the matrix columns) through
the raw harness, and refreshes the processed Excel + figure after every column.

The tools are defined here (duplicated from ``workspace/framework_paper/map-bf/bf.py``)
so the script is self-contained and it is clear what is being evaluated:

- ``pd`` / ``hopa``: baselines (PD and HOPA) that only assign priorities.
- ``gdpa-prio``: GDPA optimizing only priorities (mapping fixed).
- ``gdpa-100/200/500``: bounded multi-start GDPA over mapping + priorities,
  with ``chunk * restarts`` total iterations (25x4, 20x10, 50x10).
- ``bf``: vectorized exhaustive search over mappings + priorities.

All tools get a 1800 s budget per system (a timeout counts as not schedulable).

Artifacts go to ``map-bf-scalability-<u>/`` and carry the scenario name:
``map-bf-scalability-<u>_raw.json`` plus the processed Excel and figure
``map-bf-scalability-<u>_processed.xlsx/.png/.pdf``.

    .venv/bin/python workspace/framework_paper/map-bf-scalability/map-bf-scalability.py
"""

import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from assignment.bf_assignment import BruteForceFPMappingAssignment
from assignment.hopa_assignment import HOPAssignment
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import AvgSeparationDelta
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import FPHandler, FPMappingHandler
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from vector.vector_fp import (MappingPrioritiesMatrix, PrioritiesMatrix,
                              VectorFPGradientFunction, VectorHolisticFPAnalysis)

import harness
import process
import systems

HERE = Path(__file__).resolve().parent

# Defaults for the scenario (all overridable from the command line).
TIMEOUT = 1800.0
N_SYSTEMS = systems.N_SYSTEMS
UTILIZATION = systems.UTILIZATION
THREADS = 6
BF_BATCH_SIZE = 10000
BF_PRUNE = True
PRUNE_OVER_UTILIZED = True
VECTOR_COST = True
ORDER = ("pd", "hopa", "gdpa-prio", "gdpa-100", "gdpa-200", "gdpa-500", "bf")

# gdpa-prio keeps the mapping fixed, so overloaded processors keep the scalar
# analysis near its pathological fixed-point regime; each call is capped.
SCALAR_ANALYSIS_MAX_TIME = 0.5


def pd_mapping_fp(system: LinearSystem) -> bool:
    pd = PDAssignment(normalize=True)
    pd.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def hopa_mapping_fp(system: LinearSystem) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    HOPAssignment(analysis=analysis).apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def gdpa_prio_fp(system: LinearSystem, prune_over_utilized: bool = False,
                 vector_cost: bool = False) -> bool:
    """GDPA optimizing only priorities, keeping the (contended) mapping."""
    parameter_handler = FPHandler()
    gradient_function = VectorFPGradientFunction(scenarios_builder=PrioritiesMatrix())
    if vector_cost:
        analysis = VectorHolisticFPAnalysis(limit_factor=10,
                                            cache=gradient_function.cache)
    else:
        analysis = HolisticFPAnalysis(limit_factor=10, reset=False,
                                      max_time=SCALAR_ANALYSIS_MAX_TIME,
                                      prune_over_utilized=prune_over_utilized)
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100)
    update_function = NoisyAdam()
    optimizer = GradientDescentOptimizer(parameter_handler=parameter_handler,
                                         cost_function=cost_function,
                                         stop_function=stop_function,
                                         gradient_function=gradient_function,
                                         update_function=update_function,
                                         verbose=False)

    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


class BlockDelta(AvgSeparationDelta):
    """Shared AvgSeparationDelta with independent finite-difference steps for
    the mapping block and the priority block (``None`` keeps the shared one)."""

    def __init__(self, sigma, mapping_prefix, mapping_delta=None, priority_delta=None):
        super().__init__(sigma=sigma)
        self.mapping_prefix = mapping_prefix
        self.mapping_delta = mapping_delta
        self.priority_delta = priority_delta

    def apply(self, system, x):
        base = super().apply(system, x)
        out = []
        for i in range(len(x)):
            if i < self.mapping_prefix:
                out.append(base[i] if self.mapping_delta is None else self.mapping_delta)
            else:
                out.append(base[i] if self.priority_delta is None else self.priority_delta)
        return out


def _gdpa_mapping(system, limit, lr=3.0, warmup=30, sigma=1.5,
                  mapping_delta=None, priority_delta=None, seed=1,
                  prune_over_utilized=False, vector_cost=False):
    parameter_handler = FPMappingHandler()
    gradient_function = VectorFPGradientFunction(scenarios_builder=MappingPrioritiesMatrix(),
                                                 sigma=sigma)
    if mapping_delta is not None or priority_delta is not None:
        p = len(system.processors)
        t = len(system.tasks)
        gradient_function.delta_function = BlockDelta(
            sigma, p * t, mapping_delta=mapping_delta, priority_delta=priority_delta)
    if vector_cost:
        analysis = VectorHolisticFPAnalysis(limit_factor=10,
                                            cache=gradient_function.cache)
    else:
        analysis = HolisticFPAnalysis(limit_factor=10, reset=False,
                                      prune_over_utilized=prune_over_utilized)
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=limit)
    update_function = NoisyAdam(
        lr=lr, seed=seed,
        warmup_iterations=warmup,
        warmup_mask=parameter_handler.mapping_mask(system))
    optimizer = GradientDescentOptimizer(parameter_handler=parameter_handler,
                                         cost_function=cost_function,
                                         stop_function=stop_function,
                                         gradient_function=gradient_function,
                                         update_function=update_function,
                                         verbose=False)

    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def gdpa_ms_fp(system: LinearSystem, chunk: int, restarts: int,
               prune_over_utilized: bool = False,
               vector_cost: bool = False) -> bool:
    """Bounded multi-start GDPA over mapping + priorities.

    Each restart gets ``chunk`` iterations with large per-block finite-difference
    steps and a different noise seed; the total is bounded by ``chunk * restarts``
    (the number in the method name).
    """
    steps = dict(lr=10.0, warmup=0, mapping_delta=2.0, priority_delta=2.0,
                 prune_over_utilized=prune_over_utilized, vector_cost=vector_cost)
    for restart in range(restarts):
        candidate = deepcopy(system)
        if _gdpa_mapping(candidate, limit=chunk, seed=1 + restart, **steps):
            return True
    return False


def bf_mapping(system: LinearSystem, batch_size: int, prune: bool = True) -> bool:
    bf = BruteForceFPMappingAssignment(batch_size=batch_size, prune=prune)
    bf.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def build_tools(vector_cost=VECTOR_COST, prune_over_utilized=PRUNE_OVER_UTILIZED,
                bf_batch_size=BF_BATCH_SIZE, bf_prune=BF_PRUNE):
    """Return ``(labels, funcs)`` for the configured tools."""
    gdpa = dict(prune_over_utilized=prune_over_utilized, vector_cost=vector_cost)
    tools = [
        ("pd", pd_mapping_fp),
        ("hopa", hopa_mapping_fp),
        ("gdpa-prio", partial(gdpa_prio_fp, **gdpa)),
        ("gdpa-100", partial(gdpa_ms_fp, chunk=25, restarts=4, **gdpa)),
        ("gdpa-200", partial(gdpa_ms_fp, chunk=20, restarts=10, **gdpa)),
        ("gdpa-500", partial(gdpa_ms_fp, chunk=50, restarts=10, **gdpa)),
        ("bf", partial(bf_mapping, batch_size=bf_batch_size, prune=bf_prune)),
    ]
    return zip(*tools)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--systems", type=int, default=N_SYSTEMS,
                        help=f"number of base systems (default: {N_SYSTEMS})")
    parser.add_argument("-u", "--utilization", type=float, default=UTILIZATION,
                        help=f"fixed utilization (default: {UTILIZATION})")
    parser.add_argument("--threads", type=int, default=THREADS,
                        help=f"parallel worker processes (default: {THREADS})")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"subset of tools to run, from {ORDER} (default: all)")
    parser.add_argument("--vector-cost", action=argparse.BooleanOptionalAction,
                        default=VECTOR_COST,
                        help="use the vectorized analysis (and cache) for the GDPA cost")
    parser.add_argument("--batch-size", type=int, default=BF_BATCH_SIZE,
                        help=f"brute-force batch size (default: {BF_BATCH_SIZE})")
    parser.add_argument("--deadline-factor-min", type=float,
                        default=systems.DEADLINE_FACTOR_MIN,
                        help="lower bound of the per-flow deadline factor "
                             f"(default: {systems.DEADLINE_FACTOR_MIN})")
    parser.add_argument("--deadline-factor-max", type=float,
                        default=systems.DEADLINE_FACTOR_MAX,
                        help="upper bound of the per-flow deadline factor "
                             f"(default: {systems.DEADLINE_FACTOR_MAX})")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="output directory (default: map-bf-scalability-<u>/ next to this script)")
    args = parser.parse_args()

    labels, funcs = build_tools(vector_cost=args.vector_cost,
                                bf_batch_size=args.batch_size)
    labels, funcs = list(labels), list(funcs)
    if args.methods:
        unknown = set(args.methods) - set(labels)
        if unknown:
            parser.error(f"unknown methods {sorted(unknown)}; choose from {ORDER}")
        keep = [i for i, label in enumerate(labels) if label in args.methods]
        labels = [labels[i] for i in keep]
        funcs = [funcs[i] for i in keep]
    timeouts = {label: TIMEOUT for label in labels}

    pool = systems.generate_pool(n_systems=args.systems,
                                 utilization=args.utilization,
                                 deadline_factor_min=args.deadline_factor_min,
                                 deadline_factor_max=args.deadline_factor_max,
                                 verbose=True)
    columns = [str(size) for size in systems.SIZES]

    default_deadline = (args.deadline_factor_min == systems.DEADLINE_FACTOR_MIN
                        and args.deadline_factor_max == systems.DEADLINE_FACTOR_MAX)
    eval_name = f"map-bf-scalability-{args.utilization:g}"
    if not default_deadline:
        eval_name += (f"-df{args.deadline_factor_min:g}-{args.deadline_factor_max:g}")
    out = Path(args.output_dir) if args.output_dir else HERE / eval_name
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"{eval_name}_raw.json"
    excel_path = out / f"{eval_name}_processed.xlsx"

    def refresh(column, records, done_columns):
        schedulable, times, finished = process.build_tables(records, labels, done_columns)
        process.write_outputs(schedulable, times, finished, str(excel_path),
                              xlabel="Tasks")
        print(f"    (excel+figure updated after column {column}: {done_columns})",
              flush=True)

    records = harness.evaluate(pool, labels, funcs, threads=args.threads,
                               columns=columns, timeouts=timeouts,
                               output=str(raw_path), on_column=refresh)

    schedulable, times, finished = process.build_tables(records, labels, columns)
    process.write_outputs(schedulable, times, finished, str(excel_path), xlabel="Tasks")
    print(f"\n=== schedulable (out of {args.systems}) ===")
    print(schedulable.to_string())
    print("\n=== mean time over schedulable (s) ===")
    print(times.round(3).to_string())
    print("\n=== finished ===")
    print(finished.to_string())
    print(f"\nwrote {out}/{eval_name}_raw.json, "
          f"{eval_name}_processed.xlsx/.png/.pdf")


if __name__ == "__main__":
    main()
