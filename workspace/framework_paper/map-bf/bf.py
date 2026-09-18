"""Brute-force reference for the joint mapping + fixed-priority scenario.

GDPA is compared against an exact brute-force search over task-to-processor
mappings and fixed priorities (the same search space that the MAP scenario
optimizes). To keep the exhaustive search tractable the systems are small
(9 tasks on 3 processors) and start from an unbalanced/contended initial
mapping, which GDPA must repair. The brute force is an upper bound: every
system GDPA solves must also be solved by it.

Usage (from ``code/``):

    python workspace/framework_paper/map-bf/bf.py [--n 25] [-u 0.5 ... 0.9]
                                                  [--start 1] [--batch-size 10000]

Output goes to ``map-bf/map-bf-9/``.
"""

import argparse
import os
from copy import deepcopy
from functools import partial
from random import Random

import numpy as np

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from assignment.bf_assignment import (BruteForceFPMappingAssignment,
                                      BruteForceFPSequentialMappingAssignment)
from assignment.hopa_assignment import HOPAssignment
from examples.evaluation import SchedRatioEval
from examples.example_models import get_system
from examples.generator import set_system_utilization
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import AvgSeparationDelta
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import FPHandler, FPMappingHandler
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from vector.vector_fp import (MappingPrioritiesMatrix, PrioritiesMatrix,
                              VectorFPGradientFunction)

# size key (total tasks) -> flows x tasks per flow x processors
SIZES = {9: (3, 3, 3), 10: (2, 5, 3)}
POPULATION = 25
SEED = 42
DEADLINE_FACTOR_MIN = 0.5
DEADLINE_FACTOR_MAX = 1
PERIOD_MIN = 100
PERIOD_MAX = 1000
# HOPA runs up to 160 scalar analyses; some systems make a single analysis
# converge pathologically slowly, so each call is capped (baseline only).
SCALAR_ANALYSIS_MAX_TIME = 0.5
# bf-seq is the scalar brute force: cap each candidate's analysis for the same
# reason (it is a slow reference method).
BF_SEQ_ANALYSIS_MAX_TIME = 1.0
# utilizations between 50 % and 90 %
UTILIZATIONS = np.linspace(0.5, 0.9, 20)


def get_systems(n=POPULATION, size=9):
    """Small unbalanced population; the initial mapping is contended."""
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}, use one of {sorted(SIZES)}")
    rnd = Random(SEED)
    return [get_system(SIZES[size], rnd, balanced=False, name=str(i),
                       deadline_factor_min=DEADLINE_FACTOR_MIN,
                       deadline_factor_max=DEADLINE_FACTOR_MAX,
                       period_min=PERIOD_MIN, period_max=PERIOD_MAX)
            for i in range(n)]


def pd_mapping_fp(system: LinearSystem) -> bool:
    pd = PDAssignment(normalize=True)
    pd.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def hopa_mapping_fp(system: LinearSystem) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False,
                                  max_time=SCALAR_ANALYSIS_MAX_TIME)
    HOPAssignment(analysis=analysis).apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def gdpa_prio_fp(system: LinearSystem) -> bool:
    """GDPA optimizing only priorities, keeping the (contended) mapping.

    Since the mapping cannot move, over-loaded processors keep the scalar
    analysis near its pathological regime, so each call is capped like HOPA's.
    """
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False,
                                  max_time=SCALAR_ANALYSIS_MAX_TIME)
    parameter_handler = FPHandler()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100)
    gradient_function = VectorFPGradientFunction(scenarios_builder=PrioritiesMatrix())
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
                  mapping_delta=None, priority_delta=None, seed=1):
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False,
                                  prune_over_utilized=True)
    parameter_handler = FPMappingHandler()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=limit)
    gradient_function = VectorFPGradientFunction(scenarios_builder=MappingPrioritiesMatrix(),
                                                 sigma=sigma)
    if mapping_delta is not None or priority_delta is not None:
        p = len(system.processors)
        t = len(system.tasks)
        gradient_function.delta_function = BlockDelta(
            sigma, p * t, mapping_delta=mapping_delta, priority_delta=priority_delta)
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


def gdpa_ms_fp(system: LinearSystem, chunk: int, restarts: int) -> bool:
    """Bounded multi-start GDPA (selected by the map-bf tuning study).

    The optimizer uses large per-block finite-difference steps and learning
    rate, which fixes the mapping exploration; ``restarts`` attempts with
    different noise seeds escape the remaining local minima. Each attempt gets
    ``chunk`` iterations, so the total is bounded by ``chunk * restarts`` (the
    number in the method name).
    """
    steps = dict(lr=10.0, warmup=0, mapping_delta=2.0, priority_delta=2.0)
    for restart in range(restarts):
        candidate = deepcopy(system)
        if _gdpa_mapping(candidate, limit=chunk, seed=1 + restart, **steps):
            return True
    return False


def bf_mapping(system: LinearSystem, batch_size: int) -> bool:
    bf = BruteForceFPMappingAssignment(batch_size=batch_size, prune=True)
    bf.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def bf_seq_mapping(system: LinearSystem, max_time: float = BF_SEQ_ANALYSIS_MAX_TIME) -> bool:
    """Brute force evaluating every candidate with the scalar analysis (slow)."""
    bf = BruteForceFPSequentialMappingAssignment(prune=True, max_time=max_time)
    bf.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GDPA vs brute force (mapping + priorities)")
    parser.add_argument("--n", type=int, default=POPULATION,
                        help=f"number of systems (default: {POPULATION})")
    parser.add_argument("--size", type=int, choices=sorted(SIZES), default=9,
                        help=f"total tasks (default: 9); sizes: {SIZES}")
    parser.add_argument("-u", "--utilizations", type=float, nargs="+",
                        default=list(UTILIZATIONS),
                        help="utilization levels to sweep (default: 0.5..0.9, 20 levels)")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="brute-force batch size (default: 10000)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="subset of methods to run, e.g. --methods gdpa-prio "
                             "(default: all)")
    parser.add_argument("--start", type=int, default=1,
                        help="first utilization level to run, 1-based (default: 1); "
                             "previous levels are loaded from the output directory")
    parser.add_argument("-o", "--output-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="output directory (default: script directory)")
    args = parser.parse_args()

    eval_name = f"map-bf-{args.size}"
    systems = get_systems(args.n, args.size)

    tools = [
        ("pd", pd_mapping_fp),
        ("hopa", hopa_mapping_fp),
        ("gdpa-prio", gdpa_prio_fp),
        ("gdpa-100", partial(gdpa_ms_fp, chunk=25, restarts=4)),
        ("gdpa-200", partial(gdpa_ms_fp, chunk=20, restarts=10)),
        ("gdpa-500", partial(gdpa_ms_fp, chunk=50, restarts=10)),
        ("bf", partial(bf_mapping, batch_size=args.batch_size)),
    ]
    if args.size == 9:
        # scalar brute force is only practical on the smallest population
        tools.append(("bf-seq", bf_seq_mapping))

    if args.methods:
        unknown = set(args.methods) - {name for name, _ in tools}
        if unknown:
            parser.error(f"unknown methods {sorted(unknown)}; "
                         f"choose from {[name for name, _ in tools]}")
        tools = [t for t in tools if t[0] in args.methods]

    if not 1 <= args.start <= len(args.utilizations):
        parser.error(f"--start must be between 1 and {len(args.utilizations)}")

    labels, funcs = zip(*tools)
    output_dir = os.path.join(args.output_dir, eval_name)
    os.makedirs(output_dir, exist_ok=True)
    runner = SchedRatioEval(eval_name, labels=labels, funcs=funcs,
                            systems=systems, utilizations=np.array(args.utilizations),
                            threads=6, utilization_func=set_system_utilization,
                            output_dir=output_dir)
    runner.run(start_index=args.start - 1)
