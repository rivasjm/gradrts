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
from examples.generator import set_system_utilization, set_utilization
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import BlockConstantDelta
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import CompoundHandler, FPHandler, MappingHandler
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from vector.vector_fp import (MappingPrioritiesMatrix, PrioritiesMatrix,
                              VectorFPGradientFunction, VectorHolisticFPAnalysis)

# size key (total tasks) -> flows x tasks per flow x processors
SIZES = {9: (3, 3, 3), 10: (2, 5, 3), 12: (4, 3, 3)}
POPULATION = 25
SEED = 42
DEADLINE_FACTOR_MIN = 0.5
DEADLINE_FACTOR_MAX = 1
PERIOD_MIN = 100
PERIOD_MAX = 1000
# gdpa-prio keeps the mapping fixed, so overloaded processors keep the scalar
# analysis near its pathological fixed-point regime; each call is capped.
SCALAR_ANALYSIS_MAX_TIME = 0.5
# bf-seq is the scalar brute force: cap each candidate's analysis for the same
# reason (it is a slow reference method).
BF_SEQ_ANALYSIS_MAX_TIME = 1.0
# utilizations between 50 % and 90 %
UTILIZATIONS = np.linspace(0.5, 0.9, 20)


def get_systems(n=POPULATION, size=9, balanced=False):
    """Population; the balanced variant assigns tasks evenly (assign_balanced),
    the unbalanced one starts from a contended mapping."""
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}, use one of {sorted(SIZES)}")
    rnd = Random(SEED)
    return [get_system(SIZES[size], rnd, balanced=balanced, name=str(i),
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
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    HOPAssignment(analysis=analysis).apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def gdpa_prio_fp(system: LinearSystem, prune_over_utilized: bool = False,
                 vector_cost: bool = False) -> bool:
    """GDPA optimizing only priorities, keeping the (contended) mapping.

    Since the mapping cannot move, over-loaded processors keep the scalar
    analysis near its pathological regime, so each call is capped like HOPA's.
    """
    parameter_handler = FPHandler()
    if vector_cost:
        gradient_function = VectorFPGradientFunction(scenarios_builder=PrioritiesMatrix())
        analysis = VectorHolisticFPAnalysis(limit_factor=10,
                                            cache=gradient_function.cache)
    else:
        gradient_function = VectorFPGradientFunction(scenarios_builder=PrioritiesMatrix())
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


# Bounded multi-start GDPA steps selected by the map-bf tuning study: large
# per-block finite-difference steps and learning rate so the mapping moves, and
# no warmup. Paired with ``chunk`` (the optimizer restarts every ``chunk``
# iterations with a fresh seed), the total budget is ``limit``.
MS = dict(lr=10.0, warmup=0, mapping_delta=2.0, priority_delta=2.0)


def _gdpa_mapping(system, limit, lr=3.0, warmup=30, sigma=1.5,
                  mapping_delta=None, priority_delta=None, seed=1,
                  prune_over_utilized=False, vector_cost=False,
                  chunk=0, restart_x=None):
    """GDPA over mapping + priorities.

    ``limit`` is the total iteration budget and ``chunk`` the iterations per
    restart (0 disables restarts); with restarts the optimizer returns the best
    solution found across all chunks. Defaults are a single run.
    """
    mapping_handler = MappingHandler()
    parameter_handler = CompoundHandler([mapping_handler, FPHandler()])
    gradient_function = VectorFPGradientFunction(scenarios_builder=MappingPrioritiesMatrix(),
                                                 sigma=sigma)
    if mapping_delta is not None or priority_delta is not None:
        mapping_size, priority_size = parameter_handler.block_sizes(system)
        gradient_function.delta_function = BlockConstantDelta(
            sigma=sigma, blocks=[(mapping_size, mapping_delta),
                                 (priority_size, priority_delta)])
    if vector_cost:
        # reuse the gradient's cache and over-utilization shortcut for the cost
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
        warmup_mask=parameter_handler.block_mask(system, mapping_handler))
    optimizer = GradientDescentOptimizer(parameter_handler=parameter_handler,
                                         cost_function=cost_function,
                                         stop_function=stop_function,
                                         gradient_function=gradient_function,
                                         update_function=update_function,
                                         verbose=False,
                                         chunk=chunk,
                                         restart_x=restart_x)

    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def bf_mapping(system: LinearSystem, batch_size: int, prune: bool = True) -> bool:
    bf = BruteForceFPMappingAssignment(batch_size=batch_size, prune=prune)
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
    parser.add_argument("--balanced", action="store_true",
                        help="balanced population with set_utilization (default: "
                             "unbalanced contended mapping with set_system_utilization)")
    parser.add_argument("-u", "--utilizations", type=float, nargs="+",
                        default=list(UTILIZATIONS),
                        help="utilization levels to sweep (default: 0.5..0.9, 20 levels)")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="brute-force batch size (default: 10000)")
    parser.add_argument("--no-bf-prune", action="store_true",
                        help="disable the over-utilization prune in bf (pure enumeration)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="subset of methods to run, e.g. --methods gdpa-prio "
                             "(default: all)")
    parser.add_argument("--vector-cost", action="store_true",
                        help="use the vectorized analysis (and its cache) for the "
                             "GDPA cost too, instead of the scalar one")
    parser.add_argument("--start", type=int, default=1,
                        help="first utilization level to run, 1-based (default: 1); "
                             "previous levels are loaded from the output directory")
    parser.add_argument("-o", "--output-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="output directory (default: script directory)")
    args = parser.parse_args()

    eval_name = f"map-bf-{args.size}" + ("-balanced" if args.balanced else "")
    systems = get_systems(args.n, args.size, args.balanced)
    # the over-utilization shortcut is enabled for size 10 only
    prune = args.size == 10
    utilization_func = set_utilization if args.balanced else set_system_utilization

    all_tools = [
        ("pd", pd_mapping_fp),
        ("hopa", hopa_mapping_fp),
        ("gdpa-prio", partial(gdpa_prio_fp, prune_over_utilized=prune,
                              vector_cost=args.vector_cost)),
        ("gdpa-100", partial(_gdpa_mapping, limit=100, chunk=25,
                             prune_over_utilized=prune, vector_cost=args.vector_cost, **MS)),
        ("gdpa-200", partial(_gdpa_mapping, limit=200, chunk=20,
                             prune_over_utilized=prune, vector_cost=args.vector_cost, **MS)),
        ("gdpa-500", partial(_gdpa_mapping, limit=500, chunk=50,
                             prune_over_utilized=prune, vector_cost=args.vector_cost, **MS)),
        ("bf", partial(bf_mapping, batch_size=args.batch_size,
                       prune=not args.no_bf_prune)),
        ("bf-seq", bf_seq_mapping),
    ]

    if args.methods:
        unknown = set(args.methods) - {name for name, _ in all_tools}
        if unknown:
            parser.error(f"unknown methods {sorted(unknown)}; "
                         f"choose from {[name for name, _ in all_tools]}")
        tools = [t for t in all_tools if t[0] in args.methods]
    else:
        # bf-seq (scalar brute force) is only in the default run for size 9
        tools = [t for t in all_tools if t[0] != "bf-seq" or args.size == 9]

    if not 1 <= args.start <= len(args.utilizations):
        parser.error(f"--start must be between 1 and {len(args.utilizations)}")

    labels, funcs = zip(*tools)
    output_dir = os.path.join(args.output_dir, eval_name)
    os.makedirs(output_dir, exist_ok=True)
    runner = SchedRatioEval(eval_name, labels=labels, funcs=funcs,
                            systems=systems, utilizations=np.array(args.utilizations),
                            threads=6, utilization_func=utilization_func,
                            output_dir=output_dir)
    runner.run(start_index=args.start - 1)
