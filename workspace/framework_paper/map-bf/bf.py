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
from assignment.bf_assignment import BruteForceFPMappingAssignment
from assignment.hopa_assignment import HOPAssignment
from examples.evaluation import SchedRatioEval
from examples.example_models import get_system
from examples.generator import set_system_utilization
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import FPHandler, FPMappingHandler
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from vector.vector_fp import (MappingPrioritiesMatrix, PrioritiesMatrix,
                              VectorFPGradientFunction)

# flows x tasks per flow x processors -> 9 tasks
SIZE = (3, 3, 3)
SIZE_KEY = SIZE[0] * SIZE[1]
POPULATION = 25
SEED = 42
DEADLINE_FACTOR_MIN = 0.5
DEADLINE_FACTOR_MAX = 1
PERIOD_MIN = 100
PERIOD_MAX = 1000
# HOPA runs up to 160 scalar analyses; some systems make a single analysis
# converge pathologically slowly, so each call is capped (baseline only).
SCALAR_ANALYSIS_MAX_TIME = 0.5
# utilizations between 50 % and 90 %
UTILIZATIONS = np.linspace(0.5, 0.9, 20)


def get_systems(n=POPULATION):
    """Small unbalanced population; the initial mapping is contended."""
    rnd = Random(SEED)
    return [get_system(SIZE, rnd, balanced=False, name=str(i),
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


def gdpa_mapping_fp(system: LinearSystem, limit: int) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    parameter_handler = FPMappingHandler()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=limit)
    gradient_function = VectorFPGradientFunction(scenarios_builder=MappingPrioritiesMatrix())

    update_function = NoisyAdam(
        warmup_iterations=30,
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


def bf_mapping(system: LinearSystem, batch_size: int) -> bool:
    bf = BruteForceFPMappingAssignment(batch_size=batch_size, prune=True)
    bf.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GDPA vs brute force (mapping + priorities)")
    parser.add_argument("--n", type=int, default=POPULATION,
                        help=f"number of systems (default: {POPULATION})")
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

    eval_name = f"map-bf-{SIZE_KEY}"
    systems = get_systems(args.n)

    tools = [
        ("pd", pd_mapping_fp),
        ("hopa", hopa_mapping_fp),
        ("gdpa-prio", gdpa_prio_fp),
        ("gdpa-100", partial(gdpa_mapping_fp, limit=100)),
        ("gdpa-200", partial(gdpa_mapping_fp, limit=200)),
        ("bf", partial(bf_mapping, batch_size=args.batch_size)),
    ]

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
