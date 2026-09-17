"""Diagnose the GDPA vs brute-force optimality gap in the map-bf scenario.

For selected utilization levels it runs GDPA on every system and, only for the
systems GDPA fails, runs the brute force. Cases the brute force solves but GDPA
does not are classified by comparing the initial, GDPA and brute-force
mappings (and per-processor utilisation):

  a) GDPA left the mapping unchanged from the initial (contended) one,
  b) GDPA reached a mapping different from the brute-force one,
  c) GDPA reached the same mapping as the brute force (priorities differ).

With --trace it also prints GDPA's cost/mapping per iteration for those cases.

Usage (from ``code/``):

    python workspace/framework_paper/map-bf/diagnose.py --u 0.752632 0.794737 --trace
"""

import argparse
from copy import deepcopy

import bf  # sibling module (script directory is on sys.path)

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from assignment.bf_assignment import BruteForceFPMappingAssignment
from examples.generator import set_system_utilization
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import FPMappingHandler
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from vector.vector_fp import MappingPrioritiesMatrix, VectorFPGradientFunction


def mapping_of(system):
    procs = system.processors
    return tuple(procs.index(t.processor) for t in system.tasks)


def proc_utils(system):
    utils = [0.0] * len(system.processors)
    procs = system.processors
    for t in system.tasks:
        utils[procs.index(t.processor)] += t.wcet / t.period
    return [round(x, 3) for x in utils]


def run_bf(system):
    brute = BruteForceFPMappingAssignment(batch_size=10000, prune=True)
    brute.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def run_gdpa(system, limit=200, warmup=30, callback=None):
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    handler = FPMappingHandler()
    cost = InvslackCost(parameter_handler=handler, analysis=analysis)
    stop = ThresholdStopFunction(limit=limit)
    gradient = VectorFPGradientFunction(scenarios_builder=MappingPrioritiesMatrix())
    update = NoisyAdam(warmup_iterations=warmup, warmup_mask=handler.mapping_mask(system))
    optimizer = GradientDescentOptimizer(parameter_handler=handler, cost_function=cost,
                                         stop_function=stop, gradient_function=gradient,
                                         update_function=update, callback=callback,
                                         verbose=False)
    PDAssignment(normalize=True).apply(system)
    optimizer.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def make_tracer(history):
    def callback(t, S, x, xb, cost, best, ref_cost):
        history.append((t, cost, mapping_of(S)))
    return callback


def print_trace(history):
    last = None
    for t, cost, mapping in history:
        if mapping != last:
            print(f"      iter {t:3d}: cost={cost:+.3f} mapping={mapping}")
            last = mapping
    if history:
        t_end, cost_end, _ = history[-1]
        print(f"      last iter {t_end}: cost={cost_end:+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-u", "--utilizations", type=float, nargs="+",
                        default=[0.752632, 0.794737],
                        help="utilization levels to inspect")
    parser.add_argument("--n", type=int, default=25, help="number of systems")
    parser.add_argument("--limit", type=int, default=200, help="GDPA iteration limit")
    parser.add_argument("--trace", action="store_true",
                        help="trace GDPA cost/mapping per iteration for the gap cases")
    args = parser.parse_args()

    base = bf.get_systems(args.n)
    counts = {"a": 0, "b": 0, "c": 0}
    total_gap = 0

    for u in args.utilizations:
        print(f"\n########## u={u} ##########", flush=True)
        for idx, system in enumerate(base):
            initial = deepcopy(system)
            set_system_utilization(initial, u)

            gdpa = deepcopy(initial)
            history = []
            gdpa_ok = run_gdpa(gdpa, limit=args.limit,
                               callback=make_tracer(history) if args.trace else None)
            if gdpa_ok:
                continue

            brute = deepcopy(initial)
            if not run_bf(brute):
                continue  # neither solves it: not a gap case

            total_gap += 1
            init_map, gdpa_map, bf_map = (mapping_of(initial), mapping_of(gdpa),
                                          mapping_of(brute))
            if gdpa_map == bf_map:
                category = "c"
            elif gdpa_map == init_map:
                category = "a"
            else:
                category = "b"
            counts[category] += 1

            print(f"sys={idx:2d} category={category} "
                  f"({'same mapping as bf' if category == 'c' else 'mapping unchanged (initial)' if category == 'a' else 'different mapping from bf'})",
                  flush=True)
            print(f"   init {init_map} util={proc_utils(initial)}")
            print(f"   gdpa {gdpa_map} util={proc_utils(gdpa)}")
            print(f"   bf   {bf_map} util={proc_utils(brute)}")
            if args.trace:
                print_trace(history)

    print(f"\n=== gap cases: {total_gap} | categories a={counts['a']} b={counts['b']} c={counts['c']} ===")


if __name__ == "__main__":
    main()
