"""Tuning study for the GDPA mapping optimization in map-bf.

The gap diagnostic showed that everything GDPA misses (vs brute force) comes
from mapping exploration: in some systems the mapping never leaves the initial
contended one, and in most it converges to a worse mapping than the exact
optimum. This script sweeps the knobs that could change that, running only
GDPA (the brute force is not needed: it is the fixed upper bound, 459/459).

Knobs:
  limit          stop-function iteration budget
  warmup, lr, gamma, noise, seed   NoisyAdam / Adam
  sigma          finite-difference step of the shared AvgSeparationDelta
  mapping_delta  override the finite-difference step for the mapping block

Usage (from ``code/``):

    python workspace/framework_paper/map-bf/tune.py --phase study
    python workspace/framework_paper/map-bf/tune.py --list

or via ``bash workspace/framework_paper/map-bf/run_tuning.sh``.
"""

import argparse
import copy
import time
from multiprocessing import Pool

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from examples.generator import set_system_utilization
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import AvgSeparationDelta
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import FPMappingHandler
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import Adam, NoisyAdam
from vector.vector_fp import MappingPrioritiesMatrix, VectorFPGradientFunction

import bf  # sibling module: get_systems

# Utilization levels where the gap is concentrated (0.71 .. 0.86).
GAP_LEVELS = [0.710526, 0.731579, 0.752632, 0.773684,
              0.794737, 0.815789, 0.836842, 0.857895]

BASE = {"limit": 200, "warmup": 30, "lr": 3.0, "gamma": 0.9, "noise": True,
        "seed": 1, "sigma": 1.5, "mapping_delta": None}

_BASE_SYSTEMS = None


class BlockDelta(AvgSeparationDelta):
    """Shared AvgSeparationDelta, with independent steps per parameter block.

    Coordinates before ``mapping_prefix`` are the mapping block; the rest are
    priorities. A ``None`` override keeps the shared delta for that block, so
    the two blocks can be steered independently (e.g. a larger step for the
    mapping without disturbing the priorities).
    """

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


class MarginMappingHandler(FPMappingHandler):
    """FPMappingHandler with a different initial margin between the current
    processor and the others (default is 0.55 vs 0.45)."""

    def __init__(self, current=0.55, other=0.45):
        super().__init__()
        self._current = current
        self._other = other

    def extract(self, S):
        mapping = [self._current if t.processor == p else self._other
                   for t in S.tasks for p in S.processors]
        return mapping + self.fp_handler.extract(S)


def make_handler(cfg):
    margin = cfg.get("margin")
    if margin is None:
        return FPMappingHandler()
    return MarginMappingHandler(current=margin[0], other=margin[1])


def _gdpa_once(system, cfg, callback=None):
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    handler = make_handler(cfg)
    cost = InvslackCost(parameter_handler=handler, analysis=analysis)
    stop = ThresholdStopFunction(limit=cfg.get("limit", 200))
    gradient = VectorFPGradientFunction(scenarios_builder=MappingPrioritiesMatrix(),
                                        sigma=cfg.get("sigma", 1.5))
    if cfg.get("mapping_delta") is not None or cfg.get("priority_delta") is not None:
        p = len(system.processors)
        t = len(system.tasks)
        gradient.delta_function = BlockDelta(cfg.get("sigma", 1.5), p * t,
                                             mapping_delta=cfg.get("mapping_delta"),
                                             priority_delta=cfg.get("priority_delta"))
    if cfg.get("noise", True):
        update = NoisyAdam(lr=cfg.get("lr", 3.0), gamma=cfg.get("gamma", 0.9),
                           seed=cfg.get("seed", 1),
                           warmup_iterations=cfg.get("warmup", 30),
                           warmup_mask=handler.mapping_mask(system))
    else:
        update = Adam(lr=cfg.get("lr", 3.0))
    optimizer = GradientDescentOptimizer(parameter_handler=handler,
                                         cost_function=cost,
                                         stop_function=stop,
                                         gradient_function=gradient,
                                         update_function=update, callback=callback,
                                         verbose=False)
    PDAssignment(normalize=True).apply(system)
    optimizer.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def gdpa_run(system, cfg):
    """Run GDPA with periodic restarts.

    ``chunk`` is the iteration budget of each restart and ``restarts`` how many
    are tried (each with a different noise seed); the total is bounded by
    ``restarts * chunk``. Returns (schedulable, first_success_iteration)."""
    restarts = cfg.get("restarts", 1)
    chunk = cfg.get("chunk", cfg.get("limit", 200))
    for r in range(restarts):
        candidate = copy.deepcopy(system)
        first = {"iter": None}

        def callback(t, S, x, xb, cost, best, ref_cost, _first=first):
            if _first["iter"] is None and cost < 0:
                _first["iter"] = t

        if _gdpa_once(candidate, dict(cfg, limit=chunk, seed=cfg.get("seed", 1) + r),
                      callback=callback):
            return True, first["iter"]
    return False, None


def label(cfg):
    if cfg.get("chunk") is not None:
        head = [f"chunk={cfg['chunk']}", f"restarts={cfg.get('restarts', 1)}"]
    else:
        head = [f"lim={cfg.get('limit', 200)}"]
        if cfg.get("restarts", 1) > 1:
            head.append(f"restarts={cfg['restarts']}")
    parts = head + [f"wu={cfg.get('warmup', 30)}", f"lr={cfg.get('lr', 3.0)}"]
    parts.append("noise" if cfg.get("noise", True) else "adam")
    if cfg.get("noise", True):
        parts.append(f"g={cfg.get('gamma', 0.9)}")
    if cfg.get("sigma", 1.5) != 1.5:
        parts.append(f"sigma={cfg['sigma']}")
    if cfg.get("mapping_delta") is not None:
        parts.append(f"mdelta={cfg['mapping_delta']}")
    if cfg.get("priority_delta") is not None:
        parts.append(f"pdelta={cfg['priority_delta']}")
    if cfg.get("seed", 1) != 1:
        parts.append(f"seed={cfg['seed']}")
    if cfg.get("margin") is not None:
        parts.append(f"margin={cfg['margin']}")
    return ",".join(parts)


def _init_worker(n_systems, size):
    global _BASE_SYSTEMS
    _BASE_SYSTEMS = bf.get_systems(n_systems)


def _worker(args):
    idx, u, cfg = args
    system = copy.deepcopy(_BASE_SYSTEMS[idx])
    set_system_utilization(system, u)
    t0 = time.perf_counter()
    ok, first = gdpa_run(system, cfg)
    return ok, time.perf_counter() - t0, first


def build_phase(phase):
    if phase == "baseline":
        return [dict(BASE)]
    if phase == "warmup":
        return [dict(BASE, warmup=w) for w in (0, 10, 30)]
    if phase == "lr":
        return [dict(BASE, lr=lr) for lr in (1.0, 3.0, 10.0)]
    if phase == "sigma":
        return [dict(BASE, sigma=s) for s in (0.5, 1.0, 1.5, 3.0)]
    if phase == "mapdelta":
        return [dict(BASE, mapping_delta=d) for d in (0.02, 0.05, 0.1, 0.2, 0.5)]
    if phase == "noise":
        return [dict(BASE, noise=False),
                dict(BASE, gamma=0.99),
                dict(BASE, seed=2),
                dict(BASE, seed=3)]
    if phase == "limit":
        return [dict(BASE, limit=lim) for lim in (100, 200, 500)]
    if phase == "study":
        return [
            dict(BASE),                                   # baseline
            dict(BASE, warmup=0),
            dict(BASE, warmup=10),
            dict(BASE, lr=1.0),
            dict(BASE, lr=10.0),
            dict(BASE, sigma=0.5),
            dict(BASE, sigma=3.0),
            dict(BASE, mapping_delta=0.02),
            dict(BASE, mapping_delta=0.05),
            dict(BASE, mapping_delta=0.2),
            dict(BASE, mapping_delta=0.5),
            dict(BASE, noise=False),
            dict(BASE, gamma=0.99),
            dict(BASE, seed=2),
        ]
    if phase == "combine":
        return [
            dict(BASE, lr=10.0),
            dict(BASE, mapping_delta=0.5),
            dict(BASE, lr=10.0, mapping_delta=0.5),
            dict(BASE, mapping_delta=1.0),
            dict(BASE, lr=10.0, mapping_delta=1.0),
            dict(BASE, priority_delta=0.5),
            dict(BASE, priority_delta=1.0),
            dict(BASE, mapping_delta=0.5, priority_delta=0.5),
            dict(BASE, lr=10.0, mapping_delta=0.5, priority_delta=0.5),
            dict(BASE, sigma=3.0, mapping_delta=0.5),
        ]
    if phase == "push":
        best = dict(BASE, lr=10.0, mapping_delta=1.0, priority_delta=1.0)
        return [
            dict(BASE, lr=10.0, mapping_delta=0.5, priority_delta=0.5),
            best,
            dict(BASE, lr=10.0, mapping_delta=2.0, priority_delta=2.0),
            dict(BASE, lr=20.0, mapping_delta=1.0, priority_delta=1.0),
            dict(best, limit=500),
            dict(best, gamma=0.95),
            dict(best, seed=2),
        ]
    if phase == "alt":
        best = dict(BASE, lr=10.0, mapping_delta=1.0, priority_delta=1.0)
        return [
            best,
            dict(best, restarts=3),
            dict(best, restarts=5),
            dict(best, margin=(0.8, 0.2)),
            dict(best, margin=(0.95, 0.05)),
            dict(BASE, margin=(0.8, 0.2)),
        ]
    if phase == "final":
        big = dict(BASE, lr=10.0, mapping_delta=2.0, priority_delta=2.0)
        mid = dict(BASE, lr=10.0, mapping_delta=1.0, priority_delta=1.0)
        return [
            dict(mid, restarts=3),
            dict(mid, restarts=5),
            dict(big, restarts=3),
            dict(big, restarts=5),
            dict(big, restarts=5, limit=500),
            dict(big, restarts=5, seed=2),
        ]
    if phase == "winner":
        return [dict(BASE, lr=10.0, mapping_delta=2.0, priority_delta=2.0,
                     restarts=5, limit=500)]
    if phase == "budget":
        big = dict(BASE, lr=10.0, mapping_delta=2.0, priority_delta=2.0)
        return [
            dict(BASE, restarts=1, limit=200),                # baseline gdpa-200
            dict(big, limit=500, restarts=5),                 # current winner (<=2500)
            dict(big, restarts=4, chunk=50),                  # budget <=200
            dict(big, restarts=5, chunk=100),                 # budget <=500
            dict(big, restarts=10, chunk=50),                 # budget <=500
            dict(big, restarts=10, chunk=50, warmup=0),       # budget <=500, no warmup
            dict(big, restarts=20, chunk=25, warmup=0),       # budget <=500, short chunks
            dict(big, restarts=10, chunk=20, warmup=0),       # budget <=200, short chunks
        ]
    if phase == "scale":
        big = dict(BASE, lr=10.0, mapping_delta=2.0, priority_delta=2.0, warmup=0)
        return [
            dict(big, chunk=25, restarts=4),    # gdpa-100 candidate
            dict(big, chunk=20, restarts=5),    # gdpa-100, alt
            dict(big, chunk=10, restarts=10),   # gdpa-100, alt
            dict(big, chunk=50, restarts=2),    # gdpa-100, alt
            dict(big, chunk=20, restarts=25),   # gdpa-500, uniform chunk
        ]
    raise SystemExit(f"unknown phase {phase!r} (use --list)")


PHASES = ("baseline", "warmup", "lr", "sigma", "mapdelta", "noise", "limit",
          "study", "combine", "push", "alt", "final", "winner", "budget", "scale")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", default="study", help="phase to run (default: study)")
    parser.add_argument("-u", "--utilizations", type=float, nargs="+",
                        default=None, help="utilization levels (default: gap levels)")
    parser.add_argument("--levels", choices=("gap", "all"), default="gap",
                        help="'gap' levels (default) or all 20 levels")
    parser.add_argument("--n", type=int, default=25, help="number of systems")
    parser.add_argument("--threads", type=int, default=6, help="worker processes")
    parser.add_argument("--list", action="store_true", help="list phases and exit")
    args = parser.parse_args()

    if args.list:
        print("phases:", ", ".join(PHASES))
        return

    if args.utilizations is None:
        args.utilizations = list(bf.UTILIZATIONS) if args.levels == "all" else GAP_LEVELS

    _init_worker(args.n, 9)
    configs = build_phase(args.phase)
    summary = []
    print(f"phase={args.phase} | levels={args.utilizations} | n={args.n} | "
          f"configs={len(configs)}", flush=True)

    with Pool(args.threads, initializer=_init_worker, initargs=(args.n, 9)) as pool:
        for cfg in configs:
            t0 = time.perf_counter()
            total = 0
            worst = 0
            firsts = []
            for u in args.utilizations:
                results = pool.map(_worker, [(i, u, cfg) for i in range(args.n)])
                total += sum(int(ok) for ok, _, _ in results)
                worst = max(worst, max(dt for _, dt, _ in results))
                firsts += [f for ok, _, f in results if ok and f is not None]
            dt = time.perf_counter() - t0
            lab = label(cfg)
            denom = args.n * len(args.utilizations)
            avg_first = sum(firsts) / len(firsts) if firsts else float("nan")
            print(f"  {lab:55s} {total}/{denom} ({dt:.0f}s, worst_run={worst:.0f}s, "
                  f"avg_first_iter={avg_first:.0f})", flush=True)
            summary.append((lab, total, denom))

    print("\n=== SUMMARY ===")
    for lab, total, denom in summary:
        print(f"{lab:55s} {total}/{denom}")


if __name__ == "__main__":
    main()
