"""Tuning harness for the EDF local-deadline GDPA (paper evaluation, size 25).

The size-25 EDF scenario is where GDPA was found to underperform HOPA. This
script sweeps deadline encodings (ParameterHandler ``extract``/``insert``) and
optimizer hyperparameters against the HOPA and PD baselines, at the low/medium
utilizations where the difference is visible (high utilizations are slow and
are avoided on purpose).

Each configuration is run in parallel over a population of systems and the
number of schedulable ones is printed. Results are printed line by line so the
run can be followed and resumed/curtailed.

Usage (from ``code/``, with the venv active):

    python workspace/edf_tuning/edf_tune.py --phase handlers --u 0.66 0.68 0.70
    python workspace/edf_tuning/edf_tune.py --phase update  --u 0.68 --n 50
    python workspace/edf_tuning/edf_tune.py --list

or via the launcher: ``bash workspace/edf_tuning/run_tuning.sh``.
"""

import argparse
import copy
import math
import time
from multiprocessing import Pool

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment
from assignment.hopa_assignment import HOPAssignment
from examples.generator import set_utilization, to_edf
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import SequentialGradientFunction
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.interfaces import ParameterHandler
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import Adam, NoisyAdam
from model.linear_system import LinearSystem
from workspace.framework_paper.systems import SIZES, get_systems

_BASE = None
_SIZE = None


# --------------------------------------------------------------------------
# Deadline encodings: extract maps local deadlines to the optimizer's vector,
# insert maps it back. A configuration is only well posed when the two are
# (approximately) inverses, otherwise the PD initialization is destroyed.
# --------------------------------------------------------------------------

class LinearHandler(ParameterHandler):
    """d -> d/max_d, clamped back to [0, max_d]."""

    def extract(self, system: LinearSystem):
        max_d = max(f.deadline for f in system.flows)
        return [t.deadline / max_d for t in system.tasks]

    def insert(self, system: LinearSystem, x):
        max_d = max(f.deadline for f in system.flows)
        for v, t in zip(x, system.tasks):
            t.deadline = min(max(v, 0.0), 1.0) * max_d


class PerFlowHandler(ParameterHandler):
    """Normalize each local deadline by its own flow deadline."""

    def extract(self, system: LinearSystem):
        return [min(max(t.deadline / t.flow.deadline, 0.0), 1.0) for t in system.tasks]

    def insert(self, system: LinearSystem, x):
        for v, t in zip(x, system.tasks):
            t.deadline = min(max(v, 0.0), 1.0) * t.flow.deadline


class LogHandler(ParameterHandler):
    """Logarithmic normalization by the largest flow deadline."""

    def extract(self, system: LinearSystem):
        max_d = max(f.deadline for f in system.flows)
        return [math.log(1 + t.deadline) / math.log(1 + max_d) for t in system.tasks]

    def insert(self, system: LinearSystem, x):
        max_d = max(f.deadline for f in system.flows)
        for v, t in zip(x, system.tasks):
            t.deadline = (1 + max_d) ** min(max(v, 0.0), 1.0) - 1


class LogitHandler(ParameterHandler):
    """Consistent sigmoid pair: x = logit(d/max_d), d = sigmoid(x)*max_d."""

    def extract(self, system: LinearSystem):
        max_d = max(f.deadline for f in system.flows)
        eps = 1e-6
        out = []
        for t in system.tasks:
            p = min(max(t.deadline / max_d, eps), 1 - eps)
            out.append(math.log(p / (1 - p)))
        return out

    def insert(self, system: LinearSystem, x):
        max_d = max(f.deadline for f in system.flows)
        for v, t in zip(x, system.tasks):
            t.deadline = max_d / (1 + math.exp(-v))


class SigmoidHandler(ParameterHandler):
    """Legacy encoding (sigmoid extract + linear insert); kept for reference.

    It is NOT a round trip, so the PD initialization is distorted (deadlines
    are inflated), which severely degrades GDPA on size 25."""

    def extract(self, system: LinearSystem):
        max_d = max(f.deadline for f in system.flows)
        return [1 / (1 + math.exp(-t.deadline / max_d)) for t in system.tasks]

    def insert(self, system: LinearSystem, x):
        max_d = max(f.deadline for f in system.flows)
        for v, t in zip(x, system.tasks):
            t.deadline = v * max_d


HANDLERS = {
    "linear": LinearHandler,
    "perflow": PerFlowHandler,
    "log": LogHandler,
    "logit": LogitHandler,
    "sigmoid": SigmoidHandler,
}

BASE = {"handler": "linear", "limit": 100, "lf": 10,
        "noise": True, "lr": 3.0, "gamma": 0.9, "seed": 1, "sigma": 1.5}


# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------

def make_handler(kind):
    return HANDLERS[kind]()


def gdpa_run(system, cfg):
    analysis = HolisticLocalEDFAnalysis(limit_factor=cfg.get("lf", 10), reset=False)
    handler = make_handler(cfg["handler"])
    cost = InvslackCost(parameter_handler=handler, analysis=analysis)
    stop = ThresholdStopFunction(limit=cfg["limit"], patience=cfg.get("patience"))
    gradient = SequentialGradientFunction(cost_function=cost, sigma=cfg.get("sigma", 1.5))
    if cfg.get("noise", True):
        update = NoisyAdam(lr=cfg.get("lr", 3.0), gamma=cfg.get("gamma", 0.9),
                           seed=cfg.get("seed", 1))
    else:
        update = Adam(lr=cfg.get("lr", 3.0))
    optimizer = GradientDescentOptimizer(parameter_handler=handler, cost_function=cost,
                                         stop_function=stop, gradient_function=gradient,
                                         update_function=update, verbose=False)
    PDAssignment().apply(system)
    optimizer.apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def label(cfg):
    parts = [cfg["handler"], f"lim={cfg['limit']}", f"lf={cfg.get('lf', 10)}"]
    if cfg.get("noise", True):
        parts += [f"lr={cfg.get('lr', 3.0)}", f"g={cfg.get('gamma', 0.9)}",
                  f"seed={cfg.get('seed', 1)}"]
    else:
        parts += ["adam", f"lr={cfg.get('lr', 3.0)}"]
    if cfg.get("sigma", 1.5) != 1.5:
        parts.append(f"sigma={cfg['sigma']}")
    if cfg.get("patience"):
        parts.append(f"pat={cfg['patience']}")
    return ",".join(parts)


# --------------------------------------------------------------------------
# Parallel evaluation
# --------------------------------------------------------------------------

def _init_worker(size):
    global _BASE, _SIZE
    _SIZE = size
    systems = get_systems(size)
    for s in systems:
        to_edf(s)
    _BASE = systems


def _worker(args):
    idx, u, cfg = args
    system = copy.deepcopy(_BASE[idx])
    set_utilization(system, u)
    return idx, gdpa_run(system, cfg)


def _pd(system):
    PDAssignment().apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def _hopa(system):
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    HOPAssignment(analysis=analysis).apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def build_phase(phase):
    if phase == "candidate":
        return [dict(BASE)]
    if phase == "handlers":
        return [dict(BASE, handler=h) for h in HANDLERS]
    if phase == "update":
        cfgs = []
        for handler in ("perflow", "linear"):
            base = dict(BASE, handler=handler)
            cfgs += [
                dict(base, lr=1.0),
                dict(base, lr=5.0),
                dict(base, lr=10.0),
                dict(base, gamma=0.99),
                dict(base, noise=False),
                dict(base, noise=False, lr=1.0),
                dict(base, seed=2),
            ]
        return cfgs
    if phase == "gradient":
        return [dict(BASE, handler="perflow", sigma=s) for s in (0.1, 0.5, 1.0, 3.0, 5.0)]
    if phase == "stop":
        return [
            dict(BASE, handler="perflow", limit=200),
            dict(BASE, handler="perflow", limit=100, patience=30),
            dict(BASE, handler="linear", limit=200),
        ]
    raise SystemExit(f"unknown phase {phase!r} (use --list)")


PHASES = ("baseline", "candidate", "handlers", "update", "gradient", "stop")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", action="append", default=None,
                        help="phase to run; repeatable (default: candidate). "
                             "See --list.")
    parser.add_argument("-u", "--utilizations", type=float, nargs="+",
                        default=[0.66, 0.68, 0.70],
                        help="utilization levels to sweep (default: 0.66 0.68 0.70)")
    parser.add_argument("--n", type=int, default=50,
                        help="number of systems from the population (default: 50)")
    parser.add_argument("--idx", type=int, nargs="+", default=None,
                        help="explicit system indices to evaluate instead of 0..n-1")
    parser.add_argument("--size", type=int, default=25, choices=sorted(SIZES),
                        help="system size (default: 25)")
    parser.add_argument("--threads", type=int, default=6,
                        help="worker processes (default: 6)")
    parser.add_argument("--list", action="store_true", help="list phases and exit")
    args = parser.parse_args()

    if args.list:
        print("phases:")
        for name in PHASES:
            print(f"  {name}")
        print("\nhandlers:", ", ".join(HANDLERS))
        return

    global _SIZE
    _SIZE = args.size
    indices = args.idx if args.idx else list(range(args.n))
    phases = args.phase if args.phase else ["candidate"]
    unknown = [p for p in phases if p != "baseline" and p not in PHASES]
    if unknown:
        raise SystemExit(f"unknown phase(s): {unknown}")

    _init_worker(args.size)
    summary = []
    with Pool(args.threads, initializer=_init_worker, initargs=(args.size,)) as pool:
        for u in args.utilizations:
            systems = [copy.deepcopy(_BASE[i]) for i in indices]
            for s in systems:
                set_utilization(s, u)
            pd = sum(_pd(s) for s in systems)
            hopa = sum(_hopa(s) for s in systems)
            print(f"=== u={u} (n={len(indices)}): pd={pd} hopa={hopa} ===", flush=True)
            summary.append((u, "pd", pd, len(indices)))
            summary.append((u, "hopa", hopa, len(indices)))
            for phase in phases:
                if phase == "baseline":
                    continue
                print(f"-- phase: {phase}", flush=True)
                for cfg in build_phase(phase):
                    t0 = time.perf_counter()
                    results = pool.map(_worker, [(i, u, cfg) for i in indices])
                    count = sum(int(ok) for _, ok in results)
                    dt = time.perf_counter() - t0
                    lab = label(cfg)
                    print(f"  {lab:60s} {count}/{len(indices)} ({dt:.1f}s)", flush=True)
                    summary.append((u, lab, count, len(indices)))

    print("\n=== SUMMARY ===")
    for u, lab, count, total in summary:
        print(f"u={u} {lab:60s} {count}/{total}")


if __name__ == "__main__":
    main()
