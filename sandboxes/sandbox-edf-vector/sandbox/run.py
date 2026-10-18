#!/usr/bin/env python3
"""Check the vectorized local-EDF analysis in ``algorithms/``.

Usage (from the sandbox root)::

    python run.py                    # uses algorithms/analyze.py (the entry point)
    python run.py --instances 25 --k 100
    python run.py --levels 8         # spread over fewer utilization levels
    python run.py --module myimpl    # use algorithms/myimpl.py instead
    python run.py --reference        # only print the scalar reference timing

The candidate must expose a module in ``algorithms/`` with an
``analyze(system, deadlines)`` function:

    analyze(system, deadlines) -> wcrts

where ``deadlines`` is a list/array of shape ``(K, n_tasks)`` (K local-deadline
assignments) and the return value is the ``(K, n_tasks)`` worst-case response
times for the same tasks on the system, in ``system.tasks`` order. ``run.py``
checks that the values match the scalar reference analysis (equivalence) and
reports the speedup over calling that scalar analysis K times.

This file is part of the fixed evaluation; do not edit it.
"""

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness, pool  # noqa: E402

DEFAULT_MODULE = "algorithms.analyze"


def load_analyze(module_name):
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import algorithms  # noqa: F401  (ensures the package is importable)
    module = importlib.import_module(module_name)
    if not hasattr(module, "analyze"):
        sys.exit(f"module {module_name!r} has no 'analyze' function")
    return module.analyze


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", default=DEFAULT_MODULE,
                        help=f"module to load (default: {DEFAULT_MODULE})")
    parser.add_argument("--instances", type=int, default=25,
                        help="number of instances to check (default: 25)")
    parser.add_argument("--levels", type=int, default=None,
                        help="spread the instances over this many utilization levels")
    parser.add_argument("--k", type=int, default=100,
                        help="assignments per instance (default: 100)")
    parser.add_argument("--rtol", type=float, default=1e-6,
                        help="relative tolerance for equivalence (default: 1e-6)")
    parser.add_argument("--atol", type=float, default=1e-6,
                        help="absolute tolerance for equivalence (default: 1e-6)")
    parser.add_argument("--reference", action="store_true",
                        help="only time the scalar reference and exit")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="do not print per-instance lines")
    args = parser.parse_args()

    if args.reference:
        instances = harness.build_instances(population=args.instances,
                                            levels=args.levels, k=args.k)
        harness.reference_wcrts(instances, collect_time=True)
        total = sum(i["scalar_time"] for i in instances)
        print(f"scalar reference: {args.instances} instances x {args.k} assignments "
              f"in {total:.2f}s")
        return

    analyze = load_analyze(args.module)
    print(f"checking {args.module} on {args.instances} instances x {args.k} "
          f"assignments (tolerance rtol={args.rtol:g}, atol={args.atol:g})...")
    report = harness.evaluate(analyze, population=args.instances, levels=args.levels,
                              k=args.k, rtol=args.rtol, atol=args.atol,
                              verbose=not args.quiet)

    print()
    print("=== summary ===")
    print(f"  equivalence : {'OK' if report['ok'] else 'FAIL'} "
          f"({report['mismatches']} mismatches out of {report['values']} values, "
          f"max rel error {report['max_rel']:.2e})")
    print(f"  scalar time : {report['scalar_time']:.3f} s "
          f"({report['instances']} x {args.k} scalar calls)")
    print(f"  candidate   : {report['candidate_time']:.3f} s")
    speedup = report['speedup']
    print(f"  speedup     : {speedup:.1f}x" if speedup == speedup else "  speedup     : n/a")

    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
