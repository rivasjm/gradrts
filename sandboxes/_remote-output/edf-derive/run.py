#!/usr/bin/env python3
"""Evaluate every registered deadline-assignment technique on the example pool.

    python run.py                 # 50 systems x 20 utilization levels
    python run.py --per-level     # also break the counts down by utilization
    python run.py --systems 10 --levels 6 --quiet
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness, pool  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=15, choices=sorted(pool.SIZES),
                        help="pool size in tasks (default: 15)")
    parser.add_argument("--systems", type=int, default=pool.POPULATION,
                        help=f"systems per utilization level (default: {pool.POPULATION})")
    parser.add_argument("--levels", type=int, default=None,
                        help="number of utilization levels (default: all)")
    parser.add_argument("--seed", type=int, default=pool.SEED,
                        help=f"pool seed (default: {pool.SEED})")
    parser.add_argument("--per-level", action="store_true",
                        help="also print the schedulable count per utilization level")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="do not print per-level progress")
    args = parser.parse_args()

    utilizations = list(pool.UTILIZATIONS)
    if args.levels:
        utilizations = utilizations[:args.levels]

    report = harness.evaluate(seed=args.seed, size=args.size, population=args.systems,
                              utilizations=utilizations, verbose=not args.quiet)

    instances = report["instances"]
    print()
    print(f"=== size={args.size}  seed={args.seed}  systems={args.systems}  "
          f"levels={len(utilizations)}  instances={instances} ===")
    print(f"  {'technique':26s} {'schedulable':>11s} {'ratio':>7s} {'mean t':>9s} {'total t':>9s}")
    print("  " + "-" * 64)
    for name, m in report["techniques"].items():
        ratio = m["schedulable"] / instances if instances else 0.0
        print(f"  {name:26s} {m['schedulable']:11d} {ratio:7.1%} "
              f"{m['mean_time']:9.5f} {m['total_time']:9.4f}")
        for error in m["errors"]:
            print(f"      error: {error}")

    if args.per_level:
        names = list(report["techniques"])
        print()
        print("  per utilization level:")
        print(f"  {'u':>7s}" + "".join(f"{n[:12]:>14s}" for n in names))
        for i, u in enumerate(utilizations):
            line = f"  {u:7.3f}"
            for name in names:
                line += f"{report['techniques'][name]['per_level'][i]:14d}"
            print(line)


if __name__ == "__main__":
    main()
