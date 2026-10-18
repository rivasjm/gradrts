#!/usr/bin/env python3
"""Evaluate the fixed-priority assignment algorithms in ``algorithms/``.

Usage (from the sandbox root)::

    python run.py                       # the 15 pool, every registered algorithm
    python run.py --pool 25             # the larger 25-task pool
    python run.py --pool both           # both pools
    python run.py --workers 8           # evaluate instances over 8 processes
    python run.py --systems 10 --levels 5      # fast smoke run
    python run.py --per-level           # also print the per-utilization table
    python run.py --list                # list the algorithms found
    python run.py --leaderboard         # rebuild/print the leaderboard only

While a sweep runs, results and a chart are refreshed as each utilization
level completes: raw per-instance records go to ``results/<pool>/raw.json`` and
the figure (schedulable systems on top, mean time below) to
``results/<pool>/pool.png``.

Every evaluated method is written to ``results/<pool>/<method>.json`` with the
same record schema as the pool's reference (``brute-force`` for 15, ``gdpa`` for
25), and a per-pool leaderboard ``results/<pool>/leaderboard.csv`` is derived
from the records (never written by hand).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import algorithms  # noqa: E402
from frozen import chart, harness, pool, results  # noqa: E402
from frozen.baselines import pd_assign, random_assign  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")
IDENTITY = os.path.join(HERE, "IDENTITY.md")


def _pd(system):
    pd_assign(system)


def _random(system):
    random_assign(system)


def read_identity():
    """Read the key/value lines of IDENTITY.md (if present)."""
    if not os.path.exists(IDENTITY):
        return None
    info = {}
    with open(IDENTITY) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("-") and ":" in line:
                key, _, value = line[1:].partition(":")
                key = key.strip().lower()
                if key in ("provider", "model", "started", "focus"):
                    info[key] = value.strip()
    return info or None


def collect_methods(args):
    funcs = {}
    if not args.no_baselines:
        funcs["pd"] = _pd
        funcs["random"] = _random
    funcs.update(algorithms.load())
    if args.methods:
        unknown = [m for m in args.methods if m not in funcs]
        if unknown:
            print(f"unknown methods: {unknown}\nknown: {sorted(funcs)}", file=sys.stderr)
            sys.exit(2)
        funcs = {k: v for k, v in funcs.items() if k in args.methods}
    return funcs


def _rows_by_summary_key(rows):
    groups = {}
    for row in rows:
        key = (row["pool"], row["seed"], row["size"], row["population"], row["n_levels"])
        groups.setdefault(key, []).append(row)
    return groups


def _fmt_time(value):
    return "        -" if value is None else f"{value:9.4f}"


def print_groups(rows, keys=None, per_level=False):
    groups = _rows_by_summary_key(rows)
    for key, group in groups.items():
        if keys is not None and key not in keys:
            continue
        pool_label, seed, size, population, n_levels = key
        instances = population * n_levels
        print(f"=== pool: {pool_label}  (seed={seed}, systems={population}, "
              f"levels={n_levels}, instances={instances}) ===")

        reference_method = pool.REFERENCE_METHOD.get(pool_label)
        reference = group[0]["reference"]
        has_reference_row = (reference_method is not None
                             and any(r["method"] == reference_method for r in group))

        header = (f"  {'method':26s} {'sched':>6s} {'ratio':>7s} {'mean t':>9s} "
                  f"{'total t':>9s} {'timeouts':>9s}")
        if reference is not None:
            header += f" {'gap':>6s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for row in group:
            line = (f"  {row['method']:26s} {row['schedulable']:6d} {row['ratio']:7.1%} "
                    f"{_fmt_time(row['mean_time'])} {_fmt_time(row['total_time'])} "
                    f"{row['timeouts']:9d}")
            if reference is not None:
                line += f" {row['gap']:6d}"
            print(line)

        if reference is not None and not has_reference_row:
            ratio = reference / instances if instances else 0.0
            print("  " + "-" * (len(header) - 2))
            print(f"  {reference_method:26s} {reference:6d} {ratio:7.1%} "
                  f"{_fmt_time(None)} {_fmt_time(None)} {'-':>9s} {0:6d}")

        if reference_method is not None and reference_method != pool.BRUTE_FORCE_METHOD:
            print(f"  note: reference = {reference_method} (a strong heuristic, not a "
                  f"proven optimum; beating it is a win)")

        if per_level:
            print()
            print("  per utilization level:")
            methods = [r for r in group
                       if reference_method is None or r["method"] != reference_method]
            col = "  " + f"{'u':>7s}" + "".join(f"{r['method'][:10]:>12s}" for r in methods)
            if reference is not None:
                col += f"{'REF':>7s}"
            print(col)
            utilizations = group[0]["utilizations"]
            ref_levels = group[0]["reference_per_level"]
            for i in range(n_levels):
                line = f"  {utilizations[i]:7.3f}"
                for r in methods:
                    line += f"{r['per_level'][i]:12d}"
                if ref_levels is not None:
                    line += f"{ref_levels[i]:7d}"
                print(line)
        print()


def _partial_records(label, size, population, utilizations, finished_levels,
                     raw, identity, names):
    """Records for the levels completed so far (used to refresh mid-sweep)."""
    methods = harness.aggregate(raw, names, population, utilizations)
    utils = [utilizations[i] for i in finished_levels]
    n_fin = len(finished_levels)
    records = []
    for name in names:
        per_level = [methods[name]["per_level"][i] for i in finished_levels]
        total_time = methods[name]["total_time"]
        records.append(results.make_record(
            name, label, pool.SEED, size, population, utils, per_level,
            mean_time=total_time / (population * n_fin) if n_fin else 0.0,
            total_time=total_time, timeouts=methods[name]["timeouts"],
            errors=methods[name]["errors"], identity=identity,
            per_level_time=[methods[name]["per_level_time"][i] for i in finished_levels],
            per_level_finished=[methods[name]["per_level_finished"][i] for i in finished_levels]))
    return records


def _pool_records(label, size, population, n_levels):
    """Records of every method available for this pool (references included)."""
    key = (label, pool.SEED, size, population, n_levels)
    return [r for r in results.load_all(RESULTS_DIR)
            if (r.get("pool"), r.get("seed"), r.get("size"),
                r.get("population"), r.get("n_levels")) == key]


def _write_charts(label, size, population, utilizations, finished_levels, methods,
                  pool_dir):
    """Draw both charts, one line per method that has a record for this pool.

    ``methods`` holds the live per-level aggregates of the methods evaluated in
    this run; the remaining lines come from the stored records (e.g. the
    references).
    """
    utils = [utilizations[i] for i in finished_levels]
    n_fin = len(finished_levels)
    title = f"pool {label} (seed {pool.SEED})"

    schedulable, times = {}, {}
    for name, m in methods.items():
        schedulable[name] = [m["per_level"][i] for i in finished_levels]
        times[name] = [m["per_level_time"][i] for i in finished_levels]
    for record in _pool_records(label, size, population, len(utilizations)):
        name = record.get("method")
        if name in schedulable:
            continue
        schedulable[name] = list(record.get("per_level", []))[:n_fin]
        times[name] = list(record.get("per_level_time", []))[:n_fin]

    try:
        chart.write_pool_chart(os.path.join(pool_dir, "pool.png"), title, utils,
                               schedulable, times)
    except Exception as exc:
        print(f"warning: could not draw the chart: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", choices=[str(s) for s in pool.POOLS] + ["both"],
                        default=str(pool.POOLS[0]))
    parser.add_argument("--seed", type=int, default=None,
                        help="override the pool seed (no precomputed reference)")
    parser.add_argument("--systems", type=int, default=None,
                        help="use only the first N systems of the pool")
    parser.add_argument("--levels", type=int, default=None,
                        help="use only the first N utilization levels")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="subset of method names to run")
    parser.add_argument("--workers", type=int, default=1,
                        help="worker processes for the sweep (1 = sequential)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="per-instance time budget in seconds (0 disables)")
    parser.add_argument("--no-baselines", action="store_true",
                        help="do not run pd / random")
    parser.add_argument("--no-raw", action="store_true",
                        help="do not write the incremental results/<pool>/raw.json")
    parser.add_argument("--no-plot", action="store_true",
                        help="do not draw the charts")
    parser.add_argument("--per-level", action="store_true",
                        help="also print the per-utilization breakdown")
    parser.add_argument("--no-save", action="store_true",
                        help="do not write anything (records, leaderboard, raw, chart)")
    parser.add_argument("--leaderboard", action="store_true",
                        help="rebuild and print the leaderboard from existing records, then exit")
    parser.add_argument("--list", action="store_true",
                        help="list registered algorithms and exit")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="do not print per-level progress")
    args = parser.parse_args()

    if args.list:
        for name in sorted(algorithms.load()):
            print(name)
        return

    if args.leaderboard:
        rows = results.leaderboard_rows(results.load_all(RESULTS_DIR))
        print_groups(rows, keys=None, per_level=args.per_level)
        for path in results.write_leaderboards(rows, RESULTS_DIR):
            print(f"leaderboard: {os.path.relpath(path, HERE)}")
        return

    funcs = collect_methods(args)
    if not funcs:
        print("no algorithms registered in algorithms/", file=sys.stderr)
        sys.exit(1)

    n_systems = args.systems or pool.POPULATION
    level_indices = list(range(args.levels)) if args.levels else list(range(len(pool.UTILIZATIONS)))
    utilizations = [pool.UTILIZATIONS[i] for i in level_indices]

    seed = args.seed if args.seed is not None else pool.SEED
    pools = [(str(size), size) for size in pool.POOLS
             if args.pool in (str(size), "both")]

    identity = read_identity()
    names = list(funcs)
    for label, size in pools:
        pool_dir = os.path.join(RESULTS_DIR, label)
        raw_path = None if (args.no_raw or args.no_save) else os.path.join(pool_dir, "raw.json")

        def on_level(raw, finished, label=label, size=size):
            for record in _partial_records(label, size, n_systems, utilizations,
                                           finished, raw, identity, names):
                results.save(RESULTS_DIR, record)
            rows = results.leaderboard_rows(results.load_all(RESULTS_DIR))
            results.write_leaderboards(rows, RESULTS_DIR)
            if not args.no_plot:
                methods = harness.aggregate(raw, names, n_systems, utilizations)
                _write_charts(label, size, n_systems, utilizations, finished,
                              methods, pool_dir)

        if not args.no_save and not args.quiet:
            print(f"running {label}-task pool (seed={seed}) on {len(funcs)} methods x "
                  f"{n_systems} systems x {len(level_indices)} levels, "
                  f"workers={args.workers} (timeout={args.timeout}s)...", flush=True)

        report = harness.evaluate(funcs, seed=seed, size=size, population=n_systems,
                                  utilizations=utilizations, timeout=args.timeout,
                                  verbose=not args.quiet, workers=args.workers,
                                  raw_path=raw_path,
                                  on_level=on_level if not args.no_save else None)
        records = results.records_from_report(report, label, identity=identity)
        if not args.no_save:
            for record in records:
                results.save(RESULTS_DIR, record)
        keys = {(r["pool"], r["seed"], r["size"], r["population"], r["n_levels"])
                for r in records}
        disk = results.load_all(RESULTS_DIR)
        new_slots = {(r["pool"], r["method"]) for r in records}
        all_records = [r for r in disk if (r["pool"], r["method"]) not in new_slots] + records
        rows = results.leaderboard_rows(all_records)
        print_groups(rows, keys=keys, per_level=args.per_level)
        if not args.no_save and not args.no_plot:
            _write_charts(label, size, n_systems, utilizations, level_indices,
                          report["methods"], pool_dir)

    if not args.no_save and pools:
        rows = results.leaderboard_rows(results.load_all(RESULTS_DIR))
        for path in results.write_leaderboards(rows, RESULTS_DIR):
            print(f"leaderboard: {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
