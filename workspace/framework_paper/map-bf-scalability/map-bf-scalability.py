"""Scenario runner for map-bf-scalability.

Runs the map-bf tools on the nested map-bf-scalability pool (fixed utilization,
task count growing across the matrix columns) through the raw harness, using
``harness`` for the run, and refreshes the processed Excel + figure after every
column.

The tools and their configuration mirror ``workspace/framework_paper/map-bf/bf.py``
(pd, hopa, gdpa-prio, the bounded multi-start gdpa-100/200/500 and the
vectorized brute force), all with a 1000 s budget per system and tool.

Artifacts go to ``map-bf-scalability-<u>/`` and carry the scenario name:
``map-bf-scalability-<u>_raw.json`` plus the processed Excel and figure
``map-bf-scalability-<u>_processed.xlsx/.png/.pdf``.

    .venv/bin/python workspace/framework_paper/map-bf-scalability/map-bf-scalability.py
    .venv/bin/python workspace/framework_paper/map-bf-scalability/map-bf-scalability.py -u 0.6 --threads 6
"""

import argparse
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "map-bf"))  # real map-bf tools

import bf  # noqa: E402
import harness  # noqa: E402
import process  # noqa: E402
import systems  # noqa: E402

# Defaults for the scenario (all overridable from the command line).
TIMEOUT = 1000.0
N_SYSTEMS = systems.N_SYSTEMS
UTILIZATION = systems.UTILIZATION
THREADS = 6
BF_BATCH_SIZE = 10000
# The over-utilization shortcut/prune is always on, for GDPA's cost and for bf.
BF_PRUNE = True
PRUNE_OVER_UTILIZED = True
# GDPA's cost uses the vectorized analysis (and its cache) instead of the scalar one.
VECTOR_COST = True
ORDER = ("pd", "hopa", "gdpa-prio", "gdpa-100", "gdpa-200", "gdpa-500", "bf")


def build_tools(prune_over_utilized=PRUNE_OVER_UTILIZED, vector_cost=VECTOR_COST,
                bf_batch_size=BF_BATCH_SIZE, bf_prune=BF_PRUNE):
    """Return ``(labels, funcs)`` mirroring the map-bf configuration."""
    gdpa = dict(prune_over_utilized=prune_over_utilized, vector_cost=vector_cost)
    tools = [
        ("pd", bf.pd_mapping_fp),
        ("hopa", bf.hopa_mapping_fp),
        ("gdpa-prio", partial(bf.gdpa_prio_fp, **gdpa)),
        ("gdpa-100", partial(bf.gdpa_ms_fp, chunk=25, restarts=4, **gdpa)),
        ("gdpa-200", partial(bf.gdpa_ms_fp, chunk=20, restarts=10, **gdpa)),
        ("gdpa-500", partial(bf.gdpa_ms_fp, chunk=50, restarts=10, **gdpa)),
        ("bf", partial(bf.bf_mapping, batch_size=bf_batch_size, prune=bf_prune)),
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
                                 utilization=args.utilization, verbose=True)
    columns = [str(size) for size in systems.SIZES]

    eval_name = f"map-bf-scalability-{args.utilization:g}"
    out = Path(args.output_dir) if args.output_dir else HERE / eval_name
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"{eval_name}_raw.json"
    excel_path = out / f"{eval_name}_processed.xlsx"

    def refresh(column, records, finished):
        counts, times = process.build_tables(records, labels, finished)
        process.write_outputs(counts, times, str(excel_path), xlabel="Tasks")
        print(f"    (excel+figure updated after column {column}: {finished})",
              flush=True)

    records = harness.evaluate(pool, labels, funcs, threads=args.threads,
                               columns=columns, timeouts=timeouts,
                               output=str(raw_path), on_column=refresh)

    counts, times = process.build_tables(records, labels, columns)
    process.write_outputs(counts, times, str(excel_path), xlabel="Tasks")
    print(f"\n=== schedulable (out of {args.systems}) ===")
    print(counts.to_string())
    print("\n=== mean time over schedulable (s) ===")
    print(times.round(3).to_string())
    print(f"\nwrote {out}/{eval_name}_raw.json, "
          f"{eval_name}_processed.xlsx/.png/.pdf")


if __name__ == "__main__":
    main()
