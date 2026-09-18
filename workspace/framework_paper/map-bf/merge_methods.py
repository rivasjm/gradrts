"""Merge method columns from partial map-bf runs into the main results.

A run of ``bf.py`` restricted with ``--methods`` writes a full ``map-bf-<size>/``
directory containing only those methods. Since every metric (schedulables,
times, times_success) is computed per method, those columns can simply be
combined with the existing results, as if all methods had been run together.
A column provided by a source run replaces the one in the target.

Usage (from ``code/``):

    python workspace/framework_paper/map-bf/merge_methods.py \
        --size 10 \
        --target workspace/framework_paper/map-bf/map-bf-10 \
        --source /tmp/gms/map-bf-10
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from examples.evaluation import SchedRatioEval

SUFFIXES = ("schedulables", "times", "times_success")
ORDER = ("pd", "hopa", "gdpa-prio", "gdpa-100", "gdpa-200", "gdpa-500",
         "bf", "bf-seq")


def merge(target_file, source_files, order):
    sources = [pd.read_excel(f, index_col=0) for f in source_files if Path(f).exists()]
    source_cols = set()
    for frame in sources:
        source_cols |= set(frame.columns)
    target = pd.read_excel(target_file, index_col=0)
    # a column provided by a source run replaces the one in the target
    target = target[[c for c in target.columns if c not in source_cols]]
    df = pd.concat([target] + sources, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    cols = [c for c in order if c in df.columns]
    return df[cols]


def regenerate_diagnostics(target, name, labels, n_systems):
    """Rebuild the in-directory PNGs from the merged spreadsheets."""
    sched = pd.read_excel(target / f"{name}_schedulables.xlsx", index_col=0)[list(labels)]
    succ = pd.read_excel(target / f"{name}_times_success.xlsx", index_col=0)[list(labels)]
    results = sched.to_numpy(dtype=float)
    success_sums = np.nan_to_num(succ.to_numpy(dtype=float)) * results

    runner = SchedRatioEval(name, labels=list(labels), funcs=[None] * len(labels),
                            systems=[None] * n_systems,
                            utilizations=sched.index.to_numpy(), threads=1,
                            output_dir=str(target))
    runner.start = time.time()
    runner._save(results, "schedulables")
    runner._efficiency_chart(results, success_sums)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=9, choices=(9, 10),
                        help="scenario size (default: 9)")
    parser.add_argument("--target", required=True, help="directory with the full results")
    parser.add_argument("--source", nargs="+", required=True,
                        help="directory(ies) with partial runs to merge in")
    parser.add_argument("--systems", type=int, default=25,
                        help="number of systems (for the efficiency chart; default: 25)")
    parser.add_argument("--no-diagnostics", action="store_true",
                        help="do not regenerate the in-directory PNGs")
    args = parser.parse_args()

    name = f"map-bf-{args.size}"
    target = Path(args.target)
    for suffix in SUFFIXES:
        target_file = target / f"{name}_{suffix}.xlsx"
        if not target_file.exists():
            raise SystemExit(f"missing {target_file}")
        source_files = [Path(s) / f"{name}_{suffix}.xlsx" for s in args.source]
        df = merge(target_file, source_files, ORDER)
        df.to_excel(target_file)
        print(f"{target_file}: {list(df.columns)}")

    if not args.no_diagnostics:
        labels = tuple(pd.read_excel(target / f"{name}_schedulables.xlsx",
                                     index_col=0).columns)
        regenerate_diagnostics(target, name, labels, args.systems)
        print(f"{target}: regenerated diagnostics (schedulables, efficiency)")


if __name__ == "__main__":
    main()
