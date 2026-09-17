"""Merge method columns from partial map-bf runs into the main results.

A run of ``bf.py`` restricted with ``--methods`` writes a full ``map-bf-9/``
directory containing only those methods. Since every metric (schedulables,
times, times_success) is computed per method, those columns can simply be
combined with the existing results, as if all methods had been run together.

Usage (from ``code/``):

    python workspace/framework_paper/map-bf/merge_methods.py \
        --target workspace/framework_paper/map-bf/map-bf-9 \
        --source /tmp/gprio/map-bf-9
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from examples.evaluation import SchedRatioEval

NAME = "map-bf-9"
SUFFIXES = ("schedulables", "times", "times_success")
ORDER = ("pd", "hopa", "gdpa-prio", "gdpa-100", "gdpa-200", "bf")


def merge(target_file, source_files, order):
    frames = [pd.read_excel(target_file, index_col=0)]
    frames += [pd.read_excel(f, index_col=0) for f in source_files if Path(f).exists()]
    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    cols = [c for c in order if c in df.columns]
    cols += [c for c in df.columns if c not in order]
    return df[cols]


def regenerate_diagnostics(target, labels, n_systems):
    """Rebuild the in-directory PNGs from the merged spreadsheets."""
    sched = pd.read_excel(target / f"{NAME}_schedulables.xlsx", index_col=0)[list(labels)]
    succ = pd.read_excel(target / f"{NAME}_times_success.xlsx", index_col=0)[list(labels)]
    results = sched.to_numpy(dtype=float)
    success_sums = np.nan_to_num(succ.to_numpy(dtype=float)) * results

    runner = SchedRatioEval(NAME, labels=list(labels), funcs=[None] * len(labels),
                            systems=[None] * n_systems,
                            utilizations=sched.index.to_numpy(), threads=1,
                            output_dir=str(target))
    runner.start = time.time()
    runner._save(results, "schedulables")
    runner._efficiency_chart(results, success_sums)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, help="directory with the full results")
    parser.add_argument("--source", nargs="+", required=True,
                        help="directory(ies) with partial runs to merge in")
    parser.add_argument("--systems", type=int, default=25,
                        help="number of systems (for the efficiency chart; default: 25)")
    parser.add_argument("--no-diagnostics", action="store_true",
                        help="do not regenerate the in-directory PNGs")
    args = parser.parse_args()

    target = Path(args.target)
    for suffix in SUFFIXES:
        target_file = target / f"{NAME}_{suffix}.xlsx"
        if not target_file.exists():
            raise SystemExit(f"missing {target_file}")
        source_files = [Path(s) / f"{NAME}_{suffix}.xlsx" for s in args.source]
        df = merge(target_file, source_files, ORDER)
        df.to_excel(target_file)
        print(f"{target_file}: {list(df.columns)}")

    if not args.no_diagnostics:
        labels = tuple(pd.read_excel(target / f"{NAME}_schedulables.xlsx",
                                     index_col=0).columns)
        regenerate_diagnostics(target, labels, args.systems)
        print(f"{target}: regenerated diagnostics (schedulables, efficiency)")


if __name__ == "__main__":
    main()
