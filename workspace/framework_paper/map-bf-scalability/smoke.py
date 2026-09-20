"""Smoke test for the map-bf scalability harness: pd and hopa at a fixed U.

Generates the nested pool at ``U=0.6`` and runs the real ``pd`` and ``hopa``
tools (imported from the sibling ``map-bf/bf.py``) through the raw harness,
then writes the processed Excel. It is a small end-to-end check, not the
scenario runner.

    .venv/bin/python workspace/framework_paper/map-bf-scalability/smoke.py [--systems N]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "map-bf"))  # real pd/hopa implementations

import bf  # noqa: E402
import harness  # noqa: E402
import process  # noqa: E402
import systems  # noqa: E402

UTILIZATION = 0.6
LABELS = ["pd", "hopa"]
FUNCS = [bf.pd_mapping_fp, bf.hopa_mapping_fp]
TIMEOUTS = {"pd": 30.0, "hopa": 120.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", type=int, default=systems.N_SYSTEMS,
                        help="number of base systems (default: all)")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    pool = systems.generate_pool(n_systems=args.systems,
                                 utilization=UTILIZATION, verbose=True)
    out = HERE / f"smoke-{UTILIZATION}"
    out.mkdir(exist_ok=True)
    column_labels = [str(size) for size in systems.SIZES]

    def refresh_excel(column, records, done_columns):
        schedulable, times, finished = process.build_tables(records, LABELS, done_columns)
        process.write_outputs(schedulable, times, finished,
                              str(out / "processed.xlsx"), xlabel="Tasks")
        print(f"    (excel+figure updated after column {column}: {done_columns})",
              flush=True)

    records = harness.evaluate(pool, LABELS, FUNCS, threads=args.threads,
                               columns=column_labels,
                               timeouts=TIMEOUTS,
                               output=str(out / "raw.json"),
                               on_column=refresh_excel)
    schedulable, times, finished = process.build_tables(records, LABELS, column_labels)

    print(f"\n=== schedulable (out of {args.systems}) ===")
    print(schedulable.to_string())
    print("\n=== mean time over schedulable (s) ===")
    print(times.round(3).to_string())
    print("\n=== finished ===")
    print(finished.to_string())
    print(f"\nwrote {out / 'raw.json'} and {out / 'processed.xlsx'}")


if __name__ == "__main__":
    main()
