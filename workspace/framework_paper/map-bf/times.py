"""Mean time-to-schedulable-solution figure for the map-bf scenario (log scale).

Mirrors the other scenarios' ``times.py``: reads
``map-bf-<size>/map-bf-<size>_times_success.xlsx`` and writes
``map-bf-<size>_times.pdf|png``.

    python workspace/framework_paper/map-bf/times.py --size 9
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from charts import COLUMNS, HERE, STYLES, name


def load(size):
    n = name(size)
    path = HERE / n / f"{n}_times_success.xlsx"
    df = pd.read_excel(path, index_col=0)
    return df[[c for c in COLUMNS if c in df.columns]]


def main(size):
    df = load(size)
    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    for col in df.columns:
        style = STYLES.get(col, {})
        df[col].plot.line(ax=ax, label=col, linewidth=1.5, markersize=5, **style)
    ax.set_xlabel("Average Utilization", fontweight="bold")
    ax.set_ylabel("Mean Time to Schedulable Solution (s)", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="x")
    ax.legend(loc="upper left", ncol=2, prop={"size": 8})
    fig.savefig(HERE / f"{name(size)}_times.pdf")
    fig.savefig(HERE / f"{name(size)}_times.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=9, choices=(9, 10, 12),
                        help="total tasks / scenario size (default: 9)")
    args = parser.parse_args()
    main(args.size)
