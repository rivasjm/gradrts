"""Schedulability figure for the map-bf scenario: GDPA vs brute force.

Reads ``map-bf-<size>/map-bf-<size>_schedulables.xlsx`` and writes a figure plus
the optimality gap (systems the brute force solves that GDPA does not).

    python workspace/framework_paper/map-bf/charts.py --size 9
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent

STYLES = {
    "pd": {"color": "#8B4513", "marker": "s", "ls": ":"},
    "hopa": {"color": "#008000", "marker": "x", "ls": "--"},
    "gdpa-prio": {"color": "#9467BD", "marker": "D", "ls": "-"},
    "gdpa-100": {"color": "#0000FF", "marker": "^", "ls": "-"},
    "gdpa-200": {"color": "#FF8C00", "marker": "v", "ls": "-"},
    "gdpa-500": {"color": "#000000", "marker": "P", "ls": "-"},
    "bf": {"color": "#FF0000", "marker": "*", "ls": "--"},
    "bf-seq": {"color": "#E377C2", "marker": ".", "ls": ":"},
}

COLUMNS = ("pd", "hopa", "gdpa-prio", "gdpa-100", "gdpa-200", "gdpa-500", "bf", "bf-seq")


def name(size):
    return f"map-bf-{size}"


def load(size):
    n = name(size)
    path = HERE / n / f"{n}_schedulables.xlsx"
    df = pd.read_excel(path, index_col=0)
    return df[[c for c in COLUMNS if c in df.columns]]


def plot(df, size):
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    for col in df.columns:
        style = STYLES.get(col, {})
        df[col].plot.line(ax=ax, label=col, linewidth=1.5, markersize=5, **style)
    ax.set_xlabel("Average Utilization", fontweight="bold")
    ax.set_ylabel("Schedulable Systems", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", axis="x")
    ax.legend(loc="lower left")
    fig.savefig(HERE / f"{name(size)}_schedulables.pdf")
    fig.savefig(HERE / f"{name(size)}_schedulables.png")
    plt.close(fig)


def report_gap(df, size):
    print(f"=== optimality gap ({name(size)}) ===")
    for col in df.columns:
        if col in ("bf", "bf-seq"):
            continue
        gap = (df["bf"] - df[col]).clip(lower=0)
        total = int(df["bf"].sum())
        solved = int(df[col].sum())
        print(f"{col:9s}: {solved}/{total} schedulable, gap vs bf = {int(gap.sum())} "
              f"(max/level {int(gap.max())})")
    over = (df[[c for c in df.columns if c not in ("bf", "bf-seq")]].max(axis=1) > df["bf"])
    if over.any():
        print("WARNING: GDPA exceeded bf at levels:", list(df.index[over]))


def main(size):
    df = load(size)
    plot(df, size)
    report_gap(df, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=9, choices=(9, 10),
                        help="total tasks / scenario size (default: 9)")
    args = parser.parse_args()
    main(args.size)
