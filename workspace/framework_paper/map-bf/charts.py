"""Schedulability figure for the map-bf scenario: GDPA vs brute force.

Reads ``map-bf-9/map-bf-9_schedulables.xlsx`` and writes a two-line figure plus
the optimality gap (systems the brute force solves that GDPA does not).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
NAME = "map-bf-9"

STYLES = {
    "gdpa-100": {"color": "#0000FF", "marker": "^", "ls": "-"},
    "gdpa-200": {"color": "#FF8C00", "marker": "v", "ls": "-"},
    "bf": {"color": "#FF0000", "marker": "*", "ls": "--"},
}


def load():
    path = HERE / NAME / f"{NAME}_schedulables.xlsx"
    df = pd.read_excel(path, index_col=0)
    return df[[c for c in ("gdpa-100", "gdpa-200", "bf") if c in df.columns]]


def plot(df):
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    for col in df.columns:
        style = STYLES.get(col, {})
        df[col].plot.line(ax=ax, label=col, linewidth=1.5, markersize=5, **style)
    ax.set_xlabel("Average Utilization", fontweight="bold")
    ax.set_ylabel("Schedulable Systems", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", axis="x")
    ax.legend(loc="lower left")
    fig.savefig(HERE / f"{NAME}_schedulables.pdf")
    fig.savefig(HERE / f"{NAME}_schedulables.png")
    plt.close(fig)


def report_gap(df):
    print(f"=== optimality gap ({NAME}) ===")
    for col in df.columns:
        if col == "bf":
            continue
        gap = (df["bf"] - df[col]).clip(lower=0)
        total = int(df["bf"].sum())
        solved = int(df[col].sum())
        print(f"{col:9s}: {solved}/{total} schedulable, gap vs bf = {int(gap.sum())} "
              f"(max/level {int(gap.max())})")
    over = (df[[c for c in df.columns if c != 'bf']].max(axis=1) > df['bf'])
    if over.any():
        print("WARNING: GDPA exceeded bf at levels:",
              list(df.index[over]))


def main():
    df = load()
    plot(df)
    report_gap(df)


if __name__ == "__main__":
    main()
