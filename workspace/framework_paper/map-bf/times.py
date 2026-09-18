"""Mean time-to-schedulable-solution figure for the map-bf scenario (log scale).

Mirrors the style of the other scenarios' ``times.py``: reads
``map-bf-9/map-bf-9_times_success.xlsx`` (mean time over the systems each
method made schedulable) and writes ``map-bf-9_times.pdf|png``.
"""

import matplotlib.pyplot as plt
import pandas as pd

from charts import COLUMNS, HERE, NAME, STYLES


def load():
    path = HERE / NAME / f"{NAME}_times_success.xlsx"
    df = pd.read_excel(path, index_col=0)
    return df[[c for c in COLUMNS if c in df.columns]]


def main():
    df = load()
    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    for col in df.columns:
        style = STYLES.get(col, {})
        df[col].plot.line(ax=ax, label=col, linewidth=1.5, markersize=5, **style)
    ax.set_xlabel("Average Utilization", fontweight="bold")
    ax.set_ylabel("Mean Time to Schedulable Solution (s)", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="x")
    ax.legend(loc="upper left", ncol=2, prop={"size": 8})
    fig.savefig(HERE / f"{NAME}_times.pdf")
    fig.savefig(HERE / f"{NAME}_times.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
