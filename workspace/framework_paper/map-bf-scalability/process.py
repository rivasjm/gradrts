"""Build the processed Excel and figure from the raw evaluator JSON.

Reads the flat records produced by ``harness.evaluate`` and writes:

- an Excel with two tables, both ``tool x column``:
  - ``schedulable``: number of systems the tool made schedulable (timeouts and
    tools that ran out without finding a schedule count as not schedulable).
  - ``mean_time``: mean time over the systems the tool made schedulable
    (per tool and column; blank when the tool never succeeded).
- a two-panel line figure (``<stem>.png`` and ``<stem>.pdf``): schedulable
  systems on top, mean time below, one line per tool, the horizontal axis
  being the matrix columns.

The column label comes from the ``"column"`` field of each record, which the
caller provides to ``harness.evaluate``.

Usage::

    .venv/bin/python workspace/framework_paper/map-bf-scalability/process.py \
        raw.json processed.xlsx [--tools pd hopa gdpa-500 bf] [--xlabel Tasks]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    with open(path) as handle:
        return json.load(handle)


def _ordered_columns(values):
    """Preserve first appearance, then sort numerically when possible."""
    labels = list(dict.fromkeys(values))
    try:
        return sorted(labels, key=float)
    except (TypeError, ValueError):
        return sorted(labels)


def build_tables(records, tools=None, columns=None):
    """Return ``(schedulable_counts, mean_times)`` as ``tool x column`` frames."""
    frame = pd.DataFrame(records)
    frame["ok"] = frame["schedulable"].astype(int)

    counts = frame.pivot_table(index="tool", columns="column", values="ok",
                               aggfunc="sum")
    times = (frame[frame["schedulable"]]
             .pivot_table(index="tool", columns="column", values="time",
                          aggfunc="mean"))

    tool_order = list(tools) if tools else sorted(frame["tool"].unique())
    col_order = list(columns) if columns else _ordered_columns(frame["column"])
    counts = counts.reindex(index=tool_order, columns=col_order).astype("Int64")
    times = times.reindex(index=tool_order, columns=col_order)
    return counts, times


def write_excel(counts, times, path):
    """Write both tables to ``path``, replacing it atomically."""
    tmp = f"{path}.tmp.xlsx"
    try:
        with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
            counts.to_excel(writer, sheet_name="schedulable")
            times.to_excel(writer, sheet_name="mean_time")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _atomic_savefig(fig, path):
    ext = Path(path).suffix
    tmp = f"{path}.tmp{ext}"
    try:
        fig.savefig(tmp, bbox_inches="tight")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_figure(counts, times, path, xlabel="column"):
    """Two-panel line figure: schedulable systems (top) and mean time (bottom),
    one line per tool, the horizontal axis being the matrix columns."""
    labels = [str(column) for column in counts.columns]
    positions = list(range(len(labels)))

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True,
                                      constrained_layout=True)
    for tool in counts.index:
        top.plot(positions, counts.loc[tool].astype(float).to_numpy(),
                 marker="o", markersize=4, linewidth=1.5, label=tool)
        bottom.plot(positions, times.loc[tool].astype(float).to_numpy(),
                    marker="o", markersize=4, linewidth=1.5, label=tool)

    top.set_ylabel("Schedulable systems", fontweight="bold")
    bottom.set_ylabel("Mean time to schedulable (s)", fontweight="bold")
    bottom.set_xlabel(xlabel, fontweight="bold")
    bottom.set_yscale("log")
    top.set_ylim(bottom=0)
    bottom.set_xticks(positions)
    bottom.set_xticklabels(labels)
    top.grid(True, axis="x")
    bottom.grid(True, axis="x")
    top.legend(fontsize=8, ncol=2)
    bottom.legend(fontsize=8, ncol=2)

    _atomic_savefig(fig, path)
    plt.close(fig)


def write_outputs(counts, times, excel_path, xlabel="column"):
    """Write the Excel and the matching ``.png``/``.pdf`` figure together."""
    write_excel(counts, times, excel_path)
    stem = Path(excel_path).with_suffix("")
    for extension in (".png", ".pdf"):
        write_figure(counts, times, str(stem.with_suffix(extension)), xlabel=xlabel)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="raw JSON produced by harness.evaluate")
    parser.add_argument("output", help="output Excel path")
    parser.add_argument("--tools", nargs="+", default=None,
                        help="tool order for the tables (default: alphabetical)")
    parser.add_argument("--xlabel", default="column",
                        help="horizontal axis label of the figure (default: column)")
    args = parser.parse_args()

    counts, times = build_tables(load(args.input), args.tools)
    write_outputs(counts, times, args.output, xlabel=args.xlabel)
    print(f"{args.output}: schedulable\n{counts}\n\nmean_time\n{times}")


if __name__ == "__main__":
    main()
