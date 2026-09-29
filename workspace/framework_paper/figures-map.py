"""Schedulability, time and scalability figure for the MAP scenario.

Reads the processed workbook of the ``map-bf-scalability-0.75-n15`` run (fixed
utilization 0.75 on 25 systems whose task count grows from the 4-task base up to
15) and writes ``map-scalability.pdf/.png`` with three panels: schedulable
systems, mean time to a schedulable solution (log scale) and finished systems.

The single figure covers both the exact comparison at small sizes (where the
brute force still terminates) and the scalability limit (where it times out
while GDPA keeps scaling).

    python workspace/framework_paper/figures-map.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from figures_common import (HERE, METHOD_STYLES, add_scenario_label,
                            legend_label, save_figure)

plt.rcParams['font.size'] = 8

RUN = 'map-bf-scalability-0.75-n15'
XLSX = HERE / 'map-bf-scalability' / RUN / f'{RUN}_processed.xlsx'
# Only methods that optimize the mapping and the priorities are shown, so that
# all the lines in the figure are directly comparable.
TOOLS = ('gdpa-100', 'gdpa-200', 'gdpa-500', 'bf')


def _sheet(name):
    df = pd.read_excel(XLSX, sheet_name=name, index_col=0)
    return df.reindex(list(TOOLS))


def main():
    sched = _sheet('schedulable')
    times = _sheet('mean_time')
    finished = _sheet('finished')
    x = [int(c) for c in sched.columns]

    fig = plt.figure(figsize=(7.5, 2.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.2, 1])
    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis('off')
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    top, mid, bot = axes

    for tool in TOOLS:
        style = METHOD_STYLES[tool]
        for ax, data in zip(axes, (sched, times, finished)):
            ax.plot(x, data.loc[tool].to_numpy(dtype=float), color=style['color'],
                    marker=style['marker'], linestyle=style['ls'], linewidth=1.2,
                    markersize=4, label=legend_label(tool))

    top.set_ylabel("Schedulable Systems", fontweight='bold')
    mid.set_ylabel("Mean Time to Schedulable (s)", fontweight='bold')
    bot.set_ylabel("Finished Systems", fontweight='bold')
    mid.set_yscale('log')
    ticks = list(range(4, 16, 2))
    corners = ('bottom-left', 'top-left', 'bottom-left')
    for ax, label, corner in zip(axes, ('(a)', '(b)', '(c)'), corners):
        ax.grid(True, which='major', axis='x')
        ax.set_xlabel("Number of Tasks", fontweight='bold')
        ax.set_xticks(ticks)
        ax.set_title(label, loc='left', fontweight='bold')
        add_scenario_label(ax, 'MAP', corner)
    top.set_ylim(bottom=0)
    bot.set_ylim(bottom=0)

    handles, labels = top.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='center', ncol=4, frameon=False,
                     columnspacing=1.0, handletextpad=0.5,
                     prop={'weight': 'bold', 'size': 8})
    save_figure(fig, HERE / 'map-scalability')


if __name__ == '__main__':
    main()
