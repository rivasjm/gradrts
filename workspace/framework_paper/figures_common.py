"""Shared configuration and helpers for the paper evaluation figures.

Each scenario has its own figure script (``figures-fp.py``, ``figures-edf.py``,
``figures-map.py``); this module holds the common configuration so that a given
technique keeps the same color, marker and line style across every figure in the
paper. The executable scripts use hyphens, but this module must be importable
and therefore uses an underscore.

The FP and EDF scripts read ``<key>/<key>-<size>/<key>-<size>_<suffix>.xlsx``,
produced by ``fp/fp.py`` and ``edf/edf.py``. The MAP script reads the processed
workbook of the ``map-bf-scalability`` run.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

SIZES = (15, 25)

# Scenarios that follow the ``<key>/<key>-<size>/`` layout. ``columns`` is the
# order/methods shown; ``bf`` is only added where brute force was run.
SCENARIOS = {
    'fp': {'directory': 'fp', 'columns': ('gdpa', 'hopa', 'pd')},
    'edf': {'directory': 'edf', 'columns': ('gdpa', 'hopa', 'pd')},
}

# GDPA variants share the blue family (light -> dark with the iteration budget);
# ``gdpa-prio`` is grey because it keeps the mapping fixed and is not directly
# comparable with the mapping variants. The baselines keep their classic colors.
METHOD_STYLES = {
    'gdpa':      {'color': '#0000FF', 'marker': 'o', 'ls': '-'},
    'gdpa-100':  {'color': '#4C72FF', 'marker': '^', 'ls': '-'},
    'gdpa-200':  {'color': '#7F9BFF', 'marker': 'v', 'ls': '-'},
    'gdpa-500':  {'color': '#000080', 'marker': 'D', 'ls': '-'},
    'gdpa-prio': {'color': '#999999', 'marker': '+', 'ls': ':'},
    'hopa':      {'color': '#008000', 'marker': 'x', 'ls': '--'},
    'pd':        {'color': '#8B4513', 'marker': 's', 'ls': ':'},
    'bf':        {'color': '#FF0000', 'marker': '*', 'ls': '-'},
}

# Legend labels: the paper writes every method in uppercase (GDPA, HOSPA, PD,
# BF, GDPA-100, ...), so the column names are upper-cased unless overridden here.
# The ``hopa`` column is HOSPA, the generalization of HOPA to FP and EDF (under
# FP it is equivalent to HOPA), which is the name used in the paper.
METHOD_LABELS = {'hopa': 'HOSPA'}


def legend_label(name):
    return METHOD_LABELS.get(name, name.upper())


def data_file(key, size, suffix):
    directory = SCENARIOS[key]['directory']
    return HERE / directory / f"{key}-{size}" / f"{key}-{size}_{suffix}.xlsx"


def load(key, size, suffix, extra=()):
    """Load a scenario's data keeping only the columns shown. ``extra`` adds
    optional columns (e.g. ``bf`` at the sizes where it was run); columns that
    are not present are dropped."""
    columns = tuple(SCENARIOS[key]['columns']) + tuple(extra)
    df = pd.read_excel(data_file(key, size, suffix), index_col=0)
    return df[[c for c in columns if c in df.columns]].copy()


def draw(ax, df, logy=False):
    """Plot every column of ``df`` on ``ax`` using the shared styles."""
    for col in df.columns:
        style = METHOD_STYLES[col]
        ax.plot(df.index, df[col], color=style['color'], marker=style['marker'],
                linestyle=style['ls'], linewidth=1.2, markersize=4,
                label=legend_label(col))
    ax.set_xlabel("Average Utilization", fontweight='bold', fontsize=8)
    ax.grid(True, which='major', axis='x')
    if logy:
        ax.set_yscale('log')
        ax.margins(y=0.15)


def subfigure_labels(axs):
    for i, a in enumerate(axs):
        label = '(' + chr(ord('a') + i) + ')'
        a.text(-0.05, -0.1, label, fontweight='bold', fontsize='medium',
               horizontalalignment='right', transform=a.transAxes)


def save_figure(fig, out_stem):
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png")
    import matplotlib.pyplot as plt
    plt.close(fig)


UTILIZATION_TICKS = (0.5, 0.6, 0.7, 0.8, 0.9)


def _box(ax, text, xy, ha, va):
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va,
            fontweight='bold', fontsize=8,
            bbox=dict(boxstyle='round', ec='black', fc='bisque'))


def add_scenario_label(ax, text, corner):
    """Scenario (and, when applicable, system size) label inside the panel:
    bottom-left for count panels (schedulable and finished systems), top-left for
    the time panels where the curves grow."""
    if corner == 'top-left':
        _box(ax, text, (0.03, 0.95), 'left', 'top')
    else:
        _box(ax, text, (0.03, 0.05), 'left', 'bottom')


def plot_grid(sched, times, panel_titles, out_stem, scenario, legend_cols=4):
    """One double-column 2x2 figure per scenario: schedulable systems on the top
    row and mean time to a schedulable solution (log scale) on the bottom row,
    one column per system size. Every panel shares the utilization axis, and the
    panels of a row share their vertical axis so that both sizes can be compared
    directly."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    # Full page width, but a reduced height and extra horizontal space between
    # the two columns so that each panel stays compact.
    fig = plt.figure(figsize=(7.0, 4.0), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.05, hspace=0.04)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.1, 1, 1])
    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis('off')
    top_left = fig.add_subplot(gs[1, 0])
    axes = [[top_left, fig.add_subplot(gs[1, 1], sharex=top_left, sharey=top_left)]]
    bottom_left = fig.add_subplot(gs[2, 0], sharex=top_left)
    axes.append([bottom_left, fig.add_subplot(gs[2, 1], sharex=top_left, sharey=bottom_left)])

    rows = ((sched, 'Schedulable Systems', False, 'bottom-left'),
            (times, 'Mean Time to\nSchedulable (s)', True, 'top-left'))
    letters = iter('abcd')
    for row, (frames, ylabel, logy, corner) in zip(axes, rows):
        for col, (ax, df, title) in enumerate(zip(row, frames, panel_titles)):
            draw(ax, df, logy=logy)
            if logy:
                # one major tick per decade, also in the reduced panel height
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
            ax.set_xticks(UTILIZATION_TICKS)
            ax.tick_params(labelsize=8)
            ax.set_title(f'({next(letters)})', loc='left', fontweight='bold', fontsize=8)
            add_scenario_label(ax, f'{scenario} {title}', corner)
            if col == 0:
                ax.set_ylabel(ylabel, fontweight='bold', fontsize=8)
            else:
                ax.tick_params(labelleft=False)
    for ax in axes[0]:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)

    # The legend collects the methods of every panel (e.g. brute force is only
    # present at the smallest size).
    handles, labels = {}, []
    for ax in (a for row in axes for a in row):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in handles:
                handles[label] = handle
                labels.append(label)
    legend_ax.legend([handles[l] for l in labels], labels, loc='center', ncol=legend_cols,
                     frameon=False, columnspacing=0.8, handletextpad=0.4,
                     prop={'weight': 'bold', 'size': 8})
    save_figure(fig, out_stem)
