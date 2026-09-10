"""Shared configuration and loading helpers for the evaluation figures.

The figure scripts (`charts.py`, `times.py`, `charts-efficiency.py`) iterate over
``SIZES`` and read the ``.xlsx`` files produced by each scenario under
``<scenario-dir>/<scenario>-<size>/``.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

SIZES = (15, 25)

METHOD_STYLES = {
    'gdpa':     {'color': '#0000FF', 'marker': 'o', 'ls': '-'},
    'gdpa-100': {'color': '#0000FF', 'marker': '^', 'ls': '-'},
    'gdpa-200': {'color': '#FF8C00', 'marker': 'v', 'ls': '-'},
    'hopa':     {'color': '#008000', 'marker': 'x', 'ls': '--'},
    'pd':       {'color': '#8B4513', 'marker': 's', 'ls': ':'},
    'bf':       {'color': '#FF0000', 'marker': '*', 'ls': '-'},
}

# One entry per figure panel. ``directory`` is where the scenario script writes
# its output, ``key`` is the evaluation name prefix (also the output subdir name).
SCENARIOS = (
    {'key': 'fp', 'title': 'FP', 'directory': 'fp',
     'columns': ('gdpa-vec', 'hopa', 'pd', 'bf')},
    {'key': 'map', 'title': 'MAP', 'directory': 'map',
     'columns': ('pd', 'hopa', 'gdpa-100', 'gdpa-200')},
    {'key': 'map-unbalanced', 'title': 'MAP unbal.', 'directory': 'map',
     'columns': ('pd', 'hopa', 'gdpa-100', 'gdpa-200')},
    {'key': 'edf', 'title': 'EDF', 'directory': 'edf',
     'columns': ('pd', 'hopa', 'gdpa')},
)


def data_file(scenario, size, suffix):
    name = scenario['key']
    return HERE / scenario['directory'] / f"{name}-{size}" / f"{name}-{size}_{suffix}.xlsx"


def load(scenario, size, suffix):
    """Load a scenario's data, keeping only the columns shown and renaming
    ``gdpa-vec`` to ``gdpa``. Missing optional columns (e.g. ``bf`` for sizes
    where brute force is not run) are dropped."""
    df = pd.read_excel(data_file(scenario, size, suffix), index_col=0)
    columns = [c for c in scenario['columns'] if c in df.columns]
    df = df[columns].copy()
    df.rename(columns={'gdpa-vec': 'gdpa'}, inplace=True)
    return df


def subfigure_labels(axs):
    for i, a in enumerate(axs):
        label = '(' + chr(ord('a') + i) + ')'
        a.text(-0.05, -0.1, label, fontweight='bold', fontsize='medium',
               horizontalalignment='right', transform=a.transAxes)
