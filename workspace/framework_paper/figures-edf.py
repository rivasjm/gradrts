"""Schedulability and execution-time figures for the EDF scenario.

Produces, for both system sizes (15 and 25 tasks), one figure:

- ``edf.pdf/.png``: a 2x2 figure with the number of schedulable systems per
  utilization on the top row and the mean time to a schedulable solution (log
  scale) on the bottom row, one column per system size.

    python workspace/framework_paper/figures-edf.py
"""

from figures_common import HERE, load, plot_grid

KEY = 'edf'
SIZES = (15, 25)
TITLES = ['15 tasks', '25 tasks']
SCENARIO = 'EDF'


def main():
    sched = [load(KEY, size, 'schedulables') for size in SIZES]
    times = [load(KEY, size, 'times_success') for size in SIZES]

    plot_grid(sched, times, TITLES, HERE / KEY, SCENARIO, legend_cols=3)


if __name__ == '__main__':
    main()
