"""Schedulability and execution-time figures for the FP scenario.

Produces, for both system sizes (15 and 25 tasks), one figure:

- ``fp.pdf/.png``: a 2x2 figure with the number of schedulable systems per
  utilization on the top row and the mean time to a schedulable solution (log
  scale) on the bottom row, one column per system size.

Brute force (``bf``) was only run at 15 tasks, so it only appears in the left
column.

    python workspace/framework_paper/figures-fp.py
"""

from figures_common import HERE, load, plot_grid

KEY = 'fp'
SIZES = (15, 25)
TITLES = ['15 tasks', '25 tasks']
SCENARIO = 'FP'


def main():
    sched = [load(KEY, size, 'schedulables', extra=('bf',)) for size in SIZES]
    times = [load(KEY, size, 'times_success', extra=('bf',)) for size in SIZES]

    plot_grid(sched, times, TITLES, HERE / KEY, SCENARIO, legend_cols=4)


if __name__ == '__main__':
    main()
