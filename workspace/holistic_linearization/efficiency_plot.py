"""Post-processes the SchedRatioEval XLSX outputs to create an
efficiency scatter plot: total schedulable systems vs total execution time.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
sched = pd.read_excel(os.path.join(OUTPUT_DIR, 'v3_schedulability_schedulables.xlsx'),
                      index_col=0)
times = pd.read_excel(os.path.join(OUTPUT_DIR, 'v3_schedulability_times.xlsx'),
                      index_col=0)

# Total schedulable (out of 2000) and total time (seconds)
total_sched = sched.sum()       # one value per method
total_time = times.sum()        # one value per method

fig, ax = plt.subplots(figsize=(7, 5))

colors = {'DM': '#999999', 'V3-opt': '#d6604d', 'HOPA': '#4393c3', 'GDPA': '#2166ac'}

for method in total_sched.index:
    color = colors.get(method, '#333333')
    ax.scatter(total_time[method], total_sched[method],
               s=140, color=color, edgecolors='white', linewidth=1.5, zorder=5)
    offset_x = total_time[method] * 0.01
    offset_y = 15
    ha = 'left'
    if method == 'V3-opt':
        offset_y = -30
    ax.annotate(method, (total_time[method], total_sched[method]),
                textcoords="offset points", xytext=(8, offset_y),
                fontsize=11, fontweight='bold', color=color, ha=ha)

ax.set_xlabel('Total execution time (seconds)')
ax.set_ylabel('Total schedulable systems (out of 2000)')
ax.set_title('Efficiency: schedulability vs computation cost')
ax.grid(True, alpha=0.3)

# Pareto frontier (manually: DM < V3 < HOPA < GDPA in both dimensions)
pareto_x = [total_time['DM'], total_time['HOPA'], total_time['GDPA']]
pareto_y = [total_sched['DM'], total_sched['HOPA'], total_sched['GDPA']]
ax.plot(pareto_x, pareto_y, '--', color='grey', alpha=0.5, linewidth=1, zorder=1)

ax.set_xlim(left=-0.05)
ax.set_ylim(bottom=1100, top=1850)

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, 'efficiency_scatter.png')
fig.savefig(path, dpi=150)
print(f'Saved to {path}')
