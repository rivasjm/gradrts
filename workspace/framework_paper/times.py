import matplotlib.pyplot as plt

from figures import HERE, METHOD_STYLES, SCENARIOS, SIZES, load, subfigure_labels


def plot_times(size):
    frames = [load(scenario, size, 'times_success') for scenario in SCENARIOS]

    fig, axes = plt.subplots(nrows=1, ncols=len(SCENARIOS), constrained_layout=True,
                             figsize=(16, 4))

    for ax, df in zip(axes, frames):
        for col in df.columns:
            style = METHOD_STYLES[col]
            df[col].plot.line(ax=ax, color=style['color'], marker=style['marker'],
                              linestyle=style['ls'], linewidth=1.5, markersize=5)

    for i, (ax, scenario, df) in enumerate(zip(axes, SCENARIOS, frames)):
        if i == 0:
            ax.set_ylabel("Mean Time to Schedulable Solution (s)", fontweight='bold')
        ax.set_xlabel("Average Utilization", fontweight='bold')
        ax.grid(True, which='major', axis='x')
        ax.set_yscale('log')
        anchor = 1.2 if len(df.columns) > 3 else 1.15
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, anchor), ncol=3,
                  columnspacing=0.5, frameon=True, prop={'weight': 'bold', 'size': 9})
        ax.text(0.95, 0.95, scenario['title'], ha='right', va='top',
                transform=ax.transAxes, fontweight='bold',
                bbox=dict(boxstyle="round", ec='black', fc='bisque'))

    subfigure_labels(axes)
    fig.savefig(HERE / f"times_{size}.pdf")
    fig.savefig(HERE / f"times_{size}.png")
    plt.close(fig)


def main():
    for size in SIZES:
        plot_times(size)


if __name__ == '__main__':
    main()
