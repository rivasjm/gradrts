import matplotlib.pyplot as plt

from figures import HERE, METHOD_STYLES, SCENARIOS, SIZES, load, subfigure_labels


def plot_schedulables(size):
    frames = [load(scenario, size, 'schedulables') for scenario in SCENARIOS]

    fig, axes = plt.subplots(nrows=1, ncols=len(SCENARIOS), constrained_layout=True,
                             figsize=(16, 3.2))

    for ax, df in zip(axes, frames):
        for col in df.columns:
            style = METHOD_STYLES[col]
            df[col].plot.line(ax=ax, color=style['color'], marker=style['marker'],
                              linestyle=style['ls'], linewidth=1.5, markersize=5)

    for i, (ax, scenario) in enumerate(zip(axes, SCENARIOS)):
        if i == 0:
            ax.set_ylabel("Schedulable Systems", fontweight='bold')
        ax.set_xlabel("Average Utilization", fontweight='bold')
        ax.grid(True, which='major', axis='x')
        ax.legend(loc='lower left', ncol=3, columnspacing=0.5,
                  prop={'weight': 'bold', 'size': 9})
        ax.text(0.95, 0.95, scenario['title'], ha='right', va='top',
                transform=ax.transAxes, fontweight='bold',
                bbox=dict(boxstyle="round", ec='black', fc='bisque'))

    subfigure_labels(axes)
    fig.savefig(HERE / f"schedulables_{size}.pdf")
    fig.savefig(HERE / f"schedulables_{size}.png")
    plt.close(fig)


def main():
    for size in SIZES:
        plot_schedulables(size)


if __name__ == '__main__':
    main()
