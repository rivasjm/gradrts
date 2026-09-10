"""Generate the aggregate schedulability-versus-time figure (one per size)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from figures import HERE, METHOD_STYLES, SCENARIOS, SIZES, load


def add_panel_label(ax, label):
    ax.text(-0.05, -0.1, label, transform=ax.transAxes, fontweight="bold",
            fontsize=18, horizontalalignment="right")


def add_scenario_label(ax, label):
    ax.text(0.95, 0.05, label, transform=ax.transAxes, ha="right", va="bottom",
            fontweight="bold", fontsize=18,
            bbox={"boxstyle": "round", "ec": "black", "fc": "bisque"})


def plot_efficiency(size):
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(18, 5.5), constrained_layout=True)

    for index, (ax, scenario) in enumerate(zip(axes, SCENARIOS)):
        schedulables = load(scenario, size, "schedulables").sum()
        times = load(scenario, size, "times_success").sum()

        for label in schedulables.index:
            color = METHOD_STYLES[label]["color"]
            ax.scatter(times[label], schedulables[label], s=180, color=color,
                       edgecolors="white", linewidth=1.5, zorder=3)
            ax.annotate(label, (times[label], schedulables[label]),
                        xytext=(8 if label == "pd" else -12, 6),
                        textcoords="offset points", fontsize=18, fontweight="bold",
                        color=color, ha="left" if label == "pd" else "right")

        ax.set_xscale("log")
        ax.set_ylim(0, 1000)
        ax.set_yticks(range(0, 1001, 200))
        ax.set_xlabel("Total execution time (s)", fontweight="bold", fontsize=18)
        if index == 0:
            ax.set_ylabel("Schedulable systems (/1000)", fontweight="bold", fontsize=18)
        ax.grid(True, which="both", axis="both", alpha=0.3)
        ax.tick_params(axis="both", labelsize=16)
        add_scenario_label(ax, scenario["title"])
        add_panel_label(ax, f"({chr(ord('a') + index)})")

    fig.savefig(HERE / f"efficiency_{size}.pdf")
    fig.savefig(HERE / f"efficiency_{size}.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    for size in SIZES:
        plot_efficiency(size)
