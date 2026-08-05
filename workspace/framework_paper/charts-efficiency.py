"""Generate the aggregate schedulability-versus-time figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE

METHOD_COLORS = {
    "gdpa-vec": "#1f77b4",
    "gdpa-seq": "#9467bd",
    "gdpa+map-vec": "#ff7f0e",
    "gdpa+map-seq": "#2ca02c",
    "gdpa": "#1f77b4",
    "hopa": "#8c564b",
    "pd": "#17becf",
    "bf": "#d62728",
}

SCENARIOS = (
    {
        "name": "FP",
        "directory": "fp",
        "prefix": "gradient_fp_eval",
        "methods": ("gdpa-vec", "gdpa-seq", "hopa", "pd", "bf"),
        "labels": ("gdpa-vec", "gdpa-seq", "hopa", "pd", "bf"),
    },
    {
        "name": "MAP",
        "directory": "fp-mapping",
        "prefix": "gradient_fp_mapping_eval",
        "methods": ("gdpa-mapping-vec", "gdpa-mapping-seq", "gdpa", "pd"),
        "labels": ("gdpa+map-vec", "gdpa+map-seq", "gdpa", "pd"),
    },
    {
        "name": "EDF",
        "directory": "edf-local",
        "prefix": "gradient_edf_local_eval",
        "methods": ("EDF-L GDPA", "EDF-L HOPA", "EDF-L PD"),
        "labels": ("gdpa", "hopa", "pd"),
    },
)


def load_totals(scenario):
    data_dir = HERE / scenario["directory"]
    schedulables = pd.read_excel(
        data_dir / f'{scenario["prefix"]}_schedulables.xlsx', index_col=0
    )
    times = pd.read_excel(
        data_dir / f'{scenario["prefix"]}_times.xlsx', index_col=0
    )

    methods = list(scenario["methods"])
    return schedulables[methods].sum(), times[methods].sum()


def add_panel_label(ax, label):
    ax.text(
        -0.05,
        -0.1,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=18,
        horizontalalignment="right",
    )


def add_scenario_label(ax, label):
    ax.text(
        0.95,
        0.05,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
        fontsize=18,
        bbox={"boxstyle": "round", "ec": "black", "fc": "bisque"},
    )


def plot_efficiency():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    for index, (ax, scenario) in enumerate(zip(axes, SCENARIOS)):
        schedulables, times = load_totals(scenario)

        for method, label, color in zip(
            scenario["methods"],
            scenario["labels"],
            (METHOD_COLORS[label] for label in scenario["labels"]),
        ):
            ax.scatter(
                times[method],
                schedulables[method],
                s=180,
                color=color,
                edgecolors="white",
                linewidth=1.5,
                zorder=3,
            )
            ax.annotate(
                label,
                (times[method], schedulables[method]),
                xytext=(8 if label == "pd" else -12, 6),
                textcoords="offset points",
                fontsize=18,
                fontweight="bold",
                color=color,
                ha="left" if label == "pd" else "right",
            )

        ax.set_xscale("log")
        ax.set_ylim(0, 1000)
        ax.set_yticks(range(0, 1001, 200))
        ax.set_xlabel("Total execution time (s)", fontweight="bold", fontsize=18)
        if index == 0:
            ax.set_ylabel("Schedulable systems (/1000)", fontweight="bold", fontsize=18)
        ax.grid(True, which="both", axis="both", alpha=0.3)
        ax.tick_params(axis="both", labelsize=16)
        add_scenario_label(ax, scenario["name"])
        add_panel_label(ax, f"({chr(ord('a') + index)})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "efficiency.pdf")
    fig.savefig(OUTPUT_DIR / "efficiency.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    plot_efficiency()
