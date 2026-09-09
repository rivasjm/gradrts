import pandas as pd
import matplotlib.pyplot as plt

FP_EXCEL = "./fp/fp/fp_times_success.xlsx"
EDF_LOCAL_EXCEL = "./edf/edf/edf_times_success.xlsx"
FP_MAPPING_EXCEL = "./map/map/map_times_success.xlsx"

METHOD_STYLES = {
    'gdpa':     {'color': '#0000FF', 'marker': 'o', 'ls': '-'},
    'gdpa-100': {'color': '#0000FF', 'marker': '^', 'ls': '-'},
    'gdpa-200': {'color': '#FF8C00', 'marker': 'v', 'ls': '-'},
    'hopa':     {'color': '#008000', 'marker': 'x', 'ls': '--'},
    'pd':       {'color': '#8B4513', 'marker': 's', 'ls': ':'},
    'bf':       {'color': '#FF0000', 'marker': '*', 'ls': '-'},
}


def subfigure_labels(axs):
    for i, a in enumerate(axs):
        label = '(' + chr(ord('a') + i) + ')'
        a.text(-0.05, -0.1, label, fontweight='bold', fontsize='medium', horizontalalignment='right', transform=a.transAxes)


def plot_times():
    # load data
    fp = pd.read_excel(FP_EXCEL, index_col=0)
    mapping = pd.read_excel(FP_MAPPING_EXCEL, index_col=0)
    edfl = pd.read_excel(EDF_LOCAL_EXCEL, index_col=0)

    # reorder and select columns
    fp = fp[['gdpa-vec', 'hopa', 'pd', 'bf']]
    mapping = mapping[['pd', 'hopa', 'gdpa-100', 'gdpa-200']]
    edfl = edfl[['pd', 'hopa', 'gdpa']]

    # rename columns
    fp.rename(columns={'gdpa-vec': 'gdpa'}, inplace=True)

    # prepare chart
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=(12, 4))
    styles = ['+-', 'o-', 'x--', 's:', '*-', 'o-', 'v-', '^--']

    # plot
    def plot_line(ax, df):
        for col in df.columns:
            style = METHOD_STYLES[col]
            df[col].plot.line(ax=ax, color=style['color'], marker=style['marker'],
                              linestyle=style['ls'], linewidth=1.5, markersize=5)

    plot_line(axes[0], fp)
    plot_line(axes[1], mapping)
    plot_line(axes[2], edfl)

    # configure common properties of axes
    dataframes = [fp, mapping, edfl]
    for i, (ax, df) in enumerate(zip(axes, dataframes)):
        if i == 0:
            ax.set_ylabel("Mean Time to Schedulable Solution (s)", fontweight='bold')
        ax.set_xlabel("Average Utilization", fontweight='bold')
        ax.grid(True, which='major', axis='x')
        ax.set_yscale('log')
        anchor = 1.2 if len(df.columns) > 3 else 1.15
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, anchor), ncol=3, columnspacing=0.5, frameon=True, prop={'weight': 'bold', 'size': 9})

    # particular axes properties
    axes[0].text(0.95, 0.95, "FP", ha='right', va='top', transform=axes[0].transAxes, fontweight='bold', bbox=dict(boxstyle="round", ec='black', fc='bisque'))
    axes[1].text(0.95, 0.95, "MAP", ha='right', va='top', transform=axes[1].transAxes, fontweight='bold', bbox=dict(boxstyle="round", ec='black', fc='bisque'))
    axes[2].text(0.95, 0.95, "EDF", ha='right', va='top', transform=axes[2].transAxes, fontweight='bold', bbox=dict(boxstyle="round", ec='black', fc='bisque'))
    subfigure_labels(axes)

    # save fig
    fig.savefig("times.pdf")
    fig.savefig("times.png")


def main():
    plot_times()


if __name__ == '__main__':
    main()
