import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import openpyxl
from collections import defaultdict

FP_EXCEL = "../gradient_fp_validation/gradient_fp_validation_schedulables.xlsx"
EDF_LOCAL_EXCEL = "../gradient_edf_local_validation/gradient_edf_local_validation_schedulables.xlsx"
FP_MAPPING_EXCEL = "../gradient_fp_mapping_validation/gradient_fp_mapping_validation_schedulables.xlsx"


# def add_text(ax, posx, posy, label, size='small', align='center'):
#     ax.text(posx, posy, label, fontsize=size, ha=align, transform=ax.transAxes,
#             fontweight='bold', bbox=dict(boxstyle="round", ec='black', fc='bisque'))


def subfigure_labels(axs):
    for i, a in enumerate(axs):
        label =  '(' + chr(ord('a') + i) + ')'
        a.text(-0.05, -0.1, label, fontweight='bold', fontsize='medium', horizontalalignment='right', transform=a.transAxes)


def plot_schedulables():
    # load data
    fp = pd.read_excel(FP_EXCEL, index_col=0)
    mapping = pd.read_excel(FP_MAPPING_EXCEL, index_col=0)
    edfl = pd.read_excel(EDF_LOCAL_EXCEL, index_col=0)

    # reorder and select columns
    fp = fp[['gdpa', 'hopa', 'pd']]
    mapping = mapping[['gdpa-mapping', 'gdpa', 'pd']]
    edfl = edfl[['EDF-L GDPA', 'EDF-L HOPA', 'EDF-L PD']]

    # rename columns
    mapping.rename(columns={'gdpa-mapping':'gdpa+map'}, inplace=True)
    edfl.rename(columns={'EDF-L GDPA':'gdpa', 'EDF-L HOPA': 'hopa', 'EDF-L PD': 'pd'}, inplace=True)

    # prepare chart
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=(12, 3))
    styles = ['+-', 'o-', 'x--', 's:', '*-', 'o-']

    # plot
    fp.plot.line(ax=axes[0], style=styles)
    mapping.plot.line(ax=axes[1], style=styles)
    edfl.plot.line(ax=axes[2], style=styles)

    # configure common properties of axes
    for i, ax in enumerate(axes):
        if i == 0:
            ax.set_ylabel("Schedulable Systems", fontweight='bold')
        ax.set_xlabel("Average Utilization", fontweight='bold')
        ax.grid(True, which='major', axis='x')
        ax.legend(loc='lower left', ncol=3, columnspacing=0.5, prop={'weight': 'bold', 'size': 9})

    # particular axes properties
    axes[0].text(0.95, 0.95, "FP", ha='right', va='top', transform=axes[0].transAxes, fontweight='bold', bbox=dict(boxstyle="round", ec='black', fc='bisque'))
    axes[1].text(0.95, 0.95, "MAP", ha='right', va='top', transform=axes[1].transAxes, fontweight='bold',bbox=dict(boxstyle="round", ec='black', fc='bisque'))
    axes[2].text(0.95, 0.95, "EDF", ha='right', va='top', transform=axes[2].transAxes, fontweight='bold', bbox=dict(boxstyle="round", ec='black', fc='bisque'))
    subfigure_labels(axes)

    # save fig
    fig.savefig("schedulables.pdf")
    fig.savefig("schedulables.png")


def main():
    plot_schedulables()


if __name__ == '__main__':
    main()