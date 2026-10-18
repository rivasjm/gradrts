"""Generate the figures for the IEEE paper (edf_wad/paper)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 200})


def ours(sz):
    d = os.path.join(CODE, "workspace", "edf_wad", f"edf-wad-{sz}")
    return (pd.read_excel(os.path.join(d, f"edf-wad-{sz}_schedulables.xlsx"), index_col=0),
            pd.read_excel(os.path.join(d, f"edf-wad-{sz}_times.xlsx"), index_col=0))


def hospa(sz):
    d = os.path.join(CODE, "workspace", "edf_wad", f"hospa-{sz}")
    return (pd.read_excel(os.path.join(d, f"hospa-{sz}_schedulables.xlsx"), index_col=0),
            pd.read_excel(os.path.join(d, f"hospa-{sz}_times.xlsx"), index_col=0))


def rta(sz):
    d = os.path.join(CODE, "workspace", "edf_wad", f"rta_cd-{sz}")
    return (pd.read_csv(os.path.join(d, "schedulables.csv"), index_col=0),
            pd.read_csv(os.path.join(d, "times.csv"), index_col=0))


o15s, o15t = ours(15)
o25s, o25t = ours(25)

# --- Figure 1: the observation ---
fam = pd.read_csv(os.path.join(CODE, "workspace", "edf_wad", "families.csv"))
g = fam[fam["series"] == "gamma"]
lam = fam[fam["series"] == "lambda"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
a1.plot(g["param"], g["count"], marker="o", ms=3, lw=1.1, color="#1f77b4")
a1.set_xlabel(r"partial globalization $\gamma$"); a1.set_ylabel("schedulable / 1000")
a1.set_title(r"(a) $d_k = s_k + \gamma\sum_{j<k}s_j$", fontsize=8)
a2.plot(lam["param"], lam["count"], marker="o", ms=3, lw=1.1, color="#d62728")
a2.set_xlabel(r"uniform scale $\lambda$"); a2.set_ylabel("schedulable / 1000")
a2.set_title(r"(b) $d_k = \lambda\, s_k$", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_observation.png"))
fig.savefig(os.path.join(FIG, "fig1_observation.pdf"))
plt.close(fig)

LABEL = {"pd": "PD", "eqf": "EQF", "eqs": "EQS", "wad": "WAD",
         "wad_adapt": "WAD-Adapt", "wad_budget": "WAD-Budget",
         "hospa": "HOSPA", "rta_cd": "RTA-CD"}

STYLE = [("pd", "#7f7f7f", "-"), ("wad", "#2ca02c", "-"),
         ("wad_adapt", "#1f77b4", "-"), ("wad_budget", "#17becf", "-"),
         ("eqf", "#bcbd22", "-"), ("eqs", "#e377c2", "-"),
         ("rta_cd", "#9467bd", "-"), ("hospa", "#8c564b", "--")]


def series(sz, s, t):
    d = {}
    for k in ("pd", "wad", "wad_adapt", "wad_budget", "eqf", "eqs"):
        if k in s:
            d[k] = (s[k], t[k])
    hs, ht = hospa(sz)
    d["hospa"] = (hs["hospa"], ht["hospa"])
    rs, rt = rta(sz)
    d["rta_cd"] = (rs["rta_cd"].reindex(s.index), rt["rta_cd"].reindex(s.index))
    return d


# --- Figure 2: schedulables per level ---
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
for ax, sz, s, t in [(axes[0], 15, o15s, o15t), (axes[1], 25, o25s, o25t)]:
    ser = series(sz, s, t)
    for name, color, ls in STYLE:
        if name in ser:
            y = ser[name][0]
            ax.plot(y.index, y, marker="o", ms=2.0, lw=0.9, color=color, ls=ls, label=LABEL[name])
    ax.set_xlabel("utilization"); ax.set_ylim(0, 52); ax.set_title(f"size {sz}", fontsize=8)
axes[0].set_ylabel("schedulable / 50")
axes[1].legend(fontsize=5.5, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_schedulables.png")); fig.savefig(os.path.join(FIG, "fig2_schedulables.pdf")); plt.close(fig)

# --- Figure 3: efficiency ---
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
for ax, sz, s, t in [(axes[0], 15, o15s, o15t), (axes[1], 25, o25s, o25t)]:
    ser = series(sz, s, t)
    for name, color, _ in STYLE:
        if name in ser:
            y, tt = ser[name]
            ax.scatter(tt.mean() * 1000, y.sum(), s=38, color=color, edgecolors="white",
                       linewidth=0.6, zorder=5)
            ax.annotate(LABEL[name], (tt.mean() * 1000, y.sum()), textcoords="offset points",
                        xytext=(5, 3), fontsize=6)
    ax.set_xscale("log"); ax.set_xlabel("total time (s, log)"); ax.set_title(f"size {sz}", fontsize=8)
axes[0].set_ylabel("schedulable / 1000")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_efficiency.png")); fig.savefig(os.path.join(FIG, "fig3_efficiency.pdf")); plt.close(fig)

# --- Figure 4: times ---
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
for ax, sz, s, t in [(axes[0], 15, o15s, o15t), (axes[1], 25, o25s, o25t)]:
    ser = series(sz, s, t)
    for name, color, ls in STYLE:
        if name in ser:
            y = ser[name][1]
            ax.plot(y.index, y, marker="o", ms=2.0, lw=0.9, color=color, ls=ls, label=LABEL[name])
    ax.set_yscale("log"); ax.set_xlabel("utilization"); ax.set_title(f"size {sz}", fontsize=8)
axes[0].set_ylabel("mean time / instance (s, log)")
axes[1].legend(fontsize=5.5, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_times.png")); fig.savefig(os.path.join(FIG, "fig4_times.pdf")); plt.close(fig)

# --- Figure 5: Q x S sensitivity heatmap ---
g = pd.read_csv(os.path.join(CODE, "workspace", "edf_wad", "qs_grid.csv"), index_col=0)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
im = ax.imshow(g.values, origin="lower", aspect="auto", cmap="viridis")
ax.set_xticks(range(len(g.columns))); ax.set_xticklabels([f"{float(c):g}" for c in g.columns])
ax.set_yticks(range(len(g.index))); ax.set_yticklabels([f"{float(i):g}" for i in g.index])
ax.set_xlabel(r"$\kappa_{\mathrm{p}}$"); ax.set_ylabel(r"$\kappa_{\mathrm{w}}$"); ax.grid(False)
for i in range(g.shape[0]):
    for j in range(g.shape[1]):
        ax.text(j, i, f"{int(g.values[i, j])}", ha="center", va="center", fontsize=5,
                color="white" if g.values[i, j] < g.values.max() - 30 else "black")
fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_sensitivity.png")); fig.savefig(os.path.join(FIG, "fig5_sensitivity.pdf")); plt.close(fig)

print("figures written")
for f in sorted(os.listdir(FIG)):
    print("  ", f)
