"""Two-panel d_emb sweep figure: peak test MRR and runtime per epoch.

Data source: experiment_logs/d-sweep/<dataset>/d-<d>.log (seed 42, K_train 5,
patience 3, 2-param geo_temp head, no pop bias). MRR is best_test_mrr -- test at
the best-VAL epoch, i.e. the val-selected checkpoint, which is the reported
metric. Runtime is the mean of (train + eval) over the cell's epochs.

Palette: dataviz categorical slots 1, 2, 7 (blue / orange / violet). Validated
all-pairs on the light surface #fcfcfb -- worst normal-vision dE 16.3 (floor 15),
worst CVD dE 14.2 (target >=8), every series >=3:1 contrast, so no relief-rule
label obligation and the legend can carry identity alone. Slot 3 (aqua) was
rejected at 2.74:1 contrast; slot 6 (green) at CVD dE 3.2 vs orange under protan.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

DIMS = [8, 16, 32, 64, 128, 256]

# dataset -> (test MRR at best-val checkpoint, mean train s, mean eval s) per dim
DATA = {
    "GoogleLocal": [(0.5530, 23.51, 3.92), (0.6043, 30.18, 3.47), (0.6301, 38.30, 4.79),
                    (0.6441, 48.38, 4.96), (0.6498, 73.47, 5.55), (0.6550, 129.86, 6.63)],
    "YouTube":     [(0.3911, 36.95, 20.68), (0.4685, 41.24, 16.16), (0.5302, 50.34, 22.58),
                    (0.5677, 66.42, 23.64), (0.5899, 100.57, 26.81), (0.6031, 174.20, 31.73)],
    "ML-20M":      [(0.2000, 189.60, 25.17), (0.2161, 188.91, 23.65), (0.2224, 206.87, 24.07),
                    (0.2234, 215.74, 24.10), (0.2234, 272.37, 29.38), (0.2245, 405.81, 35.75)],
}

SERIES = [("GoogleLocal", "#2a78d6", "o"),
          ("YouTube",     "#eb6834", "s"),
          ("ML-20M",      "#4a3aa7", "^")]

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dedddb"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.linewidth": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)

for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(DIMS))
    ax.xaxis.set_major_formatter(FixedFormatter([str(d) for d in DIMS]))
    ax.set_xlim(7, 300)
    ax.set_xlabel("d_emb")
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

for name, color, marker in SERIES:
    mrr = [row[0] for row in DATA[name]]
    rt = [row[1] + row[2] for row in DATA[name]]
    kw = dict(color=color, marker=marker, linewidth=2, markersize=7,
              markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax1.plot(DIMS, mrr, label=name, **kw)
    ax2.plot(DIMS, rt, label=name, **kw)

ax1.set_ylabel("test MRR")
ax1.set_ylim(0.15, 0.70)
ax1.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))

ax2.set_ylabel("runtime / epoch (s)")
ax2.set_yscale("log", base=2)
ax2.set_ylim(22, 520)
ax2.yaxis.set_major_locator(FixedLocator([25, 50, 100, 200, 400]))
ax2.yaxis.set_major_formatter(FixedFormatter(["25", "50", "100", "200", "400"]))
ax2.minorticks_off()

ax1.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42),
           handlelength=1.6, labelcolor=INK)

fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"/its/home/ms2420/tempest-wt-masterbr/figures/d_sweep.{ext}",
                dpi=200, facecolor=SURFACE)
print("wrote figures/d_sweep.png and figures/d_sweep.svg")
