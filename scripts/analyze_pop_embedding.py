"""Quantitative analysis of a Poincaré-ball embedding export, coloured/keyed by node degree.

Produces a 4-panel figure:
  (1) radius ||E|| distribution (the hierarchy/depth axis)
  (2) radius vs degree hexbin (does depth encode popularity? Spearman rho annotated)
  (3) train-degree distribution (log-log, for context on the graph's hub structure)
  (4) PCA explained-variance ratio (how many dims the geometry actually uses)

Usage: python scripts/analyze_pop_embedding.py <embedding.npy> [csv] [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

emb_path = sys.argv[1]
csv_path = sys.argv[2] if len(sys.argv) > 2 else "datasets/YouTube/YouTube.csv"
out_path = sys.argv[3] if len(sys.argv) > 3 else "logs/plots/youtube_pop_analysis.png"

E = np.load(emb_path)                                  # [num_nodes, d]
n, d = E.shape
radius = np.linalg.norm(E, axis=1)                     # Euclidean norm = depth proxy on the ball

# TRAIN-split degree (what the model saw), matching hierarchy_rho.
df = pd.read_csv(csv_path)
train = df[df["split"] == 0]
deg = np.zeros(n, dtype=np.int64)
for col in ("src", "dst"):
    idx, cnt = np.unique(train[col].to_numpy(), return_counts=True)
    keep = idx < n
    deg[idx[keep]] += cnt[keep]

seen = deg > 0
rho, _ = spearmanr(radius[seen], deg[seen])

fig, ax = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(f"YouTube d=64 + pop_bias embedding  (n={n:,},  ||E|| mean={radius.mean():.3f} "
             f"max={radius.max():.3f})", fontsize=13)

# (1) radius distribution
ax[0, 0].hist(radius[seen], bins=120, color="#3b7dd8", alpha=0.85)
ax[0, 0].axvline(radius[seen].mean(), color="k", ls="--", lw=1, label=f"mean {radius[seen].mean():.3f}")
ax[0, 0].set(title="(1) radius ||E|| distribution (depth axis)", xlabel="||E||", ylabel="# nodes")
ax[0, 0].legend()

# (2) radius vs degree
hb = ax[0, 1].hexbin(deg[seen], radius[seen], xscale="log", gridsize=45,
                     cmap="viridis", mincnt=1, bins="log")
ax[0, 1].set(title=f"(2) radius vs train-degree   Spearman rho = {rho:+.3f}",
             xlabel="train degree (log)", ylabel="||E||")
fig.colorbar(hb, ax=ax[0, 1], label="log10(#nodes)")

# (3) degree distribution (log-log)
u_deg, u_cnt = np.unique(deg[seen], return_counts=True)
ax[1, 0].loglog(u_deg, u_cnt, ".", color="#d8663b", ms=4)
ax[1, 0].set(title="(3) train-degree distribution (log-log)", xlabel="degree", ylabel="# nodes")

# (4) PCA explained variance (effective dimensionality)
pca = PCA(n_components=min(d, 64)).fit(E[seen][:50000] if seen.sum() > 50000 else E[seen])
cum = np.cumsum(pca.explained_variance_ratio_)
ax[1, 1].plot(np.arange(1, len(cum) + 1), cum, "-o", ms=3, color="#3bd88a")
k90 = int(np.searchsorted(cum, 0.90) + 1)
ax[1, 1].axhline(0.90, color="k", ls=":", lw=1)
ax[1, 1].axvline(k90, color="k", ls=":", lw=1, label=f"90% var @ {k90} dims")
ax[1, 1].set(title="(4) PCA cumulative explained variance", xlabel="# components",
             ylabel="cum. variance ratio")
ax[1, 1].legend()

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(out_path, dpi=110)
print(f"saved {out_path}")
print(f"n={n:,}  seen={seen.sum():,}  ||E|| mean/max={radius.mean():.4f}/{radius.max():.4f}")
print(f"Spearman(||E||, degree)={rho:+.4f}  |  90% variance in {k90}/{d} dims")
print(f"radius pctiles: p10={np.percentile(radius[seen],10):.3f} p50={np.percentile(radius[seen],50):.3f} "
      f"p90={np.percentile(radius[seen],90):.3f} p99={np.percentile(radius[seen],99):.3f}")
