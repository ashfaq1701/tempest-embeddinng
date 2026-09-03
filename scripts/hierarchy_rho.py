"""Spearman(||E||, degree) for a YouTube embedding export.

Correct hierarchy = hubs (high degree) near the origin => NEGATIVE correlation
between radius ||E|| and node degree. Uses TRAIN edges only (split==0) for degree,
matching what the model saw during training.

Usage: python scripts/hierarchy_rho.py <embedding.npy> [csv]
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

emb_path = sys.argv[1]
csv_path = sys.argv[2] if len(sys.argv) > 2 else "datasets/YouTube/YouTube.csv"

E = np.load(emb_path)                                   # [num_nodes, d]
radius = np.linalg.norm(E, axis=1)

df = pd.read_csv(csv_path)
train = df[df["split"] == 0]
deg = np.zeros(E.shape[0], dtype=np.int64)
for col in ("src", "dst"):
    idx, cnt = np.unique(train[col].to_numpy(), return_counts=True)
    keep = idx < E.shape[0]
    np.add.at(deg, idx[keep], cnt[keep])

seen = deg > 0                                          # only nodes that appear in train
rho, p = spearmanr(radius[seen], deg[seen])
print(f"emb:            {emb_path}")
print(f"nodes (total):  {E.shape[0]:,}   seen-in-train: {seen.sum():,}")
print(f"|E| mean/max:   {radius.mean():.4f} / {radius.max():.4f}")
print(f"|E| (seen) mean/max: {radius[seen].mean():.4f} / {radius[seen].max():.4f}")
print(f"Spearman(||E||, degree) over seen nodes: rho={rho:+.4f}  (p={p:.2e})")
print(f"  => {'CORRECT hierarchy (hubs central)' if rho < 0 else 'INVERTED hierarchy (hubs on rim)'}")
