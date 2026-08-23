"""Minimal Poincaré-disk plot of a node-embedding export.

Reads an embedding table (``.npy`` of shape ``[num_nodes, d]``, as written by
``--export-best-embedding-table``), projects it to the 2D unit disk, and colours
each node by its degree on a log scale. The output is a bare figure: just the
points and a ``degree`` colourbar -- no title, axes, or border.

Projection: the radial coordinate is the true depth ``||E||`` (distance from the
origin -- the hierarchy axis on the Poincaré ball); the angle comes from a 2D PCA
of the embeddings. Non-isolated nodes are sampled with a mild degree bias so the
(rarer) hubs stay visible.

Degrees are read from the dataset edge CSV (columns ``src``, ``dst``), inferred
from the embedding filename (e.g. ``GoogleLocal_seed42_demb32_ep31.npy`` -> the
``GoogleLocal`` dataset under ``--tgb-root``) unless ``--csv`` is given.

Usage:
    python plots/ball_plot.py EMBEDDING.npy OUTPUT.png [--csv CSV]
                              [--tgb-root datasets] [--n-sample 6000] [--seed 1]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA

# Preferred colourbar tick positions (kept only where they fall inside the data range).
_CANDIDATE_TICKS = [3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000]


def infer_csv_path(embedding_path: str, tgb_root: str) -> str:
    """Map ``<Dataset>_seed..._.npy`` to ``<tgb_root>/<Dataset>/<Dataset>.csv``."""
    dataset = os.path.basename(embedding_path).split("_seed")[0]
    return os.path.join(tgb_root, dataset, f"{dataset}.csv")


def node_degrees(csv_path: str, num_nodes: int) -> np.ndarray:
    """Undirected degree per node id from a ``src,dst`` edge CSV."""
    edges = pd.read_csv(csv_path, usecols=["src", "dst"])
    deg = np.zeros(num_nodes, dtype=np.int64)
    np.add.at(deg, edges["src"].to_numpy(), 1)
    np.add.at(deg, edges["dst"].to_numpy(), 1)
    return deg


def ball_plot(embedding: np.ndarray, degree: np.ndarray, output: str,
              n_sample: int = 6000, seed: int = 1) -> None:
    """Write the minimal disk figure: radial = ||E||, angle = PCA-2, colour = degree (log)."""
    nodes = np.flatnonzero(degree > 0)
    if nodes.size == 0:
        raise ValueError("No non-isolated nodes to plot (all degrees are zero).")

    rng = np.random.default_rng(seed)
    weights = degree[nodes].astype(float) ** 0.6            # mild hub over-sampling
    weights /= weights.sum()
    n = min(n_sample, nodes.size)
    sample = rng.choice(nodes, size=n, replace=False, p=weights)

    radius = np.linalg.norm(embedding[sample], axis=1)      # depth = distance from the origin
    angle = PCA(n_components=2).fit_transform(embedding[sample])
    theta = np.arctan2(angle[:, 1], angle[:, 0])
    x, y = radius * np.cos(theta), radius * np.sin(theta)

    deg = degree[sample]
    order = np.argsort(deg)                                 # draw hubs last, on top

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(x[order], y[order], c=deg[order], cmap="turbo",
                    norm=LogNorm(vmin=deg.min(), vmax=deg.max()),
                    s=7, alpha=0.85, linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")

    cbar = fig.colorbar(sc, shrink=0.7)
    cbar.set_label("degree")
    ticks = [t for t in _CANDIDATE_TICKS if deg.min() <= t <= deg.max()]
    if ticks:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])       # plain integers, not 10^x
    cbar.minorticks_off()

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("embedding", help="Embedding export (.npy, [num_nodes, d]).")
    ap.add_argument("output", help="Output image path (.png).")
    ap.add_argument("--csv", default=None,
                    help="Edge CSV with src,dst columns (default: inferred from the embedding name).")
    ap.add_argument("--tgb-root", default="datasets",
                    help="Root holding <dataset>/<dataset>.csv, used when --csv is not given.")
    ap.add_argument("--n-sample", type=int, default=6000,
                    help="Number of non-isolated nodes to plot (degree-weighted sample).")
    ap.add_argument("--seed", type=int, default=1, help="Sampling seed.")
    args = ap.parse_args()

    embedding = np.load(args.embedding)
    csv_path = args.csv or infer_csv_path(args.embedding, args.tgb_root)
    degree = node_degrees(csv_path, embedding.shape[0])

    ball_plot(embedding, degree, args.output, n_sample=args.n_sample, seed=args.seed)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
