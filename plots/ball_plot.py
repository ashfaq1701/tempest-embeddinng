"""Headless 2D Poincaré-disk plot of a single embedding snapshot, coloured by node degree.

``ball_plot(snapshot_path, csv_path)`` loads an embedding table (``.npy`` of shape ``[num_nodes, d]``)
and scatters it on a disk: the radial coordinate is the node's depth ``||E||`` (distance from the
origin, the hierarchy axis on the Poincaré ball) and the angle is a 2D PCA of the embeddings. Points
are coloured by node **degree** (log scale), computed from the edge CSV. A correct hierarchy shows the
high-degree hubs at small radius (disk centre) and low-degree leaves on the rim. The sample is biased
toward high-degree nodes so the rare hubs stay visible. Returns a Figure.

The ``disk_coords`` helper (radial ``||E||`` + PCA angle) is shared with the animation frame renderer.

CLI:
    python plots/ball_plot.py SNAPSHOT.npy CSV [OUTPUT.png]
"""
import argparse
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA

# Round, plain-integer colourbar ticks spanning a wide degree range on a log scale.
_TICK_CANDIDATES = [3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 50000]


def disk_coords(embedding: np.ndarray,
                pca: Optional[PCA] = None) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """Project ``[n, d]`` embeddings to disk coordinates.

    Radial coordinate = true depth ``||E||``; angle = 2D PCA direction. Pass a fitted ``pca`` to reuse
    a fixed basis across snapshots (needed for a smooth animation); otherwise one is fit here.
    Returns ``(coords [n, 2], radius [n], pca)``.
    """
    radius = np.linalg.norm(embedding, axis=1)
    if pca is None:
        pca = PCA(n_components=2).fit(embedding)
    ab = pca.transform(embedding)
    theta = np.arctan2(ab[:, 1], ab[:, 0])
    return np.c_[radius * np.cos(theta), radius * np.sin(theta)], radius, pca


def node_degrees(csv_path: str, num_nodes: int) -> np.ndarray:
    """Undirected degree per node id from a ``src,dst`` edge CSV (+1 per endpoint, all edges)."""
    edges = pd.read_csv(csv_path, usecols=["src", "dst"])
    deg = np.zeros(num_nodes, dtype=np.int64)
    np.add.at(deg, edges["src"].to_numpy(), 1)
    np.add.at(deg, edges["dst"].to_numpy(), 1)
    return deg


def ball_plot(snapshot_path: str, csv_path: str,
              n_sample: int = 6000, seed: int = 1) -> "plt.Figure":
    """Headless disk plot of one embedding snapshot, coloured by node degree (log scale). Returns a Figure."""
    embedding = np.load(snapshot_path)
    degree = node_degrees(csv_path, embedding.shape[0])

    # Sample only nodes that appear in the graph, biased toward high degree so the rare hubs show.
    eligible = np.flatnonzero(degree > 0)
    weights = degree[eligible].astype(float) ** 0.6
    weights /= weights.sum()
    n = min(n_sample, eligible.size)
    rng = np.random.default_rng(seed)
    sample = rng.choice(eligible, size=n, replace=False, p=weights)
    sample_deg = degree[sample]
    order = np.argsort(sample_deg)                          # draw high-degree hubs last, on top

    coords, _, _ = disk_coords(embedding[sample])
    norm = LogNorm(vmin=max(int(sample_deg.min()), 1), vmax=int(sample_deg.max()))

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(coords[order, 0], coords[order, 1], c=sample_deg[order], cmap="turbo",
                    norm=norm, s=7, alpha=0.85, linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")

    cbar = fig.colorbar(sc, shrink=0.7)
    cbar.set_label("degree")
    ticks = [t for t in _TICK_CANDIDATES if sample_deg.min() <= t <= sample_deg.max()]
    if ticks:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])        # plain integers, not 10^x
    cbar.minorticks_off()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Headless Poincaré-disk plot of one embedding snapshot, coloured by node degree.")
    ap.add_argument("snapshot", help="Embedding snapshot (.npy, [num_nodes, d]).")
    ap.add_argument("csv", help="Edge CSV with src,dst columns (for node degrees).")
    ap.add_argument("output", nargs="?", default=None,
                    help="Output image path (default: the snapshot path with a .png suffix).")
    args = ap.parse_args()

    output = args.output or (args.snapshot.rsplit(".", 1)[0] + ".png")
    fig = ball_plot(args.snapshot, args.csv)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
