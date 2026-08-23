"""Headless 2D Poincaré-disk plot of a single embedding snapshot.

``ball_plot(snapshot_path)`` loads an embedding table (``.npy`` of shape ``[num_nodes, d]``) and
scatters it on a disk: the radial coordinate is the node's depth ``||E||`` (distance from the origin,
the hierarchy axis on the Poincaré ball) and the angle is a 2D PCA of the embeddings. Points are
coloured by depth. No graph, degrees, or labels are needed -- just the geometry. Returns a Figure.

The ``disk_coords`` helper (radial ``||E||`` + PCA angle) is shared with the animation frame renderer.

CLI:
    python plots/ball_plot.py SNAPSHOT.npy [OUTPUT.png]
"""
import argparse
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


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


def ball_plot(snapshot_path: str, n_sample: int = 6000, seed: int = 1) -> "plt.Figure":
    """Headless disk plot of one embedding snapshot, coloured by depth ``||E||``. Returns a Figure."""
    embedding = np.load(snapshot_path)
    rng = np.random.default_rng(seed)
    n = min(n_sample, embedding.shape[0])
    sample = rng.choice(embedding.shape[0], size=n, replace=False)

    coords, radius, _ = disk_coords(embedding[sample])
    order = np.argsort(-radius)                              # draw shallow (core) points on top

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(coords[order, 0], coords[order, 1], c=radius[order], cmap="viridis",
                    s=7, alpha=0.85, linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    cbar = fig.colorbar(sc, shrink=0.7)
    cbar.set_label("depth  ||E||")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description="Headless Poincaré-disk plot of one embedding snapshot.")
    ap.add_argument("snapshot", help="Embedding snapshot (.npy, [num_nodes, d]).")
    ap.add_argument("output", nargs="?", default=None,
                    help="Output image path (default: the snapshot path with a .png suffix).")
    args = ap.parse_args()

    output = args.output or (args.snapshot.rsplit(".", 1)[0] + ".png")
    fig = ball_plot(args.snapshot)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
