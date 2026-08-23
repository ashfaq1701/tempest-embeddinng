"""Render one training-dynamics animation frame from an embedding snapshot.

A frame shows an embedding snapshot on the disk (radial = depth ``||E||``, angle = PCA direction),
coloured by a per-node value (e.g. degree), with a two-line label:
``batch N  (epoch X.X)`` / ``Mean Reciprocal Rank: M``.

For a *smooth* animation the projection must be consistent across frames, so ``build_frame_context``
fixes the node sample, the PCA basis (fit on a reference snapshot -- typically the final one), the
colour scale, and the axis extent once, and returns a reusable Matplotlib figure inside a
``FrameContext``. ``ball_frame_plot(ctx, snapshot_path, mrr, epoch, batch)`` then only updates the point
positions and the label per frame and returns the rendered RGB frame (uint8, even dims for H.264).
"""
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA

from plots.ball_plot import disk_coords

_TICK_CANDIDATES = [3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000]


@dataclass
class FrameContext:
    """Fixed projection + reusable figure shared by every frame (built once)."""
    sample: np.ndarray          # sampled node indices (same across frames)
    order: np.ndarray           # draw order over the sample (high-value points on top)
    pca: PCA                    # fixed 2D PCA basis (fit on the reference snapshot)
    rmax: float                 # fixed axis half-extent
    fig: "plt.Figure"
    scatter: "plt.Artist"
    text: "plt.Text"


def build_frame_context(reference_snapshot: str, values: np.ndarray,
                        n_sample: int = 6000, seed: int = 1,
                        cmap: str = "turbo", value_label: str = "degree") -> FrameContext:
    """Set up the reusable figure and the FIXED projection basis for all frames.

    ``reference_snapshot`` (path) defines the PCA basis and the axis extent -- pass the FINAL snapshot
    so the fully-organised layout frames well. ``values`` is a per-node array to colour by (length =
    num_nodes); only nodes with ``values > 0`` are eligible, sampled with a mild bias so high-value
    nodes stay visible. Colours use a log scale with plain-integer ticks.
    """
    embedding = np.load(reference_snapshot)
    values = np.asarray(values)

    eligible = np.flatnonzero(values > 0)
    if eligible.size == 0:
        raise ValueError("No nodes with positive `values` to plot.")
    weights = values[eligible].astype(float) ** 0.6
    weights /= weights.sum()
    n = min(n_sample, eligible.size)
    rng = np.random.default_rng(seed)
    sample = rng.choice(eligible, size=n, replace=False, p=weights)
    sample_vals = values[sample]
    order = np.argsort(sample_vals)                          # high-value points drawn last, on top

    pca = PCA(n_components=2).fit(embedding[sample])
    rmax = float(np.linalg.norm(embedding[sample], axis=1).max() * 1.05)

    # Manual layout: near-square plot box (no wasted vertical space) + a colourbar strip.
    fig = plt.figure(figsize=(9.5, 8), dpi=110)
    ax = fig.add_axes([0.02, 0.03, 0.792, 0.94])
    cax = fig.add_axes([0.84, 0.22, 0.025, 0.56])

    coords, _, _ = disk_coords(embedding[sample], pca)
    norm = LogNorm(vmin=max(int(sample_vals.min()), 1), vmax=int(sample_vals.max()))
    scatter = ax.scatter(coords[order, 0], coords[order, 1], c=sample_vals[order], cmap=cmap,
                         norm=norm, s=7, alpha=0.85, linewidths=0)
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_aspect("equal")
    ax.axis("off")

    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label(value_label)
    ticks = [t for t in _TICK_CANDIDATES if sample_vals.min() <= t <= sample_vals.max()]
    if ticks:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])        # plain integers, not 10^x
    cbar.minorticks_off()

    text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=11, color="0.3")
    return FrameContext(sample=sample, order=order, pca=pca, rmax=rmax,
                        fig=fig, scatter=scatter, text=text)


def ball_frame_plot(ctx: FrameContext, snapshot_path: str,
                    mrr: Optional[float], epoch: float, batch: int) -> np.ndarray:
    """Render one frame from ``snapshot_path`` with the given label. Returns an RGB uint8 array."""
    embedding = np.load(snapshot_path)
    coords, _, _ = disk_coords(embedding[ctx.sample], ctx.pca)
    ctx.scatter.set_offsets(coords[ctx.order])

    mrr_str = f"{mrr:.3f}" if mrr is not None else "—"
    ctx.text.set_text(f"batch {int(batch)}   (epoch {epoch:.1f})\n"
                      f"Mean Reciprocal Rank: {mrr_str}")

    ctx.fig.canvas.draw()
    buf = np.asarray(ctx.fig.canvas.buffer_rgba())[..., :3]
    h, w = buf.shape[:2]
    return np.ascontiguousarray(buf[:h // 2 * 2, :w // 2 * 2])   # even dims for libx264
