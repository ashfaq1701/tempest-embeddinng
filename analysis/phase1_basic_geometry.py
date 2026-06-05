"""Phase 1 — basic geometric structure of E.

Inputs:
    logs/embeddings/tgbl-wiki_seed42_demb128_ep33.npy   (9227, 128) float32

Outputs (analysis/phase1/):
    stats.json          — summary scalars
    norm_hist.npy       — histogram of L2 norms (bins, counts)
    spectrum.npy        — singular values of centered E (descending)
    cumvar.npy          — cumulative explained variance
    pairwise_cos.npy    — empirical cosine sim distribution (random pair sample)
    pairwise_cos_active.npy  — same restricted to nodes seen in training

Tells us:
    - is E concentrated near a sphere or spread through the ball
    - effective dim (95% var) vs nominal 128
    - global anisotropy (top singular vs mean)
    - typical inter-node cosine (isotropy proxy) — close to 0 = isotropic,
      far from 0 = collapsed
"""
import json
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EMB_PATH = ROOT / "logs" / "embeddings" / "tgbl-wiki_seed42_demb128_ep33.npy"
OUT = ROOT / "analysis" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)


def load_active_node_set():
    """Nodes that actually appear in the TGB wiki training split.
    Inactive rows of E never received gradient and bias the geometry
    summaries if included naively."""
    from tempest_walks.data import load_tgb
    loaded = load_tgb("tgbl-wiki", root=str(ROOT / "datasets"))
    train = loaded.train
    active = np.unique(np.concatenate([train.sources, train.destinations]))
    return active.astype(np.int64), loaded


def main():
    E = np.load(EMB_PATH).astype(np.float32)
    N, D = E.shape
    print(f"E shape={E.shape}  dtype={E.dtype}")
    active_nodes, loaded = load_active_node_set()
    print(f"active (train) nodes: {len(active_nodes)} / {N}")

    # --- Norms ----------------------------------------------------------
    norms = np.linalg.norm(E, axis=1)
    norms_active = norms[active_nodes]
    inactive_mask = np.ones(N, dtype=bool); inactive_mask[active_nodes] = False
    norms_inactive = norms[inactive_mask]

    print(f"\n--- L2 norms ---")
    print(f"  ALL      mean={norms.mean():.3f}  std={norms.std():.3f}  "
          f"min={norms.min():.3f}  max={norms.max():.3f}")
    print(f"  active   mean={norms_active.mean():.3f}  std={norms_active.std():.3f}")
    print(f"  inactive mean={norms_inactive.mean():.3f}  std={norms_inactive.std():.3f}  "
          f"n_inactive={len(norms_inactive)}")

    # Histograms
    bins = np.linspace(0, max(2.5, float(norms.max()) + 0.1), 80)
    h_active, _ = np.histogram(norms_active, bins=bins)
    h_inactive, _ = np.histogram(norms_inactive, bins=bins) if len(norms_inactive) else (np.zeros(len(bins)-1, dtype=int), bins)
    np.save(OUT / "norm_hist.npy",
            {"bins": bins, "active": h_active, "inactive": h_inactive}, allow_pickle=True)

    # --- Centered SVD on ACTIVE only (real geometry) --------------------
    E_active = E[active_nodes].astype(np.float64)  # for numerical safety
    mu = E_active.mean(axis=0)
    Xc = E_active - mu
    # Singular values of Xc -> sqrt(eigenvalues of cov * (n-1))
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2 / (len(active_nodes) - 1)
    var_norm = var / var.sum()
    cumvar = np.cumsum(var_norm)
    # Effective dim @ 90/95/99
    k90 = int(np.searchsorted(cumvar, 0.90) + 1)
    k95 = int(np.searchsorted(cumvar, 0.95) + 1)
    k99 = int(np.searchsorted(cumvar, 0.99) + 1)
    # Participation ratio: (Σ λ)^2 / Σ λ^2  — a soft effective-dim measure
    part_ratio = float((var.sum() ** 2) / (var * var).sum())
    # Top-1 explained
    top1 = float(var_norm[0])
    np.save(OUT / "spectrum.npy", S)
    np.save(OUT / "cumvar.npy", cumvar)

    print(f"\n--- Centered SVD on active rows ---")
    print(f"  D_eff @ 90% var:  {k90}")
    print(f"  D_eff @ 95% var:  {k95}")
    print(f"  D_eff @ 99% var:  {k99}")
    print(f"  participation_ratio: {part_ratio:.2f}  (nominal D={D})")
    print(f"  top-1 fraction of variance: {top1:.4f}")
    print(f"  top-5 cumulative: {cumvar[4]:.4f}")
    print(f"  top-10 cumulative: {cumvar[9]:.4f}")

    # --- Pairwise cosine sample (isotropy proxy) ------------------------
    rng = np.random.default_rng(0)
    n_pairs = 200_000
    # Active-only sample (uniform with replacement, exclude self-pairs)
    a_idx = rng.choice(len(active_nodes), size=n_pairs, replace=True)
    b_idx = rng.choice(len(active_nodes), size=n_pairs, replace=True)
    keep = a_idx != b_idx
    a_idx = active_nodes[a_idx[keep]]
    b_idx = active_nodes[b_idx[keep]]
    Ea = E[a_idx]; Eb = E[b_idx]
    cos_ab = (Ea * Eb).sum(axis=1) / (
        np.linalg.norm(Ea, axis=1) * np.linalg.norm(Eb, axis=1) + 1e-12)
    np.save(OUT / "pairwise_cos_active.npy", cos_ab.astype(np.float32))

    print(f"\n--- Pairwise cosine (random active pairs, n={len(cos_ab):,}) ---")
    print(f"  mean={cos_ab.mean():.4f}  median={np.median(cos_ab):.4f}  "
          f"std={cos_ab.std():.4f}")
    print(f"  P10={np.percentile(cos_ab, 10):.4f}  P90={np.percentile(cos_ab, 90):.4f}")
    pct_above_05 = float((cos_ab > 0.5).mean())
    pct_above_08 = float((cos_ab > 0.8).mean())
    pct_below_0 = float((cos_ab < 0.0).mean())
    print(f"  P(cos > 0.5)={pct_above_05:.4f}  P(cos > 0.8)={pct_above_08:.4f}  "
          f"P(cos < 0)={pct_below_0:.4f}")

    # All-pair (active vs active) is too large; do a chunked sample of u-anchored
    # cosines for a few sample u's to inspect shapes.
    sample_u = rng.choice(active_nodes, size=10, replace=False)
    u_norms = np.linalg.norm(E[sample_u], axis=1, keepdims=True)
    pop_norms = np.linalg.norm(E[active_nodes], axis=1, keepdims=True)
    cos_u_all = (E[sample_u] @ E[active_nodes].T) / (u_norms * pop_norms.T + 1e-12)
    np.save(OUT / "sample_u_cosines.npy",
            {"u_ids": sample_u, "cos": cos_u_all.astype(np.float32),
             "active_nodes": active_nodes}, allow_pickle=True)
    print(f"  sample_u anchored cos shape: {cos_u_all.shape}")
    print(f"  per-u max cos to any other active node: "
          f"{np.array([np.partition(cos_u_all[i], -2)[-2] for i in range(len(sample_u))])}")

    # --- Effective rank via stable rank: ||X||_F^2 / ||X||_2^2 ----------
    fro_sq = float((S * S).sum())
    spec_sq = float(S[0] ** 2)
    stable_rank = fro_sq / spec_sq
    print(f"\n--- Stable rank (Fro^2 / spec^2) ---")
    print(f"  stable_rank = {stable_rank:.2f}  (≤ D={D})")

    # --- Anisotropy: ratio largest to mean ------------------------------
    aniso = float(var[0] / var.mean())
    print(f"  anisotropy (var[0] / mean(var)) = {aniso:.2f}")

    # --- Save summary ---------------------------------------------------
    stats = {
        "shape": [int(N), int(D)],
        "n_active": int(len(active_nodes)),
        "n_inactive": int(len(norms_inactive)),
        "norms": {
            "all":      {"mean": float(norms.mean()), "std": float(norms.std()),
                         "min":  float(norms.min()),  "max": float(norms.max())},
            "active":   {"mean": float(norms_active.mean()), "std": float(norms_active.std())},
            "inactive": {"mean": float(norms_inactive.mean()) if len(norms_inactive) else None,
                         "std":  float(norms_inactive.std())  if len(norms_inactive) else None},
        },
        "effective_dim": {
            "D_eff_90": k90, "D_eff_95": k95, "D_eff_99": k99,
            "participation_ratio": part_ratio,
            "stable_rank": stable_rank,
            "anisotropy_top_over_mean": aniso,
            "top1_var_fraction": top1,
            "top5_var_cumulative": float(cumvar[4]),
            "top10_var_cumulative": float(cumvar[9]),
        },
        "pairwise_cos_active": {
            "n_pairs": int(len(cos_ab)),
            "mean":   float(cos_ab.mean()),
            "median": float(np.median(cos_ab)),
            "std":    float(cos_ab.std()),
            "P10":    float(np.percentile(cos_ab, 10)),
            "P90":    float(np.percentile(cos_ab, 90)),
            "P_above_0.5": pct_above_05,
            "P_above_0.8": pct_above_08,
            "P_below_0":   pct_below_0,
        },
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nstats.json -> {OUT/'stats.json'}")


if __name__ == "__main__":
    main()
