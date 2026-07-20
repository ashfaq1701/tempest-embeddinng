"""Embedding-table diagnostic analysis.

Reads a dumped embedding table (.npy) and produces a quantitative
report covering six tiers of behaviour we expect a well-trained
temporal embedding to exhibit.

Usage:
    python scripts/analyze_embedding.py \
        --emb-path logs/embeddings/tgbl-wiki_seed42_demb128_ep45.npy \
        --dataset tgbl-wiki

Tier overview:
  Tier 1 — Basic geometry            (norms, intrinsic dim, isotropy)
  Tier 2 — Pair-level closeness      (interaction, recency, count, per-anchor)
  Tier 3 — Network-structural        (common neighbour, Jaccard, triangle)
  Tier 4 — Strength gradient         (count quantile monotonicity, burstiness)
  Tier 5 — Sanity / negatives        (far-pair asymptote, cold-start)
  Tier 6 — Temporal probes           (per-anchor recency, time prediction)

All metrics are computed on the L2-normalised embedding (cosine =
inner product). For each test the report quotes a single quantitative
summary plus a pass/fail/inconclusive interpretation against a
configurable noise band.
"""

import argparse
import pathlib
import random
import sys
from collections import defaultdict

import numpy as np

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from link_property_prediction.data import load_tgb


# ─── CLI ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--emb-path", required=True, type=str,
                   help="Path to the dumped embedding table (.npy).")
    p.add_argument("--dataset", required=True, type=str,
                   help="TGB dataset name (used to load train edges).")
    p.add_argument("--tgb-root", default="datasets", type=str)
    p.add_argument("--seed", default=42, type=int,
                   help="Numpy/random seed for the sampling-based tests.")
    p.add_argument("--n-sample", default=2000, type=int,
                   help="Sample size for pair-based tests.")
    p.add_argument("--time-probe-edges", default=20000, type=int,
                   help="How many train edges to use for the time-linear probe.")
    p.add_argument("--linkpred-edges", default=5000, type=int,
                   help="Test edges to sample for raw-E MRR (Tier 7).")
    p.add_argument("--linkpred-negs", default=100, type=int,
                   help="Random negatives per test edge for raw-E MRR.")
    return p.parse_args()


# ─── helpers ────────────────────────────────────────────────────────────


def l2_normalise(E: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalise. Cosine sim then equals inner product."""
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return E / n


def build_graph(train) -> dict:
    """Return adjacency + per-pair {count, last_t, first_t}.

    Undirected: pair key is (min(u, v), max(u, v)).
    """
    src = np.asarray(train.sources, dtype=np.int64)
    tgt = np.asarray(train.destinations, dtype=np.int64)
    ts = np.asarray(train.timestamps, dtype=np.int64)

    adj = defaultdict(set)
    pair_count: dict = {}
    pair_last: dict = {}
    pair_first: dict = {}
    pair_times: dict = defaultdict(list)
    node_last_t: dict = {}
    node_first_t: dict = {}
    node_degree: dict = defaultdict(int)

    for u, v, t in zip(src, tgt, ts):
        if u == v:
            continue
        a, b = (int(u), int(v)) if u < v else (int(v), int(u))
        adj[a].add(b)
        adj[b].add(a)
        pair_count[(a, b)] = pair_count.get((a, b), 0) + 1
        last_t = pair_last.get((a, b), -1)
        if int(t) > last_t:
            pair_last[(a, b)] = int(t)
        first_t = pair_first.get((a, b), 1 << 62)
        if int(t) < first_t:
            pair_first[(a, b)] = int(t)
        pair_times[(a, b)].append(int(t))
        if int(t) > node_last_t.get(int(u), -1):
            node_last_t[int(u)] = int(t)
        if int(t) > node_last_t.get(int(v), -1):
            node_last_t[int(v)] = int(t)
        if int(t) < node_first_t.get(int(u), 1 << 62):
            node_first_t[int(u)] = int(t)
        if int(t) < node_first_t.get(int(v), 1 << 62):
            node_first_t[int(v)] = int(t)
        node_degree[int(u)] += 1
        node_degree[int(v)] += 1

    return {
        "adj": adj,
        "pair_count": pair_count,
        "pair_last": pair_last,
        "pair_first": pair_first,
        "pair_times": dict(pair_times),
        "node_last_t": node_last_t,
        "node_first_t": node_first_t,
        "node_degree": node_degree,
        "t_min": int(ts.min()),
        "t_max": int(ts.max()),
        "n_edges": int(len(src)),
        "active_nodes": sorted(node_degree.keys()),
    }


def cos_pairs(E: np.ndarray, pairs) -> np.ndarray:
    """Return cos(E[a], E[b]) for a list of (a, b) pairs (E expected unit-norm)."""
    if not pairs:
        return np.zeros(0)
    a_idx = np.asarray([p[0] for p in pairs])
    b_idx = np.asarray([p[1] for p in pairs])
    return (E[a_idx] * E[b_idx]).sum(axis=1)


def sample_random_pairs(active, n, rng):
    """Sample n distinct (a, b) pairs uniformly from active nodes, a != b."""
    out = []
    while len(out) < n:
        a, b = int(rng.choice(active)), int(rng.choice(active))
        if a == b:
            continue
        out.append((a, b) if a < b else (b, a))
    return out


def sample_non_interacting_pairs(active, adj, n, rng, also_no_common_nbr=False):
    """Sample n pairs (a, b) with no train edge. Optionally also no common nbr."""
    out = []
    tries = 0
    max_tries = n * 50
    while len(out) < n and tries < max_tries:
        tries += 1
        a, b = int(rng.choice(active)), int(rng.choice(active))
        if a == b:
            continue
        a, b = (a, b) if a < b else (b, a)
        if b in adj[a]:
            continue
        if also_no_common_nbr and (adj[a] & adj[b]):
            continue
        out.append((a, b))
    return out


def sample_interacting_pairs(pair_count, n, rng):
    """Sample n pairs uniformly from the keys of pair_count."""
    keys = list(pair_count.keys())
    if not keys:
        return []
    idx = rng.choice(len(keys), size=min(n, len(keys)), replace=False)
    return [keys[i] for i in idx]


def sample_common_neighbour_pairs(adj, active, n, rng, no_direct=True):
    """Sample n (a, b) with a common neighbour c.
    By default we require no direct (a, b) edge."""
    out = []
    tries = 0
    max_tries = n * 100
    active_set = set(active)
    while len(out) < n and tries < max_tries:
        tries += 1
        a = int(rng.choice(active))
        if not adj[a]:
            continue
        c = random.sample(list(adj[a]), 1)[0]
        if not adj[c]:
            continue
        b = random.sample(list(adj[c]), 1)[0]
        if b == a or b not in active_set:
            continue
        a2, b2 = (a, b) if a < b else (b, a)
        if no_direct and (b2 in adj[a2]):
            continue
        out.append((a2, b2))
    return out


def mean_std(arr):
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


# ─── tier implementations ──────────────────────────────────────────────


def tier1_geometry(E_raw, E_norm):
    n, d = E_raw.shape
    norms = np.linalg.norm(E_raw, axis=1)

    # SVD-based intrinsic dim: (Σσ)² / Σσ². Bounded in [1, min(N, d)].
    s = np.linalg.svd(E_raw, full_matrices=False, compute_uv=False)
    intrinsic_dim = float((s.sum() ** 2) / max(float((s ** 2).sum()), 1e-12))

    # Isotropy: |cos| over random pairs.
    rng = np.random.default_rng(0)
    pairs = [(int(rng.integers(n)), int(rng.integers(n))) for _ in range(5000)]
    pairs = [(a, b) for (a, b) in pairs if a != b]
    cos = cos_pairs(E_norm, pairs)
    iso_mean = float(np.mean(np.abs(cos)))
    iso_std = float(np.std(cos))

    return {
        "n_nodes": n,
        "d_emb": d,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_p5_p95": (float(np.percentile(norms, 5)),
                       float(np.percentile(norms, 95))),
        "intrinsic_dim": intrinsic_dim,
        "iso_cos_abs_mean": iso_mean,
        "iso_cos_std": iso_std,
    }


def tier2_pair_closeness(E_norm, G, n_sample, rng):
    adj = G["adj"]
    pair_count = G["pair_count"]
    pair_last = G["pair_last"]
    active = G["active_nodes"]
    t_max = G["t_max"]
    t_min = G["t_min"]

    # (i) Interacting vs non-interacting.
    int_pairs = sample_interacting_pairs(pair_count, n_sample, rng)
    non_int_pairs = sample_non_interacting_pairs(active, adj, n_sample, rng)
    cos_int_mean, cos_int_std = mean_std(cos_pairs(E_norm, int_pairs))
    cos_non_mean, cos_non_std = mean_std(cos_pairs(E_norm, non_int_pairs))

    # (ii) Recency quartiles over interacting pairs.
    if pair_last:
        pairs_l = list(pair_last.keys())
        last_ts = np.asarray([pair_last[p] for p in pairs_l])
        rel = (last_ts - t_min) / max(t_max - t_min, 1)
        quartiles = np.percentile(rel, [25, 50, 75])
        cos_by_q = []
        for lo, hi in [(-0.01, quartiles[0]), (quartiles[0], quartiles[1]),
                       (quartiles[1], quartiles[2]), (quartiles[2], 1.01)]:
            mask = (rel >= lo) & (rel < hi)
            sel = [pairs_l[i] for i in np.where(mask)[0][:n_sample]]
            cos_by_q.append(mean_std(cos_pairs(E_norm, sel))[0])
    else:
        cos_by_q = [float("nan")] * 4

    # (iii) Count quartiles over interacting pairs.
    if pair_count:
        pairs_l = list(pair_count.keys())
        counts = np.asarray([pair_count[p] for p in pairs_l])
        q = np.unique(np.percentile(counts, [25, 50, 75]))
        # If counts are very degenerate (most pairs have count 1) the
        # quartile cuts may collapse; we use unique cut points instead.
        cuts = [0] + list(q) + [counts.max() + 1]
        cos_by_count = []
        labels = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            mask = (counts > lo) & (counts <= hi)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                cos_by_count.append(float("nan"))
                labels.append(f"({lo},{hi}]")
                continue
            sel = [pairs_l[i] for i in idx[:n_sample]]
            cos_by_count.append(mean_std(cos_pairs(E_norm, sel))[0])
            labels.append(f"({lo},{hi}]")
    else:
        cos_by_count = []
        labels = []

    # (iv) Per-anchor recency gradient: top-degree anchors.
    deg = G["node_degree"]
    top_anchors = sorted(deg, key=deg.get, reverse=True)[:30]
    spearmans = []
    for a in top_anchors:
        partners = list(adj[a])
        if len(partners) < 8:
            continue
        last_ts = np.asarray(
            [pair_last[(min(a, b), max(a, b))] for b in partners],
        )
        cos_vals = (E_norm[a] * E_norm[partners]).sum(axis=1)
        # Spearman = Pearson on rank-transformed.
        if len(partners) < 2:
            continue
        from scipy.stats import spearmanr
        r, _ = spearmanr(last_ts, cos_vals)
        if np.isfinite(r):
            spearmans.append(float(r))
    spearman_mean = float(np.mean(spearmans)) if spearmans else float("nan")
    spearman_std = float(np.std(spearmans)) if spearmans else float("nan")

    return {
        "i_interacting_cos": cos_int_mean,
        "i_interacting_std": cos_int_std,
        "i_noninteracting_cos": cos_non_mean,
        "i_noninteracting_std": cos_non_std,
        "i_delta": cos_int_mean - cos_non_mean,
        "ii_recency_quartiles": cos_by_q,
        "ii_recency_monotone": all(
            (np.isnan(cos_by_q[k]) or np.isnan(cos_by_q[k + 1])
             or cos_by_q[k] <= cos_by_q[k + 1] + 1e-3)
            for k in range(len(cos_by_q) - 1)
        ),
        "iii_count_bins": list(zip(labels, cos_by_count)),
        "iv_per_anchor_recency_spearman_mean": spearman_mean,
        "iv_per_anchor_recency_spearman_std": spearman_std,
        "iv_n_anchors": len(spearmans),
    }


def tier3_structural(E_norm, G, n_sample, rng):
    adj = G["adj"]
    active = G["active_nodes"]

    # (v) Common-neighbour pairs without direct edge.
    cn_pairs = sample_common_neighbour_pairs(adj, active, n_sample, rng,
                                             no_direct=True)
    non_cn_pairs = sample_non_interacting_pairs(active, adj, n_sample, rng,
                                                also_no_common_nbr=True)
    cn_cos = float(np.mean(cos_pairs(E_norm, cn_pairs))) if cn_pairs else float("nan")
    non_cn_cos = (float(np.mean(cos_pairs(E_norm, non_cn_pairs)))
                  if non_cn_pairs else float("nan"))

    # (vi) Jaccard ~ cos correlation over random sample of pairs.
    pairs_v = sample_random_pairs(active, n_sample, rng)
    jaccards = []
    cos_vals = []
    for a, b in pairs_v:
        na, nb = adj[a], adj[b]
        if not na or not nb:
            continue
        inter = len(na & nb)
        union = len(na | nb)
        if union == 0:
            continue
        jaccards.append(inter / union)
        cos_vals.append(float((E_norm[a] * E_norm[b]).sum()))
    if len(jaccards) > 1:
        jaccard_corr = float(np.corrcoef(jaccards, cos_vals)[0, 1])
    else:
        jaccard_corr = float("nan")

    # (x) Triangle inequality on chains a-b-c with a not~ c.
    triangle_hits = 0
    triangle_total = 0
    tries = 0
    rng_local = random.Random(0)
    while triangle_total < n_sample and tries < n_sample * 50:
        tries += 1
        a = rng_local.choice(active)
        if not adj[a]:
            continue
        b = rng_local.choice(list(adj[a]))
        if not adj[b]:
            continue
        c = rng_local.choice(list(adj[b]))
        if c == a or c in adj[a]:
            continue
        triangle_total += 1
        c_ab = float((E_norm[a] * E_norm[b]).sum())
        c_bc = float((E_norm[b] * E_norm[c]).sum())
        c_ac = float((E_norm[a] * E_norm[c]).sum())
        # Passes if cos(a, c) is between random baseline (≈0) and the
        # minimum of the direct edges.
        if 0.0 < c_ac < min(c_ab, c_bc) + 0.01:
            triangle_hits += 1

    triangle_pass_rate = (triangle_hits / triangle_total
                          if triangle_total > 0 else float("nan"))

    return {
        "v_common_nbr_cos": cn_cos,
        "v_no_common_nbr_cos": non_cn_cos,
        "v_delta": cn_cos - non_cn_cos if not np.isnan(cn_cos) else float("nan"),
        "vi_jaccard_cos_corr": jaccard_corr,
        "vi_n_used": len(jaccards),
        "x_triangle_pass_rate": triangle_pass_rate,
        "x_triangle_total": triangle_total,
    }


def tier4_strength(E_norm, G, n_sample, rng):
    pair_count = G["pair_count"]
    pair_times = G["pair_times"]

    # (xi) Count quantile monotonicity, finer grain.
    if pair_count:
        pairs_l = list(pair_count.keys())
        counts = np.asarray([pair_count[p] for p in pairs_l])
        # Use deciles where possible.
        cuts = np.unique(np.percentile(counts, np.arange(10, 100, 10)))
        cuts = np.concatenate([[0], cuts, [counts.max() + 1]])
        bin_cos = []
        bin_labels = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            idx = np.where((counts > lo) & (counts <= hi))[0]
            if len(idx) == 0:
                continue
            sel = [pairs_l[i] for i in idx[:n_sample]]
            bin_cos.append(float(np.mean(cos_pairs(E_norm, sel))))
            bin_labels.append(f"({int(lo)}, {int(hi)}]")
        # Monotone (allowing 1 small dip)?
        diffs = np.diff(bin_cos)
        monotone = (diffs >= -0.005).sum() / max(len(diffs), 1)
    else:
        bin_cos, bin_labels = [], []
        monotone = float("nan")

    # (xii) Burstiness vs sustained, at matched total count.
    # Define burstiness B(p) = std(times) / mean(times) for pairs with >= 4 ts.
    bursty_pairs, sustained_pairs = [], []
    for pair, ts_list in pair_times.items():
        if pair_count[pair] < 4:
            continue
        ts_arr = np.asarray(ts_list)
        gaps = np.diff(np.sort(ts_arr))
        if len(gaps) < 2:
            continue
        cv = float(gaps.std() / max(gaps.mean(), 1.0))
        # Heuristic: cv > 1.5 = bursty; cv < 0.5 = sustained.
        if cv > 1.5:
            bursty_pairs.append(pair)
        elif cv < 0.5:
            sustained_pairs.append(pair)
    cos_bursty = (float(np.mean(cos_pairs(E_norm, bursty_pairs[:n_sample])))
                  if bursty_pairs else float("nan"))
    cos_sustained = (float(np.mean(cos_pairs(E_norm, sustained_pairs[:n_sample])))
                     if sustained_pairs else float("nan"))

    return {
        "xi_count_bins": list(zip(bin_labels, bin_cos)),
        "xi_monotone_fraction": float(monotone),
        "xii_cos_bursty": cos_bursty,
        "xii_n_bursty": len(bursty_pairs),
        "xii_cos_sustained": cos_sustained,
        "xii_n_sustained": len(sustained_pairs),
    }


def tier5_sanity(E_norm, G, n_sample, rng):
    adj = G["adj"]
    active = G["active_nodes"]
    deg = G["node_degree"]

    # (xv) Far-pair asymptote: no edge AND no common neighbour.
    far_pairs = sample_non_interacting_pairs(active, adj, n_sample, rng,
                                             also_no_common_nbr=True)
    rand_pairs = sample_random_pairs(active, n_sample, rng)
    cos_far_mean, cos_far_std = mean_std(cos_pairs(E_norm, far_pairs))
    cos_rand_mean, cos_rand_std = mean_std(cos_pairs(E_norm, rand_pairs))

    # (xvi) Cold-start signal.
    cold_nodes = [v for v, d in deg.items() if d < 5]
    if cold_nodes:
        cold_to_partner_cos = []
        cold_to_random_cos = []
        for v in cold_nodes[:min(500, len(cold_nodes))]:
            partners = list(adj[v])
            if not partners:
                continue
            cold_to_partner_cos.append(
                float(np.mean((E_norm[v] * E_norm[partners]).sum(axis=1)))
            )
            randoms = rng.choice(active, size=min(20, len(active)),
                                 replace=False)
            cold_to_random_cos.append(
                float(np.mean((E_norm[v] * E_norm[randoms]).sum(axis=1)))
            )
        cold_partner_mean = (float(np.mean(cold_to_partner_cos))
                             if cold_to_partner_cos else float("nan"))
        cold_random_mean = (float(np.mean(cold_to_random_cos))
                            if cold_to_random_cos else float("nan"))
    else:
        cold_partner_mean, cold_random_mean = float("nan"), float("nan")

    return {
        "xv_far_cos_mean": cos_far_mean,
        "xv_far_cos_std": cos_far_std,
        "xv_rand_cos_mean": cos_rand_mean,
        "xv_rand_cos_std": cos_rand_std,
        "xv_far_vs_rand_delta": cos_far_mean - cos_rand_mean,
        "xvi_n_cold": len(cold_nodes),
        "xvi_cold_partner_cos": cold_partner_mean,
        "xvi_cold_random_cos": cold_random_mean,
    }


def tier7_link_prediction(E_norm, G, loaded, n_test_edges, n_negs, rng):
    """Raw-E link prediction signal (no LinkHead).

    For each test edge (u, v⁺), sample K random negatives v⁻ from the
    train destination pool and rank v⁺ against them by cos(E[u], E[·]).
    Reports overall MRR, plus MRR stratified by target degree (cold-start
    sensitivity) and by edge time (temporal robustness).
    """
    test = loaded.test
    src = np.asarray(test.sources, dtype=np.int64)
    tgt = np.asarray(test.destinations, dtype=np.int64)
    ts = np.asarray(test.timestamps, dtype=np.int64)

    dst_pool = np.unique(loaded.train.destinations).astype(np.int64)
    if len(src) > n_test_edges:
        idx = rng.choice(len(src), size=n_test_edges, replace=False)
        src, tgt, ts = src[idx], tgt[idx], ts[idx]

    # Overall raw-E MRR.
    deg = G["node_degree"]
    target_deg = np.asarray([deg.get(int(v), 0) for v in tgt])

    rrs = []  # reciprocal ranks
    per_edge_pos_cos = []
    per_edge_neg_cos_mean = []
    for i in range(len(src)):
        u = int(src[i])
        v_pos = int(tgt[i])
        neg = rng.choice(dst_pool, size=n_negs, replace=True)
        # Make sure positive isn't accidentally in negatives.
        neg = np.where(neg == v_pos, dst_pool[(neg + 1) % len(dst_pool)], neg)
        cands = np.concatenate([[v_pos], neg])
        scores = (E_norm[u] * E_norm[cands]).sum(axis=1)
        # Rank of positive (column 0).
        rank = 1 + int((scores[1:] > scores[0]).sum() + (
            (scores[1:] == scores[0]).sum() / 2
        ))
        rrs.append(1.0 / rank)
        per_edge_pos_cos.append(float(scores[0]))
        per_edge_neg_cos_mean.append(float(scores[1:].mean()))

    rrs = np.asarray(rrs)
    overall_mrr = float(rrs.mean())

    # Stratify by target degree quartiles.
    deg_q = np.percentile(target_deg, [25, 50, 75])
    deg_buckets = np.digitize(target_deg, deg_q)  # 0..3
    deg_strat = []
    for q in range(4):
        m = deg_buckets == q
        if m.any():
            deg_strat.append((int(m.sum()), float(rrs[m].mean())))
        else:
            deg_strat.append((0, float("nan")))

    # Stratify by edge time quartiles.
    t_q = np.percentile(ts, [25, 50, 75])
    t_buckets = np.digitize(ts, t_q)
    time_strat = []
    for q in range(4):
        m = t_buckets == q
        if m.any():
            time_strat.append((int(m.sum()), float(rrs[m].mean())))
        else:
            time_strat.append((0, float("nan")))

    # Positive vs negative cos distributions.
    pos_mean = float(np.mean(per_edge_pos_cos))
    pos_std = float(np.std(per_edge_pos_cos))
    neg_mean = float(np.mean(per_edge_neg_cos_mean))
    neg_std = float(np.std(per_edge_neg_cos_mean))

    # Cold-start sensitivity on the target side: degree < 5.
    cold_mask = target_deg < 5
    cold_mrr = (float(rrs[cold_mask].mean())
                if cold_mask.any() else float("nan"))
    cold_n = int(cold_mask.sum())

    return {
        "overall_raw_E_mrr": overall_mrr,
        "n_test_edges_used": int(len(src)),
        "n_negs_per_edge": n_negs,
        "deg_strat": deg_strat,
        "time_strat": time_strat,
        "pos_cos_mean": pos_mean,
        "pos_cos_std": pos_std,
        "neg_cos_mean": neg_mean,
        "neg_cos_std": neg_std,
        "pos_neg_separation": pos_mean - neg_mean,
        "cold_tgt_mrr": cold_mrr,
        "cold_tgt_n": cold_n,
    }


def tier6_temporal(E_norm, G, n_edges_for_probe, rng, train):
    # (xix) Linear probe: predict edge timestamp from concat(E[u], E[v]).
    src = np.asarray(train.sources, dtype=np.int64)
    tgt = np.asarray(train.destinations, dtype=np.int64)
    ts = np.asarray(train.timestamps, dtype=np.float64)

    if len(src) > n_edges_for_probe:
        idx = rng.choice(len(src), size=n_edges_for_probe, replace=False)
        src, tgt, ts = src[idx], tgt[idx], ts[idx]

    X = np.concatenate([E_norm[src], E_norm[tgt]], axis=1)  # [n, 2d]
    y = (ts - ts.min()) / max(ts.max() - ts.min(), 1.0)

    n_train = int(len(X) * 0.8)
    perm = rng.permutation(len(X))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    ss_res = float(np.sum((y[test_idx] - pred) ** 2))
    ss_tot = float(np.sum((y[test_idx] - y[test_idx].mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {
        "xix_time_probe_r2": float(r2),
        "xix_n_edges_used": int(len(X)),
    }


# ─── report ────────────────────────────────────────────────────────────


def _fmt_q(arr):
    return "[" + ", ".join(f"{x:+.3f}" if np.isfinite(x) else "nan"
                           for x in arr) + "]"


def print_report(args, t1, t2, t3, t4, t5, t6, t7, G):
    print(f"\n=== Embedding analysis: {args.dataset} ===\n")
    print(f"emb_path:        {args.emb_path}")
    print(f"n_nodes:         {t1['n_nodes']:,}")
    print(f"d_emb:           {t1['d_emb']}")
    print(f"n_train_edges:   {G['n_edges']:,}")
    print(f"n_active_nodes:  {len(G['active_nodes']):,}")
    print(f"t_range:         [{G['t_min']}, {G['t_max']}]")
    print()
    print("─── Tier 1: Geometry " + "─" * 40)
    print(f"  Norm:        mean={t1['norm_mean']:.3f}  "
          f"std={t1['norm_std']:.3f}  "
          f"[5%, 95%]=[{t1['norm_p5_p95'][0]:.3f}, {t1['norm_p5_p95'][1]:.3f}]  "
          f"min={t1['norm_min']:.3f}  max={t1['norm_max']:.3f}")
    print(f"  Eff dim:     {t1['intrinsic_dim']:.1f} / {t1['d_emb']}  "
          f"(intrinsic dim from SVD spectrum)")
    print(f"  Isotropy:    |cos| mean={t1['iso_cos_abs_mean']:.3f}  "
          f"std(cos)={t1['iso_cos_std']:.3f}  (random pairs)")
    print()
    print("─── Tier 2: Pair-level closeness " + "─" * 28)
    print(f"  (i)   Interacting vs not:  "
          f"int={t2['i_interacting_cos']:+.3f}  "
          f"non={t2['i_noninteracting_cos']:+.3f}  "
          f"Δ={t2['i_delta']:+.3f}")
    print(f"  (ii)  Recency quartiles cos: {_fmt_q(t2['ii_recency_quartiles'])}  "
          f"monotone={'✓' if t2['ii_recency_monotone'] else '✗'}")
    if t2["iii_count_bins"]:
        print(f"  (iii) Count bins (count_range → cos):")
        for lbl, c in t2["iii_count_bins"]:
            print(f"           {lbl:>14}  {c:+.3f}")
    print(f"  (iv)  Per-anchor recency Spearman: "
          f"mean={t2['iv_per_anchor_recency_spearman_mean']:+.3f}  "
          f"std={t2['iv_per_anchor_recency_spearman_std']:.3f}  "
          f"(n_anchors={t2['iv_n_anchors']})")
    print()
    print("─── Tier 3: Network-structural " + "─" * 31)
    print(f"  (v)   Common-nbr no-edge cos: cn={t3['v_common_nbr_cos']:+.3f}  "
          f"no_cn={t3['v_no_common_nbr_cos']:+.3f}  "
          f"Δ={t3['v_delta']:+.3f}")
    print(f"  (vi)  Jaccard(neighbour-set) ~ cos: r={t3['vi_jaccard_cos_corr']:+.3f}  "
          f"(n={t3['vi_n_used']})")
    print(f"  (x)   Triangle (a~b~c, a!~c) cos(a,c) "
          f"in (0, min(cos(a,b),cos(b,c))]: "
          f"pass {t3['x_triangle_pass_rate']:.1%}  (n={t3['x_triangle_total']})")
    print()
    print("─── Tier 4: Strength gradient " + "─" * 31)
    print(f"  (xi)  Count-bin monotonicity: "
          f"{t4['xi_monotone_fraction']:.0%} of inter-bin steps non-decreasing")
    if t4["xi_count_bins"]:
        print(f"        bins → cos:")
        for lbl, c in t4["xi_count_bins"]:
            print(f"           {lbl:>14}  {c:+.3f}")
    print(f"  (xii) Burstiness: cos(bursty)={t4['xii_cos_bursty']:+.3f}  "
          f"cos(sustained)={t4['xii_cos_sustained']:+.3f}  "
          f"(n_bursty={t4['xii_n_bursty']}, n_sustained={t4['xii_n_sustained']})")
    print()
    print("─── Tier 5: Sanity / negatives " + "─" * 30)
    print(f"  (xv)  Far-pair vs random cos: "
          f"far={t5['xv_far_cos_mean']:+.3f}  rand={t5['xv_rand_cos_mean']:+.3f}  "
          f"Δ={t5['xv_far_vs_rand_delta']:+.3f}  (should be ~0)")
    print(f"  (xvi) Cold-start (deg<5): partner_cos={t5['xvi_cold_partner_cos']:+.3f}  "
          f"random_cos={t5['xvi_cold_random_cos']:+.3f}  "
          f"(n_cold={t5['xvi_n_cold']})")
    print()
    print("─── Tier 6: Temporal probes " + "─" * 33)
    print(f"  (xix) Edge-timestamp linear probe R² (Ridge α=1): "
          f"{t6['xix_time_probe_r2']:+.3f}  "
          f"(n_edges_used={t6['xix_n_edges_used']:,})")
    print()
    print("─── Tier 7: Link predictability (raw E, no LinkHead) " + "─" * 8)
    print(f"  Overall MRR (raw cos vs {t7['n_negs_per_edge']} random negatives): "
          f"{t7['overall_raw_E_mrr']:.4f}  (n_test={t7['n_test_edges_used']:,})")
    print(f"  Positive vs negative cos: "
          f"pos={t7['pos_cos_mean']:+.3f}±{t7['pos_cos_std']:.3f}  "
          f"neg={t7['neg_cos_mean']:+.3f}±{t7['neg_cos_std']:.3f}  "
          f"Δ={t7['pos_neg_separation']:+.3f}")
    print(f"  By target-degree quartile (Q1=low → Q4=high deg):")
    for q, (n, m) in enumerate(t7["deg_strat"]):
        print(f"     Q{q + 1}  n={n:5d}  MRR={m:.4f}")
    print(f"  By edge-time quartile (Q1=earliest → Q4=latest):")
    for q, (n, m) in enumerate(t7["time_strat"]):
        print(f"     Q{q + 1}  n={n:5d}  MRR={m:.4f}")
    print(f"  Cold-start tgt (deg<5):   MRR={t7['cold_tgt_mrr']:.4f}  "
          f"(n={t7['cold_tgt_n']})")
    print()
    print("─── verdict cheat-sheet " + "─" * 38)
    print(f"  Geometry healthy if:      norm spread is tight, eff dim >> 5")
    print(f"  (i) interacting >> non:   Δ > 0.10 strong, 0.05 ok, < 0.02 weak")
    print(f"  (ii) recency monotone:    {'✓' if t2['ii_recency_monotone'] else '✗'}")
    print(f"  (vi) Jaccard correlation: r > 0.30 strong, 0.15 ok, < 0.05 weak")
    print(f"  (xv) far-pair sanity:     |Δ| < 0.02 means no spurious closeness")
    print(f"  (xix) R² > 0.20 means recency is linearly recoverable from E")
    print(f"  (T7) Raw-E MRR vs link-head MRR (from training log):")
    print(f"        if raw-E MRR > 0.5 × LinkHead MRR → E carries most of the signal")
    print(f"        if raw-E MRR << LinkHead MRR     → LinkHead is essential to keep")


# ─── entry ────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"Loading embedding from {args.emb_path} ...")
    E_raw = np.load(args.emb_path).astype(np.float32)
    print(f"Loading dataset {args.dataset} ...")
    loaded = load_tgb(name=args.dataset, root=args.tgb_root)

    E_norm = l2_normalise(E_raw)
    G = build_graph(loaded.train)

    print("Running Tier 1 (geometry) ...")
    t1 = tier1_geometry(E_raw, E_norm)
    print("Running Tier 2 (pair-level closeness) ...")
    t2 = tier2_pair_closeness(E_norm, G, args.n_sample, rng)
    print("Running Tier 3 (network-structural) ...")
    t3 = tier3_structural(E_norm, G, args.n_sample, rng)
    print("Running Tier 4 (strength gradient) ...")
    t4 = tier4_strength(E_norm, G, args.n_sample, rng)
    print("Running Tier 5 (sanity / negatives) ...")
    t5 = tier5_sanity(E_norm, G, args.n_sample, rng)
    print("Running Tier 6 (temporal probes) ...")
    t6 = tier6_temporal(E_norm, G, args.time_probe_edges, rng, loaded.train)
    print("Running Tier 7 (link predictability) ...")
    t7 = tier7_link_prediction(E_norm, G, loaded, args.linkpred_edges,
                               args.linkpred_negs, rng)

    print_report(args, t1, t2, t3, t4, t5, t6, t7, G)


if __name__ == "__main__":
    main()
