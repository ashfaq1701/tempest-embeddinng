"""Phase 6 — what makes test edges hard?

For each test edge we already have per-edge cosine margin (pos - mean neg)
from phase 2 and walk-in-walks indicator from phase 4 (sample). Goal:
characterise the bottom-decile and top-decile of margin distribution.

Per-edge covariates:
  - u_active (u seen in train)
  - v_active (v seen in train)
  - n_train_edges_u (degree of u in train)
  - n_train_edges_v (degree of v in train)
  - dt_u (gap since u's last train activity)
  - dt_v (gap since v's last train activity)
  - u_norm, v_norm (E norms)
  - pos_cos (cos(E[u], E[v+]))
  - neg_cos_mean (mean cos against TGB negs)
  - margin (pos_cos - neg_cos_mean)
  - rank_pos (rank of v+ among candidates under cos)
  - in_walks (from phase 4 sample only)

Then stratify hard vs easy:
  - hard = margin < 0
  - easy = margin > 0.5
  - report covariate distributions per group

Output (analysis/phase6/):
    per_edge_covariates.npz
    hardness_summary.json
"""
import json
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EMB_PATH = ROOT / "logs" / "embeddings" / "tgbl-wiki_seed42_demb128_ep33.npy"
OUT = ROOT / "analysis" / "phase6"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    from tempest_walks.data import load_tgb
    E = np.load(EMB_PATH).astype(np.float32)
    loaded = load_tgb("tgbl-wiki", root=str(ROOT / "datasets"))

    # Train activity per node: degree + last-seen time
    train = loaded.train
    src = train.sources.astype(np.int64)
    tgt = train.destinations.astype(np.int64)
    ts = train.timestamps.astype(np.int64)
    num_nodes = loaded.max_node_count

    # Degree (undirected wiki -> add both)
    deg = np.zeros(num_nodes, dtype=np.int64)
    np.add.at(deg, src, 1)
    np.add.at(deg, tgt, 1)

    last_seen = np.full(num_nodes, -1, dtype=np.int64)
    np.maximum.at(last_seen, src, ts)
    np.maximum.at(last_seen, tgt, ts)
    active = (last_seen >= 0)

    # Load phase2 test scores for per-edge cosine info
    p2 = np.load(ROOT / "analysis" / "phase2" / "test_scores.npz", allow_pickle=True)
    test_src = p2["src"].astype(np.int64)
    test_tgt = p2["tgt"].astype(np.int64)
    test_ts = p2["ts"].astype(np.int64)
    pos_cos = p2["margin_cos_pos"].astype(np.float32)
    neg_cos = p2["margin_cos_neg"].astype(np.float32)
    margin = pos_cos - neg_cos
    rr_cos = p2["rr_cos"].astype(np.float32)

    eps = 1e-12
    u_norm = np.linalg.norm(E, axis=1)[test_src]
    v_norm = np.linalg.norm(E, axis=1)[test_tgt]
    u_active = active[test_src]
    v_active = active[test_tgt]
    u_deg = deg[test_src]
    v_deg = deg[test_tgt]
    # log1p clamps the sentinel for inactive nodes
    dt_u = np.maximum(test_ts - last_seen[test_src], -1).astype(np.float64)
    dt_v = np.maximum(test_ts - last_seen[test_tgt], -1).astype(np.float64)

    np.savez(OUT / "per_edge_covariates.npz",
             src=test_src, tgt=test_tgt, ts=test_ts,
             margin=margin, pos_cos=pos_cos, neg_cos=neg_cos, rr_cos=rr_cos,
             u_norm=u_norm, v_norm=v_norm,
             u_active=u_active, v_active=v_active,
             u_deg=u_deg, v_deg=v_deg, dt_u=dt_u, dt_v=dt_v)

    # Stratify
    hard = margin < 0
    easy = margin > 0.5
    mid = ~hard & ~easy

    print(f"hard (margin < 0):    {hard.sum()} edges  ({hard.mean():.3%})")
    print(f"mid  (0 ≤ margin ≤ 0.5): {mid.sum()} edges  ({mid.mean():.3%})")
    print(f"easy (margin > 0.5):  {easy.sum()} edges  ({easy.mean():.3%})")

    summary = {"counts": {"hard": int(hard.sum()),
                          "mid":  int(mid.sum()),
                          "easy": int(easy.sum())}}
    for name, mask in (("hard", hard), ("mid", mid), ("easy", easy)):
        summary[name] = {
            "mrr_cos":           float(rr_cos[mask].mean()),
            "u_active_rate":     float(u_active[mask].mean()),
            "v_active_rate":     float(v_active[mask].mean()),
            "u_norm_mean":       float(u_norm[mask].mean()),
            "v_norm_mean":       float(v_norm[mask].mean()),
            "u_deg_median":      float(np.median(u_deg[mask])),
            "v_deg_median":      float(np.median(v_deg[mask])),
            "u_deg_mean":        float(u_deg[mask].mean()),
            "v_deg_mean":        float(v_deg[mask].mean()),
            "log1p_dt_u_mean":   float(np.log1p(np.maximum(dt_u[mask], 0)).mean()),
            "log1p_dt_v_mean":   float(np.log1p(np.maximum(dt_v[mask], 0)).mean()),
            "pos_cos_mean":      float(pos_cos[mask].mean()),
            "neg_cos_mean":      float(neg_cos[mask].mean()),
        }
        s = summary[name]
        print(f"\n--- {name} ---")
        for k, v in s.items():
            print(f"  {k}: {v}")

    # Special case: inactive cases
    both_active = u_active & v_active
    only_u_inactive = (~u_active) & v_active
    only_v_inactive = u_active & (~v_active)
    both_inactive = (~u_active) & (~v_active)
    print(f"\nactivity-stratified test edges (n=23621):")
    for name, mask in (("both_active", both_active),
                       ("u_inactive_only", only_u_inactive),
                       ("v_inactive_only", only_v_inactive),
                       ("both_inactive", both_inactive)):
        if mask.sum() == 0:
            continue
        summary[f"act_{name}"] = {
            "count":   int(mask.sum()),
            "fraction": float(mask.mean()),
            "mrr_cos":  float(rr_cos[mask].mean()),
            "margin_mean": float(margin[mask].mean()),
        }
        print(f"  {name}: {mask.sum()}  ({mask.mean():.3%})  "
              f"MRR_cos={rr_cos[mask].mean():.4f}  margin={margin[mask].mean():.3f}")

    (OUT / "hardness_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'hardness_summary.json'}")


if __name__ == "__main__":
    main()
