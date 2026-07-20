"""Quick re-analysis of a new embedding npy.

Usage: ./.venv/bin/python analysis/reanalyse.py <emb_path> <out_dir>

Reports the four most actionable bottom-line numbers:
    1. Basic norms + activity stratification
    2. cos / -L2 / dot val + test MRR
    3. Activity-stratified MRR (both_active / inductive)
    4. Margin distribution (P10/P50/P90)

Writes <out_dir>/reanalysis.json + console summary.
"""
import json
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _rank_pos(sp, sn):
    higher = int((sn > sp).sum())
    equal = int((sn == sp).sum())
    return higher + (equal + 2) / 2


def main():
    if len(sys.argv) < 3:
        print("usage: reanalyse.py <emb_path> <out_dir>")
        sys.exit(2)
    emb_path = pathlib.Path(sys.argv[1])
    out_dir = pathlib.Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    from link_property_prediction.data import load_tgb

    E = np.load(emb_path).astype(np.float32)
    print(f"E shape: {E.shape}   path: {emb_path}")
    loaded = load_tgb("tgbl-wiki", root=str(ROOT / "datasets"))

    # Activity classification
    src = loaded.train.sources.astype(np.int64)
    tgt = loaded.train.destinations.astype(np.int64)
    ts = loaded.train.timestamps.astype(np.int64)
    num_nodes = loaded.max_node_count
    last_seen = np.full(num_nodes, -1, dtype=np.int64)
    np.maximum.at(last_seen, src, ts)
    np.maximum.at(last_seen, tgt, ts)
    active = (last_seen >= 0)
    deg = np.zeros(num_nodes, dtype=np.int64)
    np.add.at(deg, src, 1); np.add.at(deg, tgt, 1)
    n_active = int(active.sum())
    print(f"active nodes: {n_active}/{num_nodes}  ({n_active/num_nodes:.1%})")

    norms = np.linalg.norm(E, axis=1)
    nrm_active = norms[active]
    nrm_inactive = norms[~active]
    print(f"norms — active   mean={nrm_active.mean():.3f}  std={nrm_active.std():.3f}")
    print(f"norms — inactive mean={nrm_inactive.mean():.3f}  std={nrm_inactive.std():.3f}")

    # Load TGB neg sampler
    loaded.dataset.load_val_ns(); loaded.dataset.load_test_ns()
    ns = loaded.dataset.negative_sampler

    def score_split(name, split):
        raw = ns.query_batch(split.sources, split.destinations,
                             split.timestamps, split_mode=name)
        neg_list = [np.asarray(x, dtype=np.int64) for x in raw]
        eps = 1e-12
        Eu_n = E / np.linalg.norm(E, axis=1, keepdims=True).clip(min=eps)
        rr_cos = np.empty(len(split.sources))
        rr_l2 = np.empty(len(split.sources))
        rr_dot = np.empty(len(split.sources))
        margin = np.empty(len(split.sources))
        u_active = active[split.sources]
        v_active = active[split.destinations]
        for i, ndst in enumerate(neg_list):
            u = split.sources[i]; v = split.destinations[i]
            cand = np.concatenate([[v], ndst])
            cos_sc = E[cand] @ E[u] / (
                np.linalg.norm(E[cand], axis=1).clip(min=eps) *
                np.linalg.norm(E[u]).clip(min=eps))
            l2_sc = -np.linalg.norm(E[cand] - E[u], axis=1)
            dot_sc = E[cand] @ E[u]
            rr_cos[i] = 1.0 / _rank_pos(cos_sc[0], cos_sc[1:])
            rr_l2[i] = 1.0 / _rank_pos(l2_sc[0], l2_sc[1:])
            rr_dot[i] = 1.0 / _rank_pos(dot_sc[0], dot_sc[1:])
            margin[i] = cos_sc[0] - cos_sc[1:].mean()

        out = {
            "n_edges": len(split.sources),
            "mrr_cos":   float(rr_cos.mean()),
            "mrr_negl2": float(rr_l2.mean()),
            "mrr_dot":   float(rr_dot.mean()),
            "margin_mean": float(margin.mean()),
            "margin_P10":  float(np.percentile(margin, 10)),
            "margin_P90":  float(np.percentile(margin, 90)),
        }
        both_active = u_active & v_active
        ind_u_only = (~u_active) & v_active
        ind_v_only = u_active & (~v_active)
        ind_both = (~u_active) & (~v_active)
        for sub_name, mask in (("both_active", both_active),
                               ("ind_u_only", ind_u_only),
                               ("ind_v_only", ind_v_only),
                               ("ind_both", ind_both)):
            if mask.sum() == 0:
                continue
            out[f"mrr_cos_{sub_name}"] = float(rr_cos[mask].mean())
            out[f"fraction_{sub_name}"] = float(mask.mean())
        return out, rr_cos, rr_l2, rr_dot, margin

    print("\n--- VAL ---")
    val_out, *_ = score_split("val", loaded.val)
    for k, v in val_out.items():
        print(f"  {k}: {v}")

    print("\n--- TEST ---")
    test_out, *_ = score_split("test", loaded.test)
    for k, v in test_out.items():
        print(f"  {k}: {v}")

    summary = {
        "emb_path": str(emb_path),
        "n_active": n_active,
        "n_total":  num_nodes,
        "active_norm_mean":   float(nrm_active.mean()),
        "inactive_norm_mean": float(nrm_inactive.mean()) if len(nrm_inactive) else None,
        "val":  val_out,
        "test": test_out,
    }
    (out_dir / "reanalysis.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir/'reanalysis.json'}")


if __name__ == "__main__":
    main()
