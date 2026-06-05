"""Phase 2 — how separable is (u, v+) from (u, v-) using only E?

For each val and test edge we compute four scalar similarities between
E[u] and each candidate (v+ first, then TGB's pre-generated negatives):

    cos(u, v)        on-sphere alignment
    -L2(u, v)        magnitude-aware distance, sign-flipped
    dot(u, v)        cosine times product of norms
    sum(E[u] * E[v]) identical to dot — included for completeness only

We score the positive's rank inside each per-edge candidate list, then
report mean reciprocal rank under each metric. The gap to the model's
trained link head (val 0.5418 / test 0.4709) is the headroom that the
link head bought on top of raw E geometry.

Outputs (analysis/phase2/):
    val_scores.npy  / test_scores.npy
        struct array per edge: cos_rr, neg_l2_rr, dot_rr (the per-edge
        reciprocal rank under each metric). One row per evaluated edge.
    summary.json    metric-level MRR plus margin distributions.
"""
import json
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EMB_PATH = ROOT / "logs" / "embeddings" / "tgbl-wiki_seed42_demb128_ep33.npy"
OUT = ROOT / "analysis" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)


def load_splits():
    from tempest_walks.data import load_tgb
    return load_tgb("tgbl-wiki", root=str(ROOT / "datasets"))


def per_edge_metrics(E, src, tgt, neg_dst_list):
    """For each edge i, return rr_cos, rr_negl2, rr_dot."""
    eps = 1e-12
    Eu = E[src]
    norms_u = np.linalg.norm(Eu, axis=1, keepdims=True).clip(min=eps)
    Eu_n = Eu / norms_u
    out = {"cos": [], "negl2": [], "dot": []}
    margin_cos_pos = []
    margin_cos_neg = []
    for i, neg in enumerate(neg_dst_list):
        cand = np.concatenate([[tgt[i]], neg.astype(np.int64)])
        Ev = E[cand]
        # cosine
        norms_v = np.linalg.norm(Ev, axis=1).clip(min=eps)
        cos_scores = (Eu[i] @ Ev.T) / (float(norms_u[i, 0]) * norms_v)
        # -L2
        diff = Eu[i][None, :] - Ev
        negl2_scores = -np.linalg.norm(diff, axis=1)
        # dot (also = sum(E[u] * E[v]))
        dot_scores = Eu[i] @ Ev.T

        for key, sc in (("cos", cos_scores),
                        ("negl2", negl2_scores),
                        ("dot", dot_scores)):
            # rank of positive (index 0) among all
            # bigger is better -> rank = 1 + #{j : sc[j] > sc[0]} (ties = midrank)
            higher = (sc > sc[0]).sum()
            equal = (sc == sc[0]).sum()
            rank = higher + (equal + 1) / 2  # mid-rank for ties
            out[key].append(1.0 / rank)

        # margin diagnostics on cosine
        margin_cos_pos.append(cos_scores[0])
        margin_cos_neg.append(cos_scores[1:].mean())

    return (
        np.asarray(out["cos"], dtype=np.float64),
        np.asarray(out["negl2"], dtype=np.float64),
        np.asarray(out["dot"], dtype=np.float64),
        np.asarray(margin_cos_pos, dtype=np.float32),
        np.asarray(margin_cos_neg, dtype=np.float32),
    )


def score_split(name, E, split_data, neg_sampler, eval_metric_name="mrr"):
    print(f"\n--- {name} split ---")
    print(f"  {len(split_data.sources)} edges")
    print(f"  loading TGB negatives ...")
    # TGB negative sampler: query_batch returns a list of arrays per edge.
    raw = neg_sampler.query_batch(
        split_data.sources, split_data.destinations, split_data.timestamps,
        split_mode=name,
    )
    # TGB returns list-of-lists; coerce to int64 arrays.
    neg_dst_list = [np.asarray(x, dtype=np.int64) for x in raw]
    print(f"  per-edge negatives shape: "
          f"{[arr.shape for arr in neg_dst_list[:3]]}  (first 3)")

    rr_cos, rr_negl2, rr_dot, m_pos, m_neg = per_edge_metrics(
        E, split_data.sources, split_data.destinations, neg_dst_list,
    )

    out = {
        "n_edges": int(rr_cos.shape[0]),
        "mrr_cos":    float(rr_cos.mean()),
        "mrr_negl2":  float(rr_negl2.mean()),
        "mrr_dot":    float(rr_dot.mean()),
        "hits_at_1_cos":   float((rr_cos == 1.0).mean()),
        "hits_at_10_cos":  float((rr_cos >= 0.1).mean()),
        "hits_at_1_negl2": float((rr_negl2 == 1.0).mean()),
        "hits_at_10_negl2":float((rr_negl2 >= 0.1).mean()),
        "hits_at_1_dot":   float((rr_dot == 1.0).mean()),
        "hits_at_10_dot":  float((rr_dot >= 0.1).mean()),
        "margin_cos": {
            "pos_mean":  float(m_pos.mean()),
            "neg_mean":  float(m_neg.mean()),
            "pos_minus_neg_mean": float((m_pos - m_neg).mean()),
            "pos_minus_neg_std":  float((m_pos - m_neg).std()),
            "pos_minus_neg_P10":  float(np.percentile(m_pos - m_neg, 10)),
            "pos_minus_neg_P90":  float(np.percentile(m_pos - m_neg, 90)),
        },
    }
    for k, v in out.items():
        print(f"  {k}: {v}")

    np.savez(
        OUT / f"{name}_scores.npz",
        rr_cos=rr_cos, rr_negl2=rr_negl2, rr_dot=rr_dot,
        margin_cos_pos=m_pos, margin_cos_neg=m_neg,
        src=split_data.sources, tgt=split_data.destinations,
        ts=split_data.timestamps,
    )
    return out


def main():
    E = np.load(EMB_PATH).astype(np.float32)
    print(f"E shape: {E.shape}")

    loaded = load_splits()
    print(f"dataset: {loaded.name}")

    # Load TGB's evaluator + neg sampler for the dataset.
    dataset = loaded.dataset
    dataset.load_val_ns()
    dataset.load_test_ns()
    neg_sampler = dataset.negative_sampler

    out = {
        "val":  score_split("val",  E, loaded.val,  neg_sampler),
        "test": score_split("test", E, loaded.test, neg_sampler),
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nsummary.json -> {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
