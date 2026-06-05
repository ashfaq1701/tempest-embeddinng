"""Phase 4 — does u's forward neighborhood (in walks) predict v?

For each test edge (u, v+, t):
  1. Ingest all training edges into a Tempest instance.
  2. Sample n_walks forward walks from u at time t (strict-causal — Tempest
     only knows train edges with t' < t).
  3. Gather the unique node set W_u touched by those walks (excluding u itself).
  4. Compute features:
       - hit_pos / hit_neg            : does v appear in W_u? (binary)
       - max_cos_pos / max_cos_neg     : max cos(E[w], E[v]) for w in W_u
       - mean_cos_pos / mean_cos_neg   : mean cos over W_u
       - min_l2_pos / min_l2_neg       : min ||E[w] - E[v]||
       - top3_mean_cos_pos / _neg      : mean of top-3 cos values
  5. Rank v+ against the 999 negatives under each metric → MRR.

This is the cheapest test of "do forward walks from u carry signal about
which v will be hit next" using only E geometry, no learned head.

To stay tractable: process a 2000-edge SAMPLE of test edges (proportional
random sample), not all 23k. Walks are sampled on CPU (use_gpu=False)
to avoid stepping on any concurrent GPU job; the per-edge work is dominated
by the walk sampler call itself.

Outputs (analysis/phase4/):
    walk_neighbor_scores.npz   per-sample-edge features + RR under each metric
    summary.json               aggregate MRR + how it relates to cos baseline
"""
import json
import pathlib
import sys
import time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EMB_PATH = ROOT / "logs" / "embeddings" / "tgbl-wiki_seed42_demb128_ep33.npy"
OUT = ROOT / "analysis" / "phase4"
OUT.mkdir(parents=True, exist_ok=True)


SAMPLE_SIZE = 2000
N_WALKS = 20
MAX_WALK_LEN = 20
WALK_BIAS = "ExponentialWeight"
START_BIAS = "Uniform"


def _rank_pos(sp: float, sn: np.ndarray) -> float:
    higher = int((sn > sp).sum())
    equal = int((sn == sp).sum())
    return higher + (equal + 2) / 2


def main():
    from tempest_walks.data import load_tgb
    from temporal_random_walk import TemporalRandomWalk

    E = np.load(EMB_PATH).astype(np.float32)
    eps = 1e-12
    E_norms = np.linalg.norm(E, axis=1).clip(min=eps)
    E_unit = E / E_norms[:, None]  # for cosine

    loaded = load_tgb("tgbl-wiki", root=str(ROOT / "datasets"))
    print(f"E shape: {E.shape}")
    print(f"max_node_count: {loaded.max_node_count}")

    # Ingest train + ALL val edges with t' < t_test as Tempest state.
    # Strict-causal: at test edge time t, we know train edges + val edges
    # before t. For simplicity in this analysis we ingest train edges only —
    # an upper bound on "what u has done before"; we report this conservative
    # walks-of-u_through_train.
    trw = TemporalRandomWalk(
        is_directed=False,        # wiki is undirected
        use_gpu=False,
        enable_weight_computation=True,
        timescale_bound=300,
        max_time_capacity=-1,
        shuffle_walk_order=False,
    )
    print(f"ingesting {len(loaded.train.sources)} train edges ...")
    trw.add_multiple_edges(
        loaded.train.sources.astype(np.int32),
        loaded.train.destinations.astype(np.int32),
        loaded.train.timestamps.astype(np.int64),
    )

    # Load TGB negs for test
    loaded.dataset.load_test_ns()
    ns = loaded.dataset.negative_sampler
    test_neg = ns.query_batch(
        loaded.test.sources, loaded.test.destinations,
        loaded.test.timestamps, split_mode="test",
    )
    test_neg = [np.asarray(x, dtype=np.int64) for x in test_neg]

    # Sample evaluation edges
    rng = np.random.default_rng(0)
    n_test = len(loaded.test.sources)
    sample_idx = rng.choice(n_test, size=min(SAMPLE_SIZE, n_test), replace=False)
    sample_idx.sort()
    print(f"sampling {len(sample_idx)} test edges for walk analysis")

    # Per-edge feature collection
    out = {
        "rr_cos_base":    [],   # baseline cos
        "rr_hit":         [],   # v in W_u indicator score
        "rr_max_cos":     [],   # max over W_u
        "rr_mean_cos":    [],   # mean over W_u
        "rr_top3_cos":    [],   # top-3 mean
        "rr_min_l2":      [],   # min L2
        "rr_combined":    [],   # cos + max_cos_walk (simple sum)
        "n_walk_nodes":   [],   # |W_u| per edge
        "v_in_walks":     [],   # boolean — was v+ in walks
    }

    src_arr = loaded.test.sources
    tgt_arr = loaded.test.destinations
    ts_arr  = loaded.test.timestamps

    t0 = time.time()
    progress_every = max(1, len(sample_idx) // 20)
    for k, i in enumerate(sample_idx):
        u = int(src_arr[i])
        v_pos = int(tgt_arr[i])
        ndst = test_neg[i]
        candidates = np.concatenate([[v_pos], ndst]).astype(np.int64)
        cand_unit = E_unit[candidates]
        cand_full = E[candidates]
        u_unit = E_unit[u]
        u_full = E[u]

        # Baseline cos
        cos_all = cand_unit @ u_unit
        rr_cos_base = 1.0 / _rank_pos(float(cos_all[0]), cos_all[1:])
        out["rr_cos_base"].append(rr_cos_base)

        # Forward walks from u with strict-causal state = train only
        seeds = np.array([u], dtype=np.int32)
        nodes, ts_w, lens, _ = trw.get_random_walks_and_times_for_nodes(
            seed_nodes=seeds,
            max_walk_len=MAX_WALK_LEN,
            walk_bias=WALK_BIAS,
            initial_edge_bias=START_BIAS,
            num_walks_per_node=N_WALKS,
            walk_direction="Forward_In_Time",
        )
        # Collect unique nodes touched (excluding u itself and padding/sentinel)
        nodes = nodes.flatten()
        valid_mask = (nodes >= 0) & (nodes != u)
        walk_nodes = np.unique(nodes[valid_mask])
        out["n_walk_nodes"].append(int(len(walk_nodes)))

        if len(walk_nodes) == 0:
            # u has no train history → all walk features are no-signal
            for k_ in ("rr_hit", "rr_max_cos", "rr_mean_cos",
                      "rr_top3_cos", "rr_min_l2"):
                out[k_].append(rr_cos_base)
            out["rr_combined"].append(rr_cos_base)
            out["v_in_walks"].append(False)
            continue

        Wu_unit = E_unit[walk_nodes]   # [|W_u|, d]
        Wu_full = E[walk_nodes]

        # For each candidate v, compute features:
        sim_matrix = cand_unit @ Wu_unit.T   # [n_cand, |W_u|]
        l2_matrix = np.linalg.norm(
            cand_full[:, None, :] - Wu_full[None, :, :], axis=2,
        )                                    # [n_cand, |W_u|]
        # Hit indicator: is candidate in walk_nodes?
        hit = np.isin(candidates, walk_nodes).astype(np.float32)

        max_cos = sim_matrix.max(axis=1)
        mean_cos = sim_matrix.mean(axis=1)
        k_top = min(3, sim_matrix.shape[1])
        top_part = np.partition(sim_matrix, -k_top, axis=1)[:, -k_top:]
        top3_cos = top_part.mean(axis=1)
        min_l2 = l2_matrix.min(axis=1)

        for key, scores, is_neg_l2 in [
            ("rr_hit", hit, False),
            ("rr_max_cos", max_cos, False),
            ("rr_mean_cos", mean_cos, False),
            ("rr_top3_cos", top3_cos, False),
            ("rr_min_l2", -min_l2, False),  # smaller is better -> negate
        ]:
            sp = float(scores[0])
            sn = scores[1:]
            out[key].append(1.0 / _rank_pos(sp, sn))

        # Combined score: baseline cos + max_cos to walk node (simple sum)
        comb = cos_all + max_cos
        out["rr_combined"].append(1.0 / _rank_pos(float(comb[0]), comb[1:]))

        out["v_in_walks"].append(bool(hit[0]))

        if (k + 1) % progress_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(sample_idx) - k - 1)
            print(f"  [{k+1}/{len(sample_idx)}]  elapsed {elapsed:.0f}s  ETA {eta:.0f}s  "
                  f"|W_u| running mean {np.mean(out['n_walk_nodes']):.1f}")

    # Aggregate
    arrs = {k: np.asarray(v) for k, v in out.items()}
    np.savez(OUT / "walk_neighbor_scores.npz", **arrs,
             sample_idx=sample_idx,
             src=src_arr[sample_idx],
             tgt=tgt_arr[sample_idx],
             ts=ts_arr[sample_idx])

    summary = {
        "n_sample": int(len(sample_idx)),
        "n_walks": N_WALKS,
        "max_walk_len": MAX_WALK_LEN,
        "walk_bias": WALK_BIAS,
        "start_bias": START_BIAS,
        "ingested": "train_only",
        "mean_walk_node_count": float(arrs["n_walk_nodes"].mean()),
        "fraction_v_in_walks": float(arrs["v_in_walks"].mean()),
        "mrr_cos_base":   float(arrs["rr_cos_base"].mean()),
        "mrr_hit":        float(arrs["rr_hit"].mean()),
        "mrr_max_cos":    float(arrs["rr_max_cos"].mean()),
        "mrr_mean_cos":   float(arrs["rr_mean_cos"].mean()),
        "mrr_top3_cos":   float(arrs["rr_top3_cos"].mean()),
        "mrr_min_l2":     float(arrs["rr_min_l2"].mean()),
        "mrr_combined":   float(arrs["rr_combined"].mean()),
        # Conditional: for edges where v+ IS in walks, how well does cos rank it?
        "mrr_cos_when_v_in_walks":     float(
            arrs["rr_cos_base"][arrs["v_in_walks"]].mean()
            if arrs["v_in_walks"].any() else 0
        ),
        "mrr_cos_when_v_NOT_in_walks": float(
            arrs["rr_cos_base"][~arrs["v_in_walks"]].mean()
            if (~arrs["v_in_walks"]).any() else 0
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
