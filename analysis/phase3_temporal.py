"""Phase 3 — does time help separate positives from negatives?

Question 1: for a val edge (u, v+, t), do the temporal features
    Δu(t) := t − last_train_activity(u)
    Δv+(t):= t − last_train_activity(v+)
correlate with the per-edge ranking signal cos(E[u], E[v+]) - mean cos(E[u], E[v-])?

Question 2: do time-conditioned scorers improve per-edge MRR over static
cos / dot? We test a small zoo:

    score(u, v, t) = cos(u, v) · f(Δu(t), Δv(t))

with f in
    {1                                      -- static baseline
     exp(-Δu / τ)                           -- u-side recency only
     exp(-Δv / τ)                           -- v-side recency only
     exp(-(Δu + Δv) / τ)                    -- joint
     exp(-|Δu - Δv| / τ)                    -- synchrony (active around same time)}

Sweep τ on a log-grid and pick the best per-form on VAL, report MRR
on TEST (no leakage). If none of these wins meaningfully over static,
recency is not a multiplicative modulator at the (u, v, t) level — it
would need to enter through E itself or as an additive channel.

Outputs (analysis/phase3/):
    last_activity.npz   t_last_seen per node from train
    per_edge_deltas.npz Δu, Δv per val + test edge
    correlations.json   Spearman/Pearson against margin
    scorer_sweep.json   best τ per form, val/test MRR
"""
import json
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EMB_PATH = ROOT / "logs" / "embeddings" / "tgbl-wiki_seed42_demb128_ep33.npy"
OUT = ROOT / "analysis" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)
NEG_INF_TS = -(1 << 62)


def compute_last_activity(train, num_nodes):
    """For each node, return the max train timestamp on any incident edge.
    Nodes never seen in train get NEG_INF_TS (so Δ becomes large positive)."""
    last_seen = np.full(num_nodes, NEG_INF_TS, dtype=np.int64)
    src = train.sources.astype(np.int64)
    tgt = train.destinations.astype(np.int64)
    ts = train.timestamps.astype(np.int64)
    # numpy.maximum.at handles unsorted writes safely
    np.maximum.at(last_seen, src, ts)
    np.maximum.at(last_seen, tgt, ts)
    return last_seen


def per_edge_deltas(split, last_seen):
    u, v, t = split.sources, split.destinations, split.timestamps
    du = (t - last_seen[u]).astype(np.float64)
    dv = (t - last_seen[v]).astype(np.float64)
    # Clip to non-negative; nodes unseen in train give a huge positive Δ.
    du = np.maximum(du, 0.0)
    dv = np.maximum(dv, 0.0)
    return du, dv


def margin_cos(E, src, tgt, neg_dst_list):
    """Per-edge cos(u, v+) and mean cos(u, v-)."""
    eps = 1e-12
    Eu = E[src]
    norm_u = np.linalg.norm(Eu, axis=1).clip(min=eps)
    pos = []
    neg = []
    for i, ndst in enumerate(neg_dst_list):
        Ev_pos = E[tgt[i]]
        cos_pos = float((Eu[i] @ Ev_pos) /
                        (norm_u[i] * (np.linalg.norm(Ev_pos) + eps)))
        Ev_neg = E[ndst]
        norm_n = np.linalg.norm(Ev_neg, axis=1).clip(min=eps)
        cos_neg = (Eu[i] @ Ev_neg.T) / (norm_u[i] * norm_n)
        pos.append(cos_pos)
        neg.append(float(cos_neg.mean()))
    return np.asarray(pos), np.asarray(neg)


def mrr_under_score(E, src, tgt, neg_dst_list,
                    score_pos_fn, score_neg_fn):
    """Generic: score_pos_fn(i) -> scalar for the positive;
    score_neg_fn(i) -> array for all negatives. Returns 1/rank per edge."""
    rr = np.empty(len(src), dtype=np.float64)
    for i, ndst in enumerate(neg_dst_list):
        sp = float(score_pos_fn(i))
        sn = np.asarray(score_neg_fn(i))
        higher = (sn > sp).sum()
        equal  = (sn == sp).sum()
        rank   = higher + (equal + 1) / 2
        rr[i]  = 1.0 / rank
    return rr


def main():
    from link_property_prediction.data import load_tgb
    E = np.load(EMB_PATH).astype(np.float32)
    print(f"E shape: {E.shape}")
    loaded = load_tgb("tgbl-wiki", root=str(ROOT / "datasets"))
    last_seen = compute_last_activity(loaded.train, loaded.max_node_count)
    np.save(OUT / "last_activity.npy", last_seen)
    print(f"Δu, Δv anchors computed for {loaded.max_node_count} nodes")

    # Per-split deltas (we need TGB neg lists for v+/v- comparison too)
    val_du, val_dv = per_edge_deltas(loaded.val, last_seen)
    test_du, test_dv = per_edge_deltas(loaded.test, last_seen)
    np.savez(OUT / "per_edge_deltas.npz",
             val_du=val_du, val_dv=val_dv,
             test_du=test_du, test_dv=test_dv)
    print(f"val deltas: du mean={val_du.mean():.0f}  dv mean={val_dv.mean():.0f}")
    print(f"test deltas: du mean={test_du.mean():.0f}  dv mean={test_dv.mean():.0f}")

    # Cosine margin pos vs neg (raw E)
    dataset = loaded.dataset
    dataset.load_val_ns(); dataset.load_test_ns()
    ns = dataset.negative_sampler
    val_neg = ns.query_batch(loaded.val.sources, loaded.val.destinations,
                             loaded.val.timestamps, split_mode="val")
    test_neg = ns.query_batch(loaded.test.sources, loaded.test.destinations,
                              loaded.test.timestamps, split_mode="test")
    val_pos_cos, val_neg_cos = margin_cos(E, loaded.val.sources,
                                          loaded.val.destinations, val_neg)
    test_pos_cos, test_neg_cos = margin_cos(E, loaded.test.sources,
                                            loaded.test.destinations, test_neg)
    val_margin = val_pos_cos - val_neg_cos
    test_margin = test_pos_cos - test_neg_cos

    # Correlations against deltas
    from scipy.stats import spearmanr, pearsonr
    corrs = {}
    for name, du, dv, margin in [
        ("val",  val_du,  val_dv,  val_margin),
        ("test", test_du, test_dv, test_margin),
    ]:
        # Use log1p on deltas to tame heavy tails
        ldu = np.log1p(du); ldv = np.log1p(dv)
        s_du = spearmanr(ldu, margin).correlation
        s_dv = spearmanr(ldv, margin).correlation
        s_sum = spearmanr(ldu + ldv, margin).correlation
        s_abs = spearmanr(np.abs(ldu - ldv), margin).correlation
        corrs[name] = {
            "spearman_log1p(du)_vs_margin":  float(s_du),
            "spearman_log1p(dv)_vs_margin":  float(s_dv),
            "spearman_log1p(du+dv)_vs_margin": float(s_sum),
            "spearman_|log1p(du)-log1p(dv)|_vs_margin": float(s_abs),
            "margin_mean": float(margin.mean()),
            "margin_std":  float(margin.std()),
            "margin_P10":  float(np.percentile(margin, 10)),
            "margin_P90":  float(np.percentile(margin, 90)),
        }
        print(f"\n{name}: ρ(log1p Δu, margin)={s_du:+.3f}  "
              f"ρ(log1p Δv, margin)={s_dv:+.3f}  "
              f"ρ(sum,margin)={s_sum:+.3f}  ρ(|diff|,margin)={s_abs:+.3f}")

    (OUT / "correlations.json").write_text(json.dumps(corrs, indent=2))

    # Sweep time-conditioned scorers. Build a base cos table per edge
    # (positive scalar + neg array) once, then re-rank with modulators.
    def build_cos_tables(split, neg_lists):
        u, v = split.sources, split.destinations
        eps = 1e-12
        Eu = E[u]
        norm_u = np.linalg.norm(Eu, axis=1).clip(min=eps)
        pos_scores = np.zeros(len(u))
        neg_scores = []
        for i, ndst in enumerate(neg_lists):
            Ev_pos = E[v[i]]
            pos_scores[i] = (Eu[i] @ Ev_pos) / (
                norm_u[i] * (np.linalg.norm(Ev_pos) + eps))
            Ev_neg = E[ndst]
            norm_n = np.linalg.norm(Ev_neg, axis=1).clip(min=eps)
            neg_scores.append(
                (Eu[i] @ Ev_neg.T) / (norm_u[i] * norm_n)
            )
        return pos_scores, neg_scores

    print("\n--- Building cos tables ---")
    val_pos_table, val_neg_table = build_cos_tables(loaded.val, val_neg)
    test_pos_table, test_neg_table = build_cos_tables(loaded.test, test_neg)
    # Also need per-neg-candidate Δv for v-side modulation
    val_dv_neg_per_edge = []
    for i, ndst in enumerate(val_neg):
        delta_v_neg = np.maximum(loaded.val.timestamps[i] - last_seen[ndst], 0)
        val_dv_neg_per_edge.append(delta_v_neg.astype(np.float64))
    test_dv_neg_per_edge = []
    for i, ndst in enumerate(test_neg):
        delta_v_neg = np.maximum(loaded.test.timestamps[i] - last_seen[ndst], 0)
        test_dv_neg_per_edge.append(delta_v_neg.astype(np.float64))

    def _rank_pos(sp: float, sn: np.ndarray) -> float:
        """Midrank of sp inside [sp] ∪ sn (positive included once)."""
        higher = int((sn > sp).sum())
        equal = int((sn == sp).sum())  # ties among negatives only
        # Positive itself contributes one more to the equal group →
        # there are `equal + 1` items at the positive's score level,
        # occupying ranks [higher+1, higher+equal+1]; midrank average:
        return higher + (equal + 2) / 2

    def static_mrr(pos, neg):
        rr = [1.0 / _rank_pos(sp, sn) for sp, sn in zip(pos, neg)]
        return float(np.mean(rr))

    def modulated_mrr(pos, neg, du, dv_pos, dv_neg_list, form, tau):
        rr = []
        for i, (sp, sn) in enumerate(zip(pos, neg)):
            if form == "static":
                mod_p, mod_n = 1.0, 1.0
            elif form == "u_recency":
                mod_p = np.exp(-du[i] / tau)
                mod_n = np.exp(-du[i] / tau)  # u side same for all candidates
            elif form == "v_recency":
                mod_p = np.exp(-dv_pos[i] / tau)
                mod_n = np.exp(-dv_neg_list[i] / tau)
            elif form == "joint":
                mod_p = np.exp(-(du[i] + dv_pos[i]) / tau)
                mod_n = np.exp(-(du[i] + dv_neg_list[i]) / tau)
            elif form == "synchrony":
                mod_p = np.exp(-abs(du[i] - dv_pos[i]) / tau)
                mod_n = np.exp(-np.abs(du[i] - dv_neg_list[i]) / tau)
            else:
                raise KeyError(form)
            sp_m = sp * mod_p
            sn_m = sn * mod_n
            rr.append(1.0 / _rank_pos(float(sp_m), np.asarray(sn_m)))
        return float(np.mean(rr))

    forms = ["static", "u_recency", "v_recency", "joint", "synchrony"]
    # Log-grid of taus across the data's temporal scale
    train_T = float(loaded.train.timestamps.max() - loaded.train.timestamps.min())
    tau_grid = np.geomspace(train_T / 1e5, train_T * 10, num=20)
    print(f"\n--- Tau sweep over {len(tau_grid)} τ ∈ [{tau_grid[0]:.1e}, {tau_grid[-1]:.1e}] ---")

    sweep = {"forms": forms, "tau_grid": tau_grid.tolist(), "val_mrr": {}, "test_mrr": {}}
    static_val_mrr = static_mrr(val_pos_table, val_neg_table)
    static_test_mrr = static_mrr(test_pos_table, test_neg_table)
    sweep["val_mrr"]["static"] = [static_val_mrr] * len(tau_grid)
    sweep["test_mrr"]["static"] = [static_test_mrr] * len(tau_grid)
    print(f"  static MRR  val={static_val_mrr:.4f}  test={static_test_mrr:.4f}")

    for form in forms[1:]:
        v_vals, t_vals = [], []
        for tau in tau_grid:
            v_vals.append(modulated_mrr(
                val_pos_table, val_neg_table, val_du, val_dv,
                val_dv_neg_per_edge, form, tau))
            t_vals.append(modulated_mrr(
                test_pos_table, test_neg_table, test_du, test_dv,
                test_dv_neg_per_edge, form, tau))
        sweep["val_mrr"][form] = v_vals
        sweep["test_mrr"][form] = t_vals
        best_i = int(np.argmax(v_vals))
        print(f"  {form:15s} best τ={tau_grid[best_i]:.2e}  "
              f"val={v_vals[best_i]:.4f}  test={t_vals[best_i]:.4f}  "
              f"(Δval vs static: {v_vals[best_i] - static_val_mrr:+.4f})")

    (OUT / "scorer_sweep.json").write_text(json.dumps(sweep, indent=2))
    print(f"\nwrote {OUT/'scorer_sweep.json'}")


if __name__ == "__main__":
    main()
