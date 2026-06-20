"""Strict-causal test-set MRR stratification — gated behind train.py's --stratify.

After a normal training run, this re-seeds the trainer's causal stores over
train+val and RE-RUNS the strict-causal test eval (`Trainer._eval` via its optional
`recorder` hook — no separate scoring loop), capturing per-positive reciprocal rank +
causal metadata (endpoint-seen, pair ever-bit/count, source degree). It then writes
stratum tables + headroom sizing to localize where the model's MRR is lost.

The recorder uses DEDICATED analysis counters (a per-node degree array; a dedicated
PairRecencyStore) seeded by the same update-only train+val pass that re-seeds the
trainer's real stores, then updated AFTER each test batch — mirroring `_eval`'s
strict-causal order exactly. Training is untouched; nothing here feeds gradients.

Public entry point: ``run_stratification(...)``.
"""
import json
import pathlib
from collections import OrderedDict

import numpy as np
import torch

from tempest_walks.pair_store import PairRecencyStore

# Published leaderboard refs, by dataset — only used to print a "gap" line.
LEADERBOARD_REF = {
    "tgbl-wiki": {"name": "TPNet", "test": 0.827, "val": 0.842},
}


# ──────────────────────────────────────────────────────────────────────────
# Analysis recorder — dedicated causal counters, queried pre-ingest / updated post.
# ──────────────────────────────────────────────────────────────────────────
class TestStratRecorder:
    def __init__(self, num_nodes: int):
        self.N = num_nodes
        self.deg = np.zeros(num_nodes, dtype=np.int64)   # cumulative interactions/node
        self.pair = PairRecencyStore(num_nodes)          # dedicated: pair ever-bit/count
        self.rows = []
        self._m = None

    def reset(self):
        self.deg[:] = 0
        self.pair.reset()
        self.rows = []

    # --- strict-causal hooks (called by Trainer._eval) ---
    def before_batch(self, batch):
        """Query the analysis stores for this batch's positives — PRE-ingest."""
        src = np.asarray(batch.src, np.int64)
        tgt = np.asarray(batch.tgt, np.int64)
        ts = np.asarray(batch.ts, np.int64)
        _, count_log = self.pair.query(
            torch.from_numpy(src), torch.from_numpy(tgt)[:, None], torch.from_numpy(ts))
        ever = count_log > 0   # ever-seen ⟺ count>0 ⟺ log1p(count)>0 (PairRecencyStore dropped the ever bit)
        self._m = {
            "u_deg": self.deg[src].copy(),
            "v_deg": self.deg[tgt].copy(),
            "ever": ever.squeeze(1).numpy(),
            "pair_count": np.rint(np.expm1(count_log.squeeze(1).numpy())).astype(np.int64),
        }

    def on_positive(self, batch, i, rr):
        m = self._m
        ud, vd = int(m["u_deg"][i]), int(m["v_deg"][i])
        self.rows.append({
            "rr": float(rr), "u_deg": ud, "v_deg": vd,
            "u_seen": ud > 0, "v_seen": vd > 0,
            "pair_seen": bool(m["ever"][i] > 0), "pair_count": int(m["pair_count"][i]),
        })

    def after_batch(self, batch):
        """Update the analysis stores with this batch's edges (also used to seed)."""
        src = np.asarray(batch.src, np.int64)
        tgt = np.asarray(batch.tgt, np.int64)
        ts = np.asarray(batch.ts, np.int64)
        if src.size:
            np.add.at(self.deg, src, 1)
            np.add.at(self.deg, tgt, 1)
            self.pair.update(src, tgt, ts)


# ──────────────────────────────────────────────────────────────────────────
# Stratification
# ──────────────────────────────────────────────────────────────────────────
def _row(rr, mask, name):
    cnt = int(mask.sum())
    n = rr.shape[0]
    if cnt == 0:
        return {"name": name, "count": 0, "fraction": 0.0, "mean_rr": 0.0,
                "hits1": 0.0, "hits10": 0.0, "contribution": 0.0}
    sub = rr[mask]
    frac = cnt / n
    mean = float(sub.mean())
    return {"name": name, "count": cnt, "fraction": frac, "mean_rr": mean,
            "hits1": float((sub >= 1.0 - 1e-9).mean()),
            "hits10": float((sub >= 0.1 - 1e-9).mean()),
            "contribution": frac * mean}


def stratify(rows):
    rr = np.array([r["rr"] for r in rows], dtype=np.float64)
    u_seen = np.array([r["u_seen"] for r in rows])
    v_seen = np.array([r["v_seen"] for r in rows])
    pair_seen = np.array([r["pair_seen"] for r in rows])
    u_deg = np.array([r["u_deg"] for r in rows], dtype=np.int64)
    n = rr.shape[0]

    both = u_seen & v_seen
    u_ind = (~u_seen) & v_seen      # u inductive (unseen), v seen
    v_ind = u_seen & (~v_seen)
    both_ind = (~u_seen) & (~v_seen)

    transduct = [
        _row(rr, both, "both-seen"),
        _row(rr, u_ind, "u-only-inductive"),
        _row(rr, v_ind, "v-only-inductive"),
        _row(rr, both_ind, "both-inductive"),
    ]
    pair = [
        _row(rr, pair_seen, "repeat-pair"),
        _row(rr, ~pair_seen, "new-pair"),
    ]
    deg_buckets = OrderedDict([
        ("deg=0", u_deg == 0), ("deg=1", u_deg == 1),
        ("deg 2-5", (u_deg >= 2) & (u_deg <= 5)),
        ("deg 6-20", (u_deg >= 6) & (u_deg <= 20)),
        ("deg 21-100", (u_deg >= 21) & (u_deg <= 100)),
        ("deg >100", u_deg > 100),
    ])
    src_deg = [_row(rr, m, name) for name, m in deg_buckets.items()]

    cross = []
    for pname, pmask in [("repeat", pair_seen), ("new", ~pair_seen)]:
        for tname, tmask in [("both-seen", both), ("u-only-ind", u_ind),
                             ("v-only-ind", v_ind), ("both-ind", both_ind)]:
            cross.append(_row(rr, pmask & tmask, f"{pname} x {tname}"))

    both_seen_mean = transduct[0]["mean_rr"]
    return {
        "overall": float(rr.mean()), "n": n, "both_seen_mean": both_seen_mean,
        "transductivity": transduct, "pair_recurrence": pair,
        "source_degree": src_deg, "crosstab_pair_x_transductivity": cross,
    }


def headroom(strata):
    """For each weak stratum (mean_rr < overall), Δ-overall-MRR if lifted to each
    target = fraction · max(0, target − mean_rr)."""
    targets = {"0.30": 0.30, "0.60": 0.60,
               "both-seen": strata["both_seen_mean"]}
    overall = strata["overall"]
    out = []
    seen = set()
    for part in ("transductivity", "pair_recurrence", "crosstab_pair_x_transductivity"):
        for s in strata[part]:
            if s["count"] == 0 or s["mean_rr"] >= overall or s["name"] in seen:
                continue
            seen.add(s["name"])
            out.append({
                "stratum": s["name"], "fraction": s["fraction"],
                "mean_rr": s["mean_rr"], "count": s["count"],
                "deltas": {k: s["fraction"] * max(0.0, t - s["mean_rr"])
                           for k, t in targets.items()},
            })
    out.sort(key=lambda r: r["deltas"]["both-seen"], reverse=True)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Emit
# ──────────────────────────────────────────────────────────────────────────
def _table(rows):
    head = "| stratum | count | frac | mean_rr | hits@1 | hits@10 | contrib |\n"
    head += "|---|---|---|---|---|---|---|\n"
    for s in rows:
        head += (f"| {s['name']} | {s['count']:,} | {s['fraction']:.3f} | "
                 f"{s['mean_rr']:.4f} | {s['hits1']:.3f} | {s['hits10']:.3f} | "
                 f"{s['contribution']:.4f} |\n")
    return head


def emit_md(strata, hr, meta, ref=None):
    gap_line = ""
    if ref is not None:
        g_test = ref["test"] - strata["overall"]
        gap_line = (f"- **{ref['name']} ref: test {ref['test']} / val {ref['val']}  →  "
                    f"test gap ≈ {g_test:+.4f}**\n")
    md = f"""# {meta['dataset']} test-set MRR stratification

**Model:** {meta.get('head', 'current head')}, sphere E.
seed {meta['seed']}, d_emb {meta['d_emb']}, train bs {meta['batch_size']}
/ eval bs {meta['eval_batch_size']}, best epoch {meta['best_epoch']}.

- **test MRR (this stratified run): {strata['overall']:.4f}** over {strata['n']:,} positives
- training-run best: val {meta['best_val']:.4f} / test {meta['best_test']:.4f} (walk-noise vs above)
{gap_line}
## 1. Transductivity (endpoint seen in any prior edge)
{_table(strata['transductivity'])}
## 2. Pair recurrence
{_table(strata['pair_recurrence'])}
## 3. Source-degree buckets (u cumulative interactions)
{_table(strata['source_degree'])}
## 4. Cross-tab: pair-recurrence × transductivity (decisive)
{_table(strata['crosstab_pair_x_transductivity'])}
## Headroom sizing — Δ overall-MRR if the weak stratum's mean_rr were lifted

| stratum | frac | mean_rr | →0.30 | →0.60 | →both-seen ({strata['both_seen_mean']:.3f}) |
|---|---|---|---|---|---|
"""
    for h in hr:
        d = h["deltas"]
        md += (f"| {h['stratum']} | {h['fraction']:.3f} | {h['mean_rr']:.4f} | "
               f"{d['0.30']:+.4f} | {d['0.60']:+.4f} | {d['both-seen']:+.4f} |\n")
    return md


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────
def run_stratification(trainer, train_f, val_f, test_eval, test_f,
                       num_nodes, meta, out_dir="logs/stratify"):
    """Re-seed the trainer's causal stores over train+val, then run the stratified
    strict-causal test eval and write tables. Returns (strata, headroom).

    Call AFTER `trainer.train(...)` (which restores best-val weights). Assumes the
    trainer exposes `walk_gen`, `pair_store` (optional), `node_last`, and the
    `_eval(evaluator, batches, recorder=...)` hook.
    """
    print("\n=== Re-seeding train+val, then stratified test eval ===")
    trainer.walk_gen.reset()
    if getattr(trainer, "pair_store", None) is not None:
        trainer.pair_store.reset()
    trainer.node_last.reset()

    rec = TestStratRecorder(num_nodes)
    rec.reset()
    for fac in (train_f, val_f):
        for batch in fac():
            trainer.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
            if getattr(trainer, "pair_store", None) is not None:
                trainer.pair_store.update(batch.src, batch.tgt, batch.ts)
            trainer.node_last.update(batch.src, batch.tgt, batch.ts)
            rec.after_batch(batch)

    test_mrr = trainer._eval(test_eval, test_f(), recorder=rec)
    strata = stratify(rec.rows)
    hr = headroom(strata)

    # ── Sanity: capture is complete and partitions reconstruct the overall mean ──
    assert abs(strata["overall"] - test_mrr) < 1e-4, \
        f"capture incomplete: {strata['overall']} vs {test_mrr}"
    for part in ("transductivity", "pair_recurrence", "source_degree",
                 "crosstab_pair_x_transductivity"):
        rows = strata[part]
        assert sum(s["count"] for s in rows) == strata["n"], f"{part} counts != total"
        recon = sum(s["contribution"] for s in rows)
        assert abs(recon - strata["overall"]) < 1e-4, f"{part} reconstruct {recon}"
    print(f"  sanity OK — test MRR {strata['overall']:.4f} over {strata['n']:,} positives")

    ref = LEADERBOARD_REF.get(meta["dataset"])
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{meta['dataset']}_seed{meta['seed']}_strata"
    (out / f"{stem}.md").write_text(emit_md(strata, hr, meta, ref=ref))
    json.dump({"config": meta, "test_mrr_stratified": strata["overall"],
               "n_positives": strata["n"], "leaderboard_ref": ref,
               "strata": {k: strata[k] for k in
                          ("transductivity", "pair_recurrence", "source_degree",
                           "crosstab_pair_x_transductivity")},
               "both_seen_mean": strata["both_seen_mean"], "headroom": hr},
              open(out / f"{stem}.json", "w"), indent=2)
    print(f"  wrote {out}/{stem}.{{md,json}}")
    return strata, hr
