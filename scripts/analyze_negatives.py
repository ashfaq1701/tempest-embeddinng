"""Characterise the TGB eval-negative distribution per dataset.

TGB generates eval negatives with the "hist_rnd" strategy: a designed mix of
HISTORICAL negatives (destinations the *source* has linked to before, drawn
from the training reference) and RANDOM negatives. For each dataset we measure,
over a sample of validation positives:

  negs/pos          : negatives served per positive (TGB-fixed per dataset)
  % hist (src)      : negatives n where (src, n) is a TRUE train edge — TGB's
                      own 'historical' definition (src-specific). This is the
                      train/eval mismatch axis: we TRAIN on all-random negs.
  % popular (glob)  : of the rest, negatives that were *some* destination in
                      train (globally seen) but not a partner of THIS src
  % novel (random)  : negatives never seen as a destination in train
  pos recurrence    : positives whose (src, dst) is a prior train edge — the
                      false-negative-risk driver (high ⇒ historical negs are
                      mostly future positives ⇒ mining them backfires)

Method notes (correctness):
  - Historical reference = the TRAIN split (edges before val starts), which is
    exactly the reference TGB uses to construct val negatives.
  - Per-source history is built only for the sampled sources (vectorised
    np.isin over all edges), so it is exact, not approximate, and cheap even on
    the 44M-edge sets.
  - query_batch is the official TGB sampler; negatives keep the positive's src
    by construction (it replaces the destination only).
"""
import sys
import numpy as np
from collections import defaultdict
from tgb.linkproppred.dataset import LinkPropPredDataset

N_SAMPLE = 2000


def analyse(name):
    ds = LinkPropPredDataset(name=name, root="datasets", preprocess=True)
    d = ds.full_data
    src = np.asarray(d["sources"]); dst = np.asarray(d["destinations"])
    t = np.asarray(d["timestamps"])
    ds.load_val_ns()
    vm = ds.val_mask
    val_start = int(np.argmax(vm))
    v_idx = np.where(vm)[0]
    rng = np.random.default_rng(0)
    sel = v_idx if len(v_idx) <= N_SAMPLE else rng.choice(v_idx, N_SAMPLE, replace=False)
    sel.sort()
    vs, vd, vt = src[sel], dst[sel], t[sel]

    # train reference
    tr_src, tr_dst = src[:val_start], dst[:val_start]
    global_dst = set(int(x) for x in np.unique(tr_dst))      # seen-as-destination
    ssrc = np.unique(vs)
    keep = np.isin(tr_src, ssrc)
    src_hist = defaultdict(set)                              # src -> {train dsts}
    for s, dd in zip(tr_src[keep], tr_dst[keep]):
        src_hist[int(s)].add(int(dd))

    neg = ds.negative_sampler.query_batch(vs, vd, vt, split_mode="val")

    n_hist = n_pop = n_novel = n_tot = 0
    counts = []
    pos_rec = 0
    for k in range(len(vs)):
        s = int(vs[k]); h = src_hist.get(s, set())
        pos_rec += int(int(vd[k]) in h)
        arr = np.asarray(neg[k]).astype(np.int64)
        counts.append(len(arr))
        for n in arr:
            n = int(n); n_tot += 1
            if n in h:
                n_hist += 1
            elif n in global_dst:
                n_pop += 1
            else:
                n_novel += 1
    c = np.asarray(counts)
    pct = lambda x: 100.0 * x / max(n_tot, 1)
    print(
        f"{name:14s} | negs/pos {c.mean():5.0f}"
        f" (min {c.min()}, max {c.max()}) | "
        f"hist(src) {pct(n_hist):5.1f}% | popular {pct(n_pop):5.1f}% | "
        f"novel {pct(n_novel):5.1f}% | pos-recurrence {100*pos_rec/len(vs):4.0f}%"
        f" | edges {len(src)/1e6:.1f}M",
        flush=True,
    )


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        try:
            analyse(nm)
        except Exception as e:
            print(f"{nm:14s} | ERROR {type(e).__name__}: {str(e)[:90]}", flush=True)
    print("NS ANALYSIS DONE", flush=True)
