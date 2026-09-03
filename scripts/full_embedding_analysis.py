"""Full analytical study of a saved Poincare-ball embedding E vs the shipped test edges.

Answers, with held-out validation and the EXACT TGB-Seq MRR:
  Part 1  How much of the model's ranking does static E alone explain? (analytical decoders)
  Part 2  What is E's organization? (delta-hyperbolicity, radius=depth, angular communities)
  Part 3  What drives an edge? (logistic attribution over interpretable E-features)

Only E was exported (no pooler / geo_temp / pop_bias / walk sampler), so this measures the
STATIC-E ceiling and the organization, not a reproduction of the model's exact scores.

Usage: python scripts/full_embedding_analysis.py <emb.npy> [csv] [test_ns.npy] [out_prefix]
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression

emb_path = sys.argv[1]
csv_path = sys.argv[2] if len(sys.argv) > 2 else "datasets/YouTube/YouTube.csv"
tns_path = sys.argv[3] if len(sys.argv) > 3 else "datasets/YouTube/YouTube_test_ns.npy"
out_pref = sys.argv[4] if len(sys.argv) > 4 else "logs/plots/youtube_pop"
MODEL_TEST_MRR = 0.6016   # the run this embedding came from (for the gap-to-substrate readout)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
E = torch.from_numpy(np.load(emb_path)).float().to(dev)          # [N, d]
N, d = E.shape
sq = (E * E).sum(-1)                                             # ||x||^2  [N]
one_m = (1.0 - sq).clamp_min(1e-9)                              # 1 - ||x||^2
radius = torch.arccosh((1 + sq / one_m * 2).clamp_min(1.0)).cpu().numpy()  # dist0 = hyperbolic radius

df = pd.read_csv(csv_path)
train, val, test = df[df.split == 0], df[df.split == 1], df[df.split == 2]

# TRAIN degree (undirected), what the model saw.
deg = np.zeros(N, dtype=np.int64)
for col in ("src", "dst"):
    idx, cnt = np.unique(train[col].to_numpy(), return_counts=True)
    m = idx < N
    deg[idx[m]] += cnt[m]
logdeg = np.log1p(deg).astype(np.float32)
logdeg_t = torch.from_numpy(logdeg).to(dev)


def dH_pairs(a_ids, b_ids):
    """Hyperbolic distance between E[a_ids] and E[b_ids], elementwise. a_ids,b_ids same shape."""
    xa, xb = E[a_ids], E[b_ids]
    num = ((xa - xb) ** 2).sum(-1)
    dd = 1.0 + 2.0 * num / (one_m[a_ids] * one_m[b_ids])
    return torch.arccosh(dd.clamp_min(1.0))


def score_split(u_ids, cand_ids, decoder, lam=0.0):
    """cand_ids [M] or [M,K]; u_ids [M] broadcast over K. Returns score, same shape as cand_ids."""
    u_b = u_ids.unsqueeze(-1).expand_as(cand_ids) if cand_ids.dim() == 2 else u_ids
    if decoder in ("dH", "gromov", "popdH"):
        dh = dH_pairs(u_b.reshape(-1), cand_ids.reshape(-1)).reshape(cand_ids.shape)
        if decoder == "dH":
            s = -dh
        elif decoder == "gromov":
            s = 0.5 * (radius_t[u_b] + radius_t[cand_ids] - dh)
        else:  # popdH: -dH + lam*logdeg(v)
            s = -dh + lam * logdeg_t[cand_ids]
    elif decoder == "deg":
        s = logdeg_t[cand_ids]
    elif decoder == "eucl":
        u_e, c_e = E[u_b.reshape(-1)], E[cand_ids.reshape(-1)]
        s = (-((u_e - c_e) ** 2).sum(-1)).reshape(cand_ids.shape)
    return s


radius_t = torch.from_numpy(radius).to(dev)


def mrr_official(pos, neg):
    """Exact TGB-Seq MRR. pos [M], neg [M,K] numpy."""
    pos = pos.reshape(-1, 1)
    opt = (neg > pos).sum(1)
    pess = (neg >= pos).sum(1)
    rank = 0.5 * (opt + pess) + 1.0
    return (1.0 / rank).mean()


def eval_decoder(u_ids, pos_ids, neg_ids, decoder, lam=0.0, chunk=4000):
    """Chunked MRR over a split. u/pos [M], neg [M,K] (torch, on dev)."""
    M = u_ids.shape[0]
    pos_all, neg_all = [], []
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        ps = score_split(u_ids[s:e], pos_ids[s:e], decoder, lam)
        ns = score_split(u_ids[s:e], neg_ids[s:e], decoder, lam)
        pos_all.append(ps.detach().cpu().numpy())
        neg_all.append(ns.detach().cpu().numpy())
    return mrr_official(np.concatenate(pos_all), np.concatenate(neg_all))


# ---- Build test tensors (test_ns row-aligned to chronological test order) ----
test_u = torch.from_numpy(test.src.to_numpy().astype(np.int64)).to(dev)
test_v = torch.from_numpy(test.dst.to_numpy().astype(np.int64)).to(dev)
test_neg = torch.from_numpy(np.load(tns_path).astype(np.int64)).to(dev)     # [T,100]
assert test_neg.shape[0] == test_u.shape[0], (test_neg.shape, test_u.shape)

# ---- Build val negatives with the SAME sampler+seed the trainer used (for lambda fit) ----
from link_property_prediction.data import Batch
from link_property_prediction.negatives import UniformNegativeSampler

dst_pool = np.unique(np.concatenate([train.src.to_numpy(), train.dst.to_numpy()])).astype(np.int32)
val_batch = Batch(src=val.src.to_numpy(), tgt=val.dst.to_numpy(), ts=val.time.to_numpy(), edge_feat=None)
_, val_neg_np = UniformNegativeSampler(num_neg_per_pos=100, dst_pool=dst_pool, seed=42).sample(val_batch)
val_u = torch.from_numpy(val.src.to_numpy().astype(np.int64)).to(dev)
val_v = torch.from_numpy(val.dst.to_numpy().astype(np.int64)).to(dev)
val_neg = torch.from_numpy(val_neg_np.astype(np.int64)).to(dev)

print("=== PART 1: analytical decoder MRR ladder (test) ===")
res = {}
res["D1_dH"] = float(eval_decoder(test_u, test_v, test_neg, "dH"))
res["D3_gromov"] = float(eval_decoder(test_u, test_v, test_neg, "gromov"))
res["D4_deg"] = float(eval_decoder(test_u, test_v, test_neg, "deg"))
res["D5_eucl"] = float(eval_decoder(test_u, test_v, test_neg, "eucl"))
# D2: fit lambda on VAL, report on TEST
lam_grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
val_mrrs = {lam: float(eval_decoder(val_u, val_v, val_neg, "popdH", lam)) for lam in lam_grid}
best_lam = max(val_mrrs, key=val_mrrs.get)
res["D2_popdH_test"] = float(eval_decoder(test_u, test_v, test_neg, "popdH", best_lam))
res["D2_best_lambda"] = best_lam
res["D2_val_curve"] = val_mrrs
res["MODEL_test_mrr"] = MODEL_TEST_MRR
for k in ("D4_deg", "D5_eucl", "D1_dH", "D3_gromov", "D2_popdH_test", "MODEL_test_mrr"):
    print(f"  {k:18s} {res[k]:.4f}")
print(f"  D2 best lambda={best_lam}  val curve={val_mrrs}")

print("=== PART 2: organization ===")
# 2a delta-hyperbolicity on a sampled node set (Gromov 4-point, relative to diameter)
rng = np.random.default_rng(0)
S = rng.choice(N, size=1500, replace=False)
Es = E[torch.from_numpy(S).to(dev)]
sqs = (Es * Es).sum(-1)
oms = (1 - sqs).clamp_min(1e-9)
D = torch.arccosh((1 + 2 * torch.cdist(Es, Es) ** 2 / oms[:, None] / oms[None, :]).clamp_min(1.0))
Dnp = D.cpu().numpy()
diam = Dnp.max()
q = rng.integers(0, len(S), size=(200000, 4))
d_xy = Dnp[q[:, 0], q[:, 1]] + Dnp[q[:, 2], q[:, 3]]
d_xz = Dnp[q[:, 0], q[:, 2]] + Dnp[q[:, 1], q[:, 3]]
d_xw = Dnp[q[:, 0], q[:, 3]] + Dnp[q[:, 1], q[:, 2]]
sums = np.sort(np.stack([d_xy, d_xz, d_xw], 1), axis=1)
delta = 0.5 * (sums[:, 2] - sums[:, 1])
res["delta_abs"] = float(delta.max())
res["delta_rel"] = float(delta.max() / diam)
res["delta_mean_rel"] = float(delta.mean() / diam)
res["diam_sample"] = float(diam)
print(f"  delta_abs={res['delta_abs']:.3f}  delta/diam={res['delta_rel']:.3f}  "
      f"mean(delta)/diam={res['delta_mean_rel']:.3f}  (0=tree, ~0.5=non-hyperbolic)")

# 2b angular communities via spherical-ish k-means on unit directions
dirs = (E / E.norm(dim=-1, keepdim=True).clamp_min(1e-9)).cpu().numpy()
K = 100
km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=3, batch_size=10000).fit(dirs)
lab = km.labels_
# same-community rate for test positives vs their negatives
tu = test.src.to_numpy(); tv = test.dst.to_numpy()
tn = np.load(tns_path)
same_pos = (lab[tu] == lab[tv]).mean()
same_neg = (lab[tu][:, None] == lab[tn]).mean()
res["community_K"] = K
res["same_comm_pos"] = float(same_pos)
res["same_comm_neg"] = float(same_neg)
res["same_comm_lift"] = float(same_pos / max(same_neg, 1e-9))
print(f"  P(same community | test edge)={same_pos:.4f}  vs  | negative={same_neg:.4f}  "
      f"lift={res['same_comm_lift']:.2f}x")

print("=== PART 3: logistic attribution (test edge vs one sampled negative) ===")
M = min(80000, len(tu))
sel = rng.choice(len(tu), size=M, replace=False)
u = torch.from_numpy(tu[sel].astype(np.int64)).to(dev)
vpos = torch.from_numpy(tv[sel].astype(np.int64)).to(dev)
jneg = rng.integers(0, 100, size=M)
vneg = torch.from_numpy(tn[sel, jneg].astype(np.int64)).to(dev)


def feats(u_ids, v_ids):
    dh = dH_pairs(u_ids, v_ids).cpu().numpy()
    ru, rv = radius[u_ids.cpu().numpy()], radius[v_ids.cpu().numpy()]
    grom = 0.5 * (ru + rv - dh)
    ld = logdeg[v_ids.cpu().numpy()]
    sc = (lab[u_ids.cpu().numpy()] == lab[v_ids.cpu().numpy()]).astype(np.float32)
    return np.stack([dh, grom, ru, rv, ld, sc], 1)


Xp, Xn = feats(u, vpos), feats(u, vneg)
X = np.vstack([Xp, Xn]); y = np.r_[np.ones(M), np.zeros(M)]
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd
lr = LogisticRegression(max_iter=1000, C=1.0).fit(Xs, y)
fnames = ["d_H(u,v)", "gromov(u|v)", "r_u", "r_v", "log_deg(v)", "same_comm"]
coefs = {fnames[i]: float(lr.coef_[0][i]) for i in range(len(fnames))}
res["attribution_std_coefs"] = coefs
res["attribution_train_acc"] = float(lr.score(Xs, y))
print("  standardized logistic coefficients (edge vs negative):")
for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
    print(f"    {k:14s} {v:+.3f}")
print(f"  separability acc={res['attribution_train_acc']:.3f}")

# ---- summary figure ----
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ladder = [("degree", res["D4_deg"]), ("Euclid -||.||^2", res["D5_eucl"]), ("-d_H", res["D1_dH"]),
          ("gromov", res["D3_gromov"]), (f"-d_H+{best_lam}logdeg", res["D2_popdH_test"]),
          ("MODEL (walks+pop)", MODEL_TEST_MRR)]
cols = ["#9aa0a6"] * 5 + ["#d8663b"]
ax[0].barh([x[0] for x in ladder], [x[1] for x in ladder], color=cols)
for i, (_, vv) in enumerate(ladder):
    ax[0].text(vv + 0.005, i, f"{vv:.3f}", va="center", fontsize=9)
ax[0].set(title="(1) analytical E-only decoders vs model  (test MRR)", xlabel="MRR", xlim=(0, 0.66))
ax[1].bar(list(val_mrrs.keys()), list(val_mrrs.values()), width=0.4 * (max(lam_grid) / len(lam_grid) + 0.3), color="#3b7dd8")
ax[1].axvline(best_lam, color="k", ls="--", lw=1, label=f"best lambda={best_lam}")
ax[1].set(title="(2) D2 popularity-prior weight (val MRR)", xlabel="lambda (log-degree weight)", ylabel="val MRR")
ax[1].legend()
order = sorted(coefs.items(), key=lambda kv: kv[1])
ax[2].barh([k for k, _ in order], [v for _, v in order],
           color=["#3b7dd8" if v < 0 else "#d8663b" for _, v in order])
ax[2].axvline(0, color="k", lw=0.8)
ax[2].set(title="(3) what drives a test edge (std. logistic coef)", xlabel="coefficient (+ = more likely edge)")
fig.tight_layout()
fig.savefig(out_pref + "_reconstruction.png", dpi=110)
print("saved", out_pref + "_reconstruction.png")

with open(out_pref + "_analysis.json", "w") as f:
    json.dump(res, f, indent=2)
print("saved", out_pref + "_analysis.json")
