"""Phase 5 — parametric discriminators on E.

How much MRR signal can a *parametric* similarity recover, beyond raw cos?

We try four:
    LIN     logistic-regression coefficients on per-edge feature vector
            φ(u, v) = [E[u]*E[v], (E[u]-E[v])^2, |E[u]-E[v]|]
            -> scalar logit. Linear in 3*d_emb.
    BIL     bilinear u^T W v with low-rank W = A A^T (rank r).
    MAH     Mahalanobis distance on E[u] - E[v] using Σ from positives.
    PROJ    cosine in PCA top-k subspace (vary k).

We fit on VAL positives + per-edge negatives (so the discriminator can
calibrate against the actual negative distribution it'll be ranked against),
report per-edge MRR on TEST (no leakage). For LIN and BIL we use SGD with
small batches; for MAH we just fit the positive-pair difference covariance
and use the inverse-cov norm.

Outputs (analysis/phase5/):
    fits.npz         learned parameters (W_lin, W_bil, Σ_inv, ...)
    summary.json     val/test MRR per method
"""
import json
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EMB_PATH = ROOT / "logs" / "embeddings" / "tgbl-wiki_seed42_demb128_ep33.npy"
OUT = ROOT / "analysis" / "phase5"
OUT.mkdir(parents=True, exist_ok=True)


def _rank_pos(sp: float, sn: np.ndarray) -> float:
    higher = int((sn > sp).sum())
    equal = int((sn == sp).sum())
    return higher + (equal + 2) / 2


def build_pos_neg(loaded, split_name, ns, E, max_pairs=None):
    """Return positive and negative (u, v) index lists.

    For each split edge with TGB's per-edge neg list, we sample one
    negative per edge for training (`max_pairs` caps total positives if
    set). For eval, we score the FULL list of 999 negatives per edge.
    """
    split = getattr(loaded, split_name)
    raw = ns.query_batch(split.sources, split.destinations, split.timestamps,
                         split_mode=split_name)
    neg_list = [np.asarray(x, dtype=np.int64) for x in raw]
    return split.sources.astype(np.int64), split.destinations.astype(np.int64), neg_list


def features(E_u, E_v):
    """φ(u, v) = [E_u*E_v ; (E_u - E_v)^2 ; |E_u - E_v|].
    Inputs can be batched [N, d]; output [N, 3d]."""
    h = E_u * E_v
    d = E_u - E_v
    return np.concatenate([h, d * d, np.abs(d)], axis=-1).astype(np.float32)


def lin_fit(loaded, ns, E, n_neg_per_pos=4, n_epochs=8, lr=0.01):
    """Logistic regression on φ(u, v) with positives from val edges and
    n_neg sampled negatives per positive (from TGB's per-edge neg list).
    Trained with simple SGD; bias term included."""
    src, tgt, neg_list = build_pos_neg(loaded, "val", ns, E)
    N = len(src)
    feat_dim = 3 * E.shape[1]
    W = np.zeros(feat_dim, dtype=np.float32)
    b = np.float32(0.0)
    rng = np.random.default_rng(42)

    pos_feats = features(E[src], E[tgt])
    print(f"  LIN: {N} positives, feat_dim={feat_dim}")

    for ep in range(n_epochs):
        perm = rng.permutation(N)
        bsz = 512
        total = 0.0
        for i in range(0, N, bsz):
            batch_idx = perm[i:i+bsz]
            pf = pos_feats[batch_idx]
            # Sample negs per positive
            nfs = []
            for j in batch_idx:
                nidx = rng.choice(neg_list[j], size=n_neg_per_pos, replace=False)
                nfs.append(features(E[src[j]][None, :].repeat(n_neg_per_pos, 0),
                                    E[nidx]))
            nf = np.concatenate(nfs, axis=0)  # [B*n_neg, feat_dim]
            X = np.concatenate([pf, nf], axis=0)
            y = np.concatenate([np.ones(len(pf), dtype=np.float32),
                                np.zeros(len(nf), dtype=np.float32)])
            z = X @ W + b
            sig = 1.0 / (1.0 + np.exp(-z))
            err = sig - y
            grad_W = X.T @ err / len(X)
            grad_b = float(err.mean())
            W -= lr * grad_W
            b -= lr * grad_b
            # Cross-entropy
            eps = 1e-8
            total += float(-(y * np.log(sig + eps) + (1 - y) * np.log(1 - sig + eps)).mean())
        if ep == 0 or (ep + 1) % 2 == 0:
            print(f"  LIN ep{ep+1}: train CE {total/(N/bsz):.4f}")
    return W, b


def bil_fit(loaded, ns, E, rank=16, n_neg=4, n_epochs=6, lr=0.05):
    """Bilinear score(u, v) = u^T (A A^T) v. Learn A ∈ R^{d×r}."""
    src, tgt, neg_list = build_pos_neg(loaded, "val", ns, E)
    N = len(src)
    d = E.shape[1]
    rng = np.random.default_rng(7)
    A = rng.standard_normal((d, rank)).astype(np.float32) * 0.1
    print(f"  BIL: {N} positives, rank={rank}")

    for ep in range(n_epochs):
        perm = rng.permutation(N)
        bsz = 512
        total = 0.0
        for i in range(0, N, bsz):
            batch_idx = perm[i:i+bsz]
            Eu = E[src[batch_idx]]
            Ev_pos = E[tgt[batch_idx]]
            # Sample one neg per positive
            neg_idx = np.array([rng.choice(neg_list[j]) for j in batch_idx])
            Ev_neg = E[neg_idx]
            # Score (logistic ranking)
            uA = Eu @ A             # [B, r]
            spos = (uA * (Ev_pos @ A)).sum(axis=1)  # [B]
            sneg = (uA * (Ev_neg @ A)).sum(axis=1)
            # Pairwise hinge / softplus on (spos - sneg)
            diff = sneg - spos
            sig = 1.0 / (1.0 + np.exp(-diff))
            # Gradient of softplus(sneg - spos) wrt A:
            # d/dA [u^T A A^T v_neg - u^T A A^T v_pos]
            #  = (v_neg u^T + u v_neg^T - v_pos u^T - u v_pos^T) A
            # For batches, sum row-by-row.
            grad_A_each = (
                Eu[:, :, None] * (Ev_neg - Ev_pos)[:, None, :]
                + (Ev_neg - Ev_pos)[:, :, None] * Eu[:, None, :]
            )  # [B, d, d] — but we want grad wrt A: factor through.
            # Easier: compute u^T A first, neg-pos
            # Actually use simpler: spos = (u^T A)(A^T v_pos), derivative wrt A is
            #    d spos/dA = u * (A^T v_pos)^T + v_pos * (A^T u)^T  (an outer)
            # Let m = sigmoid(diff) → loss gradient is m * (∂ diff / ∂ A)
            # ∂diff/∂A = ∂sneg/∂A - ∂spos/∂A
            #          = u * v_neg^T A + v_neg * u^T A - (u * v_pos^T A + v_pos * u^T A)
            # Combine:
            vdiff = Ev_neg - Ev_pos             # [B, d]
            grad_A = (
                Eu.T @ (sig[:, None] * (vdiff @ A))
                + vdiff.T @ (sig[:, None] * (Eu @ A))
            ) / len(Eu)
            A -= lr * grad_A
            total += float(np.log1p(np.exp(diff)).mean())
        if ep == 0 or (ep + 1) % 2 == 0:
            print(f"  BIL ep{ep+1}: ranking loss {total/(N/bsz):.4f}")
    return A


def mah_fit(loaded, ns, E, reg=1e-3):
    """Fit a Mahalanobis-style scoring metric using the cov of positive
    pair differences. score(u, v) = -(u-v)^T Σ^{-1} (u-v)."""
    src, tgt, _ = build_pos_neg(loaded, "val", ns, E)
    d = E.shape[1]
    diff = (E[src] - E[tgt]).astype(np.float64)
    diff -= diff.mean(axis=0, keepdims=True)
    Sigma = diff.T @ diff / len(diff) + reg * np.eye(d)
    Sigma_inv = np.linalg.inv(Sigma).astype(np.float32)
    print(f"  MAH: Sigma fit on {len(diff)} pos diffs, reg={reg}")
    return Sigma_inv


def proj_fit(E, k):
    """Top-k PCA projection, applied to (E centered)."""
    Ec = E.astype(np.float64) - E.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    P = Vt[:k]   # [k, d]
    return P.astype(np.float32)


def score_split_lin(W, b, E, src, tgt, neg_list):
    rr = []
    eps_arr = features(E[src], E[tgt]) @ W + b
    for i, ndst in enumerate(neg_list):
        Ev_neg = E[ndst]
        Eu_rep = np.repeat(E[src[i]][None, :], len(ndst), axis=0)
        nf = features(Eu_rep, Ev_neg) @ W + b
        rr.append(1.0 / _rank_pos(float(eps_arr[i]), nf))
    return float(np.mean(rr))


def score_split_bil(A, E, src, tgt, neg_list):
    rr = []
    Eu = E[src]
    uA = Eu @ A
    Ev = E[tgt]
    spos = (uA * (Ev @ A)).sum(axis=1)
    for i, ndst in enumerate(neg_list):
        Ev_neg = E[ndst]
        sn = (uA[i] @ (Ev_neg @ A).T)
        rr.append(1.0 / _rank_pos(float(spos[i]), sn))
    return float(np.mean(rr))


def score_split_mah(Sigma_inv, E, src, tgt, neg_list):
    rr = []
    for i, ndst in enumerate(neg_list):
        u = E[src[i]]
        v_pos = E[tgt[i]]
        d_pos = u - v_pos
        sp = -float(d_pos @ Sigma_inv @ d_pos)
        Ev_neg = E[ndst]
        d_neg = u[None, :] - Ev_neg
        # batched quadratic form: diag(d_neg @ Sigma_inv @ d_neg.T)
        tmp = d_neg @ Sigma_inv
        sn = -np.einsum("ij,ij->i", tmp, d_neg)
        rr.append(1.0 / _rank_pos(sp, sn))
    return float(np.mean(rr))


def score_split_proj(P, E, src, tgt, neg_list):
    eps = 1e-12
    Ep = E @ P.T   # [N, k]
    Ep_norm = np.linalg.norm(Ep, axis=1, keepdims=True).clip(min=eps)
    Ep_n = Ep / Ep_norm
    rr = []
    for i, ndst in enumerate(neg_list):
        u_n = Ep_n[src[i]]
        v_pos_n = Ep_n[tgt[i]]
        sp = float(u_n @ v_pos_n)
        sn = Ep_n[ndst] @ u_n
        rr.append(1.0 / _rank_pos(sp, sn))
    return float(np.mean(rr))


def main():
    from link_property_prediction.data import load_tgb
    E = np.load(EMB_PATH).astype(np.float32)
    loaded = load_tgb("tgbl-wiki", root=str(ROOT / "datasets"))
    loaded.dataset.load_val_ns()
    loaded.dataset.load_test_ns()
    ns = loaded.dataset.negative_sampler

    # Prepare test eval lists
    test_raw = ns.query_batch(loaded.test.sources, loaded.test.destinations,
                              loaded.test.timestamps, split_mode="test")
    test_neg = [np.asarray(x, dtype=np.int64) for x in test_raw]
    test_src = loaded.test.sources.astype(np.int64)
    test_tgt = loaded.test.destinations.astype(np.int64)

    val_raw = ns.query_batch(loaded.val.sources, loaded.val.destinations,
                             loaded.val.timestamps, split_mode="val")
    val_neg = [np.asarray(x, dtype=np.int64) for x in val_raw]
    val_src = loaded.val.sources.astype(np.int64)
    val_tgt = loaded.val.destinations.astype(np.int64)

    summary = {}

    print("\n--- LIN ---")
    W, b = lin_fit(loaded, ns, E)
    np.savez(OUT / "lin.npz", W=W, b=b)
    summary["lin_val"] = score_split_lin(W, b, E, val_src, val_tgt, val_neg)
    summary["lin_test"] = score_split_lin(W, b, E, test_src, test_tgt, test_neg)
    print(f"  LIN val MRR={summary['lin_val']:.4f}  test MRR={summary['lin_test']:.4f}")

    for rank in (16, 32, 64):
        print(f"\n--- BIL rank={rank} ---")
        A = bil_fit(loaded, ns, E, rank=rank)
        np.savez(OUT / f"bil_r{rank}.npz", A=A)
        summary[f"bil_r{rank}_val"] = score_split_bil(A, E, val_src, val_tgt, val_neg)
        summary[f"bil_r{rank}_test"] = score_split_bil(A, E, test_src, test_tgt, test_neg)
        print(f"  BIL r{rank} val MRR={summary[f'bil_r{rank}_val']:.4f}  "
              f"test MRR={summary[f'bil_r{rank}_test']:.4f}")

    print("\n--- MAH ---")
    Sinv = mah_fit(loaded, ns, E)
    np.save(OUT / "mah_sigma_inv.npy", Sinv)
    summary["mah_val"] = score_split_mah(Sinv, E, val_src, val_tgt, val_neg)
    summary["mah_test"] = score_split_mah(Sinv, E, test_src, test_tgt, test_neg)
    print(f"  MAH val MRR={summary['mah_val']:.4f}  test MRR={summary['mah_test']:.4f}")

    for k in (16, 32, 64, 86, 128):
        print(f"\n--- PROJ k={k} ---")
        P = proj_fit(E, k)
        summary[f"proj_k{k}_val"] = score_split_proj(P, E, val_src, val_tgt, val_neg)
        summary[f"proj_k{k}_test"] = score_split_proj(P, E, test_src, test_tgt, test_neg)
        print(f"  PROJ k={k} val MRR={summary[f'proj_k{k}_val']:.4f}  "
              f"test MRR={summary[f'proj_k{k}_test']:.4f}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
