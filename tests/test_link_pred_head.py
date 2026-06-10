"""LinkPredHead (order-aware recency pool) correctness.

Invariants pinned:
  1. Mask-aware — padded positions never contribute to the score.
  2. Cold-start — a fully-padded walk row contributes 0 and does not NaN.
  3. Recency weighting — the per-hop ω kernel down/up-weights positions as set;
     pushing one hop's ω up makes that position dominate the pool.
  4. Candidate-independent kernel — ω/pooling do not depend on E_v (the
     candidate enters only through the cos dot product).
  5. Learnability — drives CE well below log(101) on a planted positive
     (the scale fix prevents the flat-softmax stall).
"""
import math

import torch
import torch.nn.functional as F

from tempest_walks.link_pred_head import LinkPredHead


def _walks(B=2, W=3, L=6, d=16, valid=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    E_walks = F.normalize(torch.randn(B, W, L, d, generator=g), dim=-1)
    mask = torch.zeros(B, W, L, dtype=torch.bool)
    mask[:, :, :valid] = True
    K_idx = torch.arange(L)[None, None, :].expand(B, W, L).clone().long()
    t_feat = torch.zeros(B, W, L, 12)
    return dict(E_walks=E_walks, mask=mask, K_idx=K_idx, t_feat=t_feat)


def test_exposes_time_encoder_for_trainer():
    # The trainer builds the walks' t_feat via link_head.time_encoder; every
    # head must expose it even if (like this one) it ignores t_feat in scoring.
    head = LinkPredHead(max_walk_len=6)
    assert hasattr(head, "time_encoder")
    assert hasattr(head.time_encoder, "d_T")


def test_mask_aware_padding_ignored():
    head = LinkPredHead(max_walk_len=6)
    w = _walks()
    E_v = F.normalize(torch.randn(2, 5, 16), dim=-1)
    out0 = head(E_v, w)
    # corrupt PADDED positions only; score must not move
    w2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in w.items()}
    pad = ~w2["mask"]
    w2["E_walks"][pad] = torch.randn_like(w2["E_walks"][pad])
    out1 = head(E_v, w2)
    assert torch.allclose(out0, out1, atol=1e-6)


def test_cold_start_empty_walk_no_nan():
    head = LinkPredHead(max_walk_len=6)
    w = _walks()
    # make walk row (b=0, w=1) fully empty
    w["mask"][0, 1] = False
    E_v = F.normalize(torch.randn(2, 5, 16), dim=-1)
    out = head(E_v, w)
    assert torch.isfinite(out).all()
    # fully-empty ENTIRE query (no valid positions anywhere) -> finite, ~0
    w["mask"][:] = False
    out2 = head(E_v, w)
    assert torch.isfinite(out2).all()


def test_recency_weighting_dominates():
    """Raising ω on one hop makes that position's cos dominate the pool."""
    head = LinkPredHead(max_walk_len=6)
    w = _walks(valid=4)
    E_v = F.normalize(torch.randn(1, 3, 16), dim=-1)
    # force ω to put ~all mass on hop 0
    with torch.no_grad():
        head.omega.weight.zero_()
        head.omega.weight[0] = 20.0
        head.logit_scale.fill_(1.0)
    out = head(E_v, w)
    # score should ≈ mean_w cos(E_v, E_w at hop-0 position) (alpha ~ onehot)
    s_hop0 = torch.einsum("bcd,bwd->bcw", E_v, w["E_walks"][:, :, 0]).mean(-1)
    assert torch.allclose(out, s_hop0, atol=1e-3)


def test_kernel_candidate_independent():
    head = LinkPredHead(max_walk_len=6)
    w = _walks()
    # two disjoint candidate sets share the same ω/pool; scores are just the
    # per-candidate dot products through the SAME weights (no cross-talk)
    Ea = F.normalize(torch.randn(2, 4, 16), dim=-1)
    Eb = F.normalize(torch.randn(2, 7, 16), dim=-1)
    oa = head(Ea, w)
    ob = head(Eb, w)
    assert oa.shape == (2, 4) and ob.shape == (2, 7)
    # scoring Ea alone vs as a prefix of [Ea|Eb] must match (set-independent)
    both = head(torch.cat([Ea, Eb], dim=1), w)
    assert torch.allclose(both[:, :4], oa, atol=1e-6)


def test_learnability_planted_positive():
    torch.manual_seed(0)
    d = 16
    head = LinkPredHead(max_walk_len=6)
    w = _walks(B=8, W=3, L=6, d=d, valid=4)
    E_v = F.normalize(torch.randn(8, 101, d), dim=-1)
    # plant: positive = mean walk-node direction (recency-poolable signal)
    E_v[:, 0] = F.normalize(w["E_walks"].mean(dim=(1, 2)), dim=-1)
    opt = torch.optim.Adam(head.parameters(), lr=3e-2)
    last = None
    for _ in range(200):
        loss = F.cross_entropy(head(E_v, w), torch.zeros(8, dtype=torch.long))
        opt.zero_grad(); loss.backward(); opt.step()
        last = loss.item()
    assert last < 0.5 * math.log(101)
