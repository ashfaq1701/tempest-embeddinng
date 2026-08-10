"""Two-sided centroid-vs-token head on the Poincaré ball. E is the only trained tensor. Centroid on the
probe side, raw tokens on the target side, both directions:
    s(u,v) = -[ sum_q w_q^v * d(P_u, x_q^v) + sum_p w_p^u * d(P_v, x_p^u) ]
P_x = weighted gyro-midpoint of x's full bag (seeds included); x_p/x_q = raw token embeddings; w = softmax
of the -(log1p(age)+log1p(hop-1)) prior. No identity and no centroid-centroid term."""
import os
from typing import Tuple

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_NORM_EPS = 1e-5      # ||x||^2 clamp: stay strictly inside the ball
_ACOSH_EPS = 1e-7     # arcosh arg clamp: finite gradient at coincidence


class PoincareManifold:
    """Poincaré-ball geometry (c=1). `manifold` (geoopt ball) is kept for E's init + RiemannianAdam."""

    def __init__(self, c: float = 1.0):
        self.manifold = geoopt.PoincareBall(c=c)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Elementwise geodesic distance (geoopt), broadcasting over leading dims. LOWER = closer."""
        return self.manifold.dist(x, y)

    @staticmethod
    def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """arcosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2))) for x [...,n,d], y [...,m,d] -> [...,n,m].
        ||x-y||^2 expanded as ||x||^2+||y||^2-2<x,y> so the cross term is one matmul (no [...,n,m,d] diff)."""
        x2 = (x * x).sum(dim=-1).clamp(max=1.0 - _NORM_EPS)                     # [..., n]
        y2 = (y * y).sum(dim=-1).clamp(max=1.0 - _NORM_EPS)                     # [..., m]
        xy = torch.matmul(x, y.transpose(-1, -2))                               # [..., n, m]
        sq = (x2.unsqueeze(-1) + y2.unsqueeze(-2) - 2.0 * xy).clamp_min(0.0)     # [..., n, m]
        denom = (1.0 - x2).unsqueeze(-1) * (1.0 - y2).unsqueeze(-2)             # [..., n, m]
        arg = (1.0 + 2.0 * sq / denom).clamp_min(1.0 + _ACOSH_EPS)
        return torch.acosh(arg)


class LinkPredHead(nn.Module):
    """Two-sided centroid-vs-token head. Owns E (ManifoldParameter, trained by the link CE); no other
    parameter."""

    def __init__(self, num_nodes: int, d_emb: int):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()

        # Spread init: geoopt random (std=1), not the near-origin wrapped normal. ManifoldParameter so
        # RiemannianAdam keeps E in the ball.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        # --- TEMPORARY DIAGNOSTIC (diagnostic/weighted-midpoint-diagnosis) ---
        # Track, on TRAINING batches, how far the weighted gyro-midpoint P collapses onto E[seed].
        self._diag_dump_at = int(os.environ.get("MID_DIAG_DUMP_AT", "300"))    # which train batch to dump
        self._diag_path = os.environ.get("MID_DIAG_PATH", "midpoint_collapse_dump.pt")
        self._diag_count = 0
        self._diag_dumped = False

    @staticmethod
    def bag_weight_logits(tokens: WalkTokens) -> torch.Tensor:
        """Recency/hop prior LOGITS [Q, T] = -(log1p(age) + log1p(hop-1)); 0 (max) for the seed (age 0,
        hop 1). DIAGNOSTIC BRANCH: PLAIN log1p(age) (no iterated log). With review ages ~1e7 the seed logit
        (0) beats context logits (~-16) by ~e^16, so the softmax puts ~1.0 on the seed and the gyro-midpoint
        collapses onto E[seed]: P_u == E[u], P_v == E[v]. Instrumented below to prove it on real batches."""
        age = tokens.ages.clamp_min(0).to(torch.float32)                        # [Q, T]  seed=0, ctx>=1
        hop = tokens.positions.clamp_min(1).to(torch.float32)                   # [Q, T]  seed=1, ctx>=2
        return -(torch.log1p(age) + torch.log1p(hop - 1.0))                    # [Q, T]  <= 0  PLAIN log

    @staticmethod
    def bag_weights(tokens: WalkTokens, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """(nodes [Q,T], w [Q,T]): softmax the recency/hop prior over ALL real slots (seed included), 0 on
        padding, sums to 1 per row. Cold-bag guard handles a fully-empty walk (all padding) -> falls back to
        the seed; without it that row's all -inf softmax would be NaN."""
        nodes = tokens.nodes.clamp_min(0).clone()                               # [Q, T] padding(-1) -> 0
        valid = tokens.mask.clone()                                             # [Q, T] real slots (seed incl.)

        cold = ~valid.any(dim=-1)                                               # [Q]  fully-empty walk guard
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        logits = LinkPredHead.bag_weight_logits(tokens).to(dtype)               # [Q, T] <= 0
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)    # [Q, T] sums to 1
        return nodes, w

    def bag_centroid(self, nodes: torch.Tensor, w: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """P_x = weighted gyro-midpoint of the bag's token embeddings (weights w sum to 1)."""
        x = F.embedding(nodes, emb)                                            # [Q, T, d]
        return self.geom.manifold.weighted_midpoint(
            x, weights=w, reducedim=[-2], dim=-1, keepdim=False)               # [Q, d]

    @torch.no_grad()
    def _diag_midpoint_collapse(self, src_tokens: WalkTokens, cand_tokens: WalkTokens,
                                emb: torch.Tensor, w_u: torch.Tensor, w_v: torch.Tensor,
                                p_u: torch.Tensor, p_v: torch.Tensor) -> None:
        """TEMPORARY: measure how far the gyro-midpoint P collapses onto E[seed] on a training batch, and
        dump one batch of raw evidence at MID_DIAG_DUMP_AT. seed_w = softmax weight summed over the age-0
        seed slots; d(P, E[seed]) is the geodesic gap. With plain log1p(age) + review ages ~1e7, seed_w -> 1
        and d(P, E[seed]) -> 0 (P == E[seed] to float precision)."""
        self._diag_count += 1
        e_u = F.embedding(src_tokens.seeds, emb)                               # [B, d]     E[u]
        e_v = F.embedding(cand_tokens.seeds, emb)                              # [B*C, d]   E[v]
        d_pu = self.geom.dist(p_u, e_u)                                        # [B]     d(P_u, E[u])
        d_pv = self.geom.dist(p_v, e_v)                                        # [B*C]   d(P_v, E[v])
        sw_u = (w_u * src_tokens.seed_mask.to(w_u.dtype)).sum(-1)              # [B]     seed weight
        sw_v = (w_v * cand_tokens.seed_mask.to(w_v.dtype)).sum(-1)             # [B*C]

        if self._diag_count == 1 or self._diag_count % 50 == 0 or self._diag_count >= self._diag_dump_at:
            print(f"[MIDDIAG] train batch {self._diag_count}: "
                  f"d(P_u,E_u) max={float(d_pu.max()):.3e} mean={float(d_pu.mean()):.3e} "
                  f"seed_w_u min={float(sw_u.min()):.7f} | "
                  f"d(P_v,E_v) max={float(d_pv.max()):.3e} mean={float(d_pv.mean()):.3e} "
                  f"seed_w_v min={float(sw_v.min()):.7f}", flush=True)

        if self._diag_count >= self._diag_dump_at:
            k = min(8, e_u.shape[0])                                           # limit the raw dump
            torch.save({
                "note": "ctc centroid head, PLAIN log1p(age) weights, tgbl-review ages ~1e7 -> P collapses onto E[seed]",
                "train_batch_index": self._diag_count,
                "src_seeds": src_tokens.seeds[:k].cpu(),
                "d_Pu_Eu": d_pu[:k].cpu(),                                     # geodesic gap per query
                "seed_w_u": sw_u[:k].cpu(),                                    # ~1.0 => collapse
                "P_u_first4dims": p_u[:k, :4].cpu(),                           # side-by-side proof
                "E_u_first4dims": e_u[:k, :4].cpu(),
                "d_Pv_Ev": d_pv[:k].cpu(),
                "seed_w_v": sw_v[:k].cpu(),
                "P_v_first4dims": p_v[:k, :4].cpu(),
                "E_v_first4dims": e_v[:k, :4].cpu(),
                "batch_summary": {
                    "d_Pu_Eu_max": float(d_pu.max()), "d_Pu_Eu_mean": float(d_pu.mean()),
                    "d_Pv_Ev_max": float(d_pv.max()), "d_Pv_Ev_mean": float(d_pv.mean()),
                    "seed_w_u_min": float(sw_u.min()), "seed_w_v_min": float(sw_v.min()),
                    "n_src": int(d_pu.numel()), "n_cand": int(d_pv.numel()),
                },
            }, self._diag_path)
            print(f"[MIDDIAG] dumped 1-batch evidence to {self._diag_path} at train batch {self._diag_count}", flush=True)
            self._diag_dumped = True

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries (seeds u); cand = B*C candidate queries (seeds v), query-major. -> [B, C]."""
        emb = self.E.weight

        nodes_u, w_u = self.bag_weights(src_tokens, emb.dtype)                 # [B, T]
        nodes_v, w_v = self.bag_weights(cand_tokens, emb.dtype)               # [B*C, T]

        x_u = F.embedding(nodes_u, emb)                                        # [B, T, d]
        x_v = F.embedding(nodes_v, emb)                                        # [B*C, T, d]
        p_u = self.bag_centroid(nodes_u, w_u, emb)                            # [B, d]
        p_v = self.bag_centroid(nodes_v, w_v, emb)                            # [B*C, d]

        # TEMPORARY DIAGNOSTIC: prove the midpoint collapse P == E[seed] on real TRAINING batches.
        if self.training and not self._diag_dumped:
            self._diag_midpoint_collapse(src_tokens, cand_tokens, emb, w_u, w_v, p_u, p_v)

        b, d = p_u.shape
        c = p_v.shape[0] // b

        p_v = p_v.view(b, c, d)                                                # [B, C, d]
        x_v = x_v.view(b, c, x_u.shape[1], d)                                  # [B, C, T, d]
        w_v = w_v.view(b, c, x_u.shape[1])                                     # [B, C, T]

        # P_u vs v's tokens, and P_v vs u's tokens.
        d_pu_xv = self.geom.pairwise_dist(p_u[:, None, None, :], x_v).squeeze(-2)  # [B, C, T]
        term_v = (w_v * d_pu_xv).sum(-1)                                       # [B, C]
        d_pv_xu = self.geom.pairwise_dist(p_v, x_u)                            # [B, C, T]
        term_u = (w_u.unsqueeze(1) * d_pv_xu).sum(-1)                          # [B, C]

        raw = term_v + term_u                                                  # [B, C]
        return -raw                                                           # [B, C] higher = closer
