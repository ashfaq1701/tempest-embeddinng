"""Walk-mediated link-prediction head — order-aware recency pool.

Per (u, t, candidate v), for each of u's K walks and each walk position p
the head scores the candidate against the walk node at that position by a
plain cosine (E is unit-norm, so cos(E_v, E_w_p) = E_v · E_w_p), then pools
those per-position similarities with a learned per-hop recency kernel:

    s_p(v) = cos(E_v, E_w_p)
    score(v) = scale · mean_walks  Σ_p  ω_p · s_p(v)

ω_p is a learned weight per hop index, softmax-normalised over the valid
positions of each walk (a recency/order kernel over the walk), and is
candidate-INDEPENDENT — so the candidate enters only through the cosine and
the whole head is a single einsum plus a pooled sum (no per-position MLP, no
recurrence). `scale` is a learnable temperature: the pooled score is a convex
combination of cosines in [-1, 1], which would give a near-flat per-query
softmax (and vanishing gradients) without it.

The walk's seed slot is node u, so cos(E_v, E_u) — the direct u-vs-v
similarity — is one of the pooled terms.

E[v] is EXPECTED to be detached upstream — this head's gradients update only
its own parameters; E is shaped by the alignment loss alone (u enters only
through its walks, whose seed slot is node u).

forward(E_v, walks) -> [B, C] logits.  walks is the per-position feature dict
(E_walks [B,W,L,d], mask [B,W,L], K_idx [B,W,L] hop, t_feat [B,W,L,d_T]).
"""
import math

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    def __init__(self, n_omega: int = 4):
        super().__init__()
        self.n_omega = n_omega
        omegas = 2.0 * math.pi * (2.0 ** torch.arange(n_omega).float())
        self.register_buffer("omegas", omegas, persistent=False)
        self.d_T = 4 + 2 * n_omega

    def forward(self, gap_norm: torch.Tensor) -> torch.Tensor:
        g = gap_norm.clamp(0.0, 1.0)
        raw = torch.stack([g, torch.exp(-g), g * g, torch.log1p(g)], dim=-1)
        ang = g.unsqueeze(-1) * self.omegas
        sin_cos = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return torch.cat([raw, sin_cos], dim=-1)


class LinkPredHead(nn.Module):
    """Order-aware recency pool over candidate-vs-walk cosines.

    See the module docstring for the scoring rule. A learned per-hop weight
    ω (softmax over each walk's valid positions) replaces an order-blind pool;
    the candidate enters only through the cosine, so the head is ~free per
    candidate. `time_encoder` is exposed because the trainer builds the walks'
    t_feat channel through it (the head itself scores from hop + cosine only).

    forward(E_v, walks) -> [B, C]. E_v / E_walks detached upstream.
    """

    def __init__(self, max_walk_len: int):
        super().__init__()
        # The trainer reads link_head.time_encoder to build the walks' t_feat
        # channel; this head scores from hop + cosine and does not consume
        # t_feat itself, but must expose the encoder for the shared path.
        self.time_encoder = TimeEncoder()
        # one scalar log-weight per hop index; ω_p = softmax over valid p
        self.omega = nn.Embedding(max_walk_len, 1)
        nn.init.zeros_(self.omega.weight)          # uniform pool at init
        # Learnable temperature. The pooled score is a convex combination of
        # cosines (range [-1, 1]); at init the uniform pool averages ~WL
        # cosines toward 0, so a larger scale is needed for a healthy per-query
        # softmax (ω concentrates during training and widens the spread).
        # Clamped to (0, 100].
        self.logit_scale = nn.Parameter(torch.tensor(30.0))

    def forward(self, E_v: torch.Tensor, walks: dict) -> torch.Tensor:
        E_walks = walks["E_walks"]                 # [B,W,L,d]
        mask = walks["mask"]                       # [B,W,L]
        K_idx = walks["K_idx"]                     # [B,W,L] hop
        valid = mask.bool()
        s = torch.einsum("bcd,bwld->bcwl", E_v, E_walks)        # [B,C,W,L] cos
        w = self.omega(K_idx).squeeze(-1)                       # [B,W,L] log-weight
        w = w.masked_fill(~valid, float("-inf"))
        alpha = torch.softmax(w, dim=-1)                        # [B,W,L] over positions
        alpha = torch.nan_to_num(alpha, 0.0)                    # empty walk row -> 0
        score_w = (s * alpha.unsqueeze(1)).sum(dim=-1)         # [B,C,W] per-walk
        walk_valid = valid.any(dim=-1).float()                 # [B,W]
        denom = walk_valid.sum(dim=-1).clamp_min(1.0)          # [B]
        score = (score_w * walk_valid.unsqueeze(1)).sum(dim=-1) / denom.unsqueeze(1)
        return self.logit_scale.clamp(1.0, 100.0) * score      # [B,C]
