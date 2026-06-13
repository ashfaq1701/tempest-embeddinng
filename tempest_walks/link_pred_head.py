"""Candidate-conditioned sphere-attention link head.

Replaces the GRU mean-pool readout. There is NO precomputed h[u] anymore: the
source's neighbourhood is not collapsed up-front. Instead, *each candidate v
attends, separately, over the source's walk-neighbour tokens* {w_i}, and pulls a
candidate-specific residual that a small head turns into a logit.

Pipeline for one (source u, candidate v):

  1. tokens        u's CONTEXT walk-neighbours w_i (seed + padding excluded), each
                   carrying E[w_i] and its within-walk recency Time2Vec(Δt_i).
                   (No K / walk axis yet — tokens arrive flattened per source as
                   [n, d]. The K split can be re-added later without touching this
                   head: it only ever sees a flat token set + mask.)
  2. attention     q = W·E[v]  (the candidate asks),
                   k_i = W·E[w_i] + time(Δt_i)  (each neighbour answers).
                   a_i = softmax_i(q·k_i / sqrt(d_head)).  ONE tied projection W on
                   the embedding side (fixed temperature) decides WHICH neighbours
                   are relevant to this candidate.
  3. value         the GEOMETRY: Log_v(w_i), the sphere's own "w_i seen from v" — a
                   tangent vector at v whose length is the geodesic distance to w_i.
                   (Plain (E[v]-E[w_i])^2 is the flat chord through the ball and
                   ignores the curvature E is trained on; the log-map is the
                   manifold-correct residual.)
  4. pool          r(u,v) = Σ_i a_i · Log_v(w_i)  — a d-dim tangent residual.
                   Because the base point is v for every term, all the log-maps live
                   in the SAME tangent space T_v, so this sum is exact (no
                   parallel-transport approximation). r ≈ 0 when v sits on top of the
                   neighbours it attends to (a clean fit); large + structured when it
                   doesn't.
  5. readout       small MLP: r -> scalar logit. Plus the proven recency + pair
                   terms, unchanged.

Base channel (E[u]): u's own identity is deliberately NOT a token in the attention
pool — that channel answers "is v like u's NEIGHBOURS". u enters once, as a plain
chord(E[u], E[v]) term ("is v like u DIRECTLY"), the repeat-slice workhorse. Keeping
the two questions on separate, non-competing terms is the whole point of the split.

FAST: we never build the [B, C, n, d] log-map tensor. The sphere log-map has a
closed form that lets the d-axis appear only in a final [B,C,n]·[B,n,d] contraction
(see `_pooled_logmap`). The only big intermediates are scalar-per-token [B, C, n].

E stays the single sphere-trained parameter (link-trained, no detach). W / time /
readout are Euclidean.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Time2Vec(nn.Module):
    """Time2Vec (Kazemi et al. 2019): scalar τ -> [linear, sin(ω₁τ+φ₁), …].
    First channel linear, rest periodic. τ is fed pre-normalised (log1p of a Δ)."""

    def __init__(self, dim: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(dim - 1))
        self.b = nn.Parameter(torch.rand(dim - 1) * 2 * math.pi)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.unsqueeze(-1)                          # [..., 1]
        lin = self.w0 * tau + self.b0                    # [..., 1]
        per = torch.sin(tau * self.w + self.b)           # [..., dim-1]
        return torch.cat([lin, per], dim=-1)             # [..., dim]


class SourceWalkAttnHead(nn.Module):
    def __init__(self, d_emb: int, d_time: int = 16, d_head: int = 64,
                 readout_hidden: int = None, dropout: float = 0.15,
                 use_pair_features: bool = False):
        super().__init__()
        self.d_emb = d_emb
        self.d_head = d_head
        self.eps = 1e-6

        # --- attention (decides WHICH neighbours matter to a candidate) ---------
        # ONE tied projection on the embedding side: the candidate and every
        # neighbour token go through the SAME W, so the query/key metric is shared
        # (no untied-projection drift). Time augments the KEY only — a neighbour's
        # relevance may depend on how recently u touched it.
        self.W = nn.Linear(d_emb, d_head, bias=False)
        self.t2v_walk = Time2Vec(d_time)
        self.time_key = nn.Linear(d_time, d_head)

        # --- base channel: direct u-vs-v chord on the sphere --------------------
        # E[u] is deliberately NOT a token in the attention pool (that channel
        # answers "is v like u's NEIGHBOURS"). u's own identity enters here instead:
        # a plain chord(E[u], E[v]) — "is v like u directly". Strong on the repeat /
        # easy-transductive mass. Kept separate so the two questions don't compete.
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

        # --- readout (turns the geometric residual into a logit) ----------------
        hid = readout_hidden or d_emb
        self.readout = nn.Sequential(
            nn.Linear(d_emb, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

        # --- proven extra terms, unchanged from the base head -------------------
        self.t2v_rec = Time2Vec(d_time)
        self.rec_head = nn.Linear(d_time, 1)
        self.use_pair_features = use_pair_features
        if use_pair_features:
            self.t2v_pair = Time2Vec(d_time)
            self.pair_head = nn.Linear(d_time + 2, 1)

    # ----------------------------------------------------------------------
    # Attention weights:  q = W·v ,  k_i = W·w_i + time(Δt_i)
    # ----------------------------------------------------------------------
    def _attention(self, ev: torch.Tensor, ew: torch.Tensor,
                   tok_dt: torch.Tensor, tok_mask: torch.Tensor) -> torch.Tensor:
        """ev [B,C,d] (unit), ew [B,n,d] (unit), tok_dt [B,n], tok_mask [B,n] bool
        -> a [B,C,n] attention weights (softmax over neighbours, padding-safe)."""
        q = self.W(ev)                                              # [B, C, dh]
        k = self.W(ew) + self.time_key(self.t2v_walk(tok_dt))       # [B, n, dh]
        scores = torch.einsum("bcd,bnd->bcn", q, k) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~tok_mask.unsqueeze(1), float("-inf"))
        a = torch.softmax(scores, dim=-1)                           # [B, C, n]
        # Sources with zero valid tokens (cold start) softmax to NaN -> set to 0;
        # their residual becomes 0 and the logit falls back to base + recency + pair.
        return torch.nan_to_num(a, nan=0.0)

    # ----------------------------------------------------------------------
    # Pooled sphere log-map (the geometry), computed WITHOUT [B,C,n,d].
    #
    #   Log_v(w_i) = g_i · (w_i − cosθ_i · v),   g_i = θ_i / sinθ_i,  cosθ_i=⟨v,w_i⟩
    #   r = Σ_i a_i Log_v(w_i)
    #     = Σ_i (a_i g_i) w_i  −  (Σ_i a_i g_i cosθ_i) v
    #     = einsum(β, w)       −  α · v
    # so the d-axis only appears in the final [B,C,n]·[B,n,d] contraction.
    # ----------------------------------------------------------------------
    def _pooled_logmap(self, ev: torch.Tensor, ew: torch.Tensor,
                       a: torch.Tensor) -> torch.Tensor:
        """ev [B,C,d] (unit), ew [B,n,d] (unit), a [B,C,n] -> r [B,C,d] in T_v."""
        cos = torch.einsum("bcd,bnd->bcn", ev, ew).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(cos)                                   # [B,C,n]
        sin = torch.sqrt(1.0 - cos * cos).clamp_min(self.eps)       # sinθ ≥ 0
        g = theta / sin                                             # θ/sinθ → 1 as θ→0
        beta = a * g                                                # [B,C,n]
        alpha = (beta * cos).sum(dim=-1)                            # [B,C]
        r = torch.einsum("bcn,bnd->bcd", beta, ew) - alpha.unsqueeze(-1) * ev
        return r                                                    # ⊥ v, in T_v

    # ----------------------------------------------------------------------
    def forward(self, tok_emb: torch.Tensor, tok_dt: torch.Tensor,
                tok_mask: torch.Tensor, E_u: torch.Tensor, E_v: torch.Tensor,
                rec_v_log: torch.Tensor,
                pair_rec_log: torch.Tensor = None,
                pair_ever: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        """tok_emb [B,n,d]  source walk-neighbour embeddings (context only; the
                            trainer gathers these per batch row via u_pos and zeroes
                            the seed + padding, which `tok_mask` marks invalid).
           tok_dt  [B,n]    log1p within-walk Δt per token (0 where masked).
           tok_mask[B,n]    bool, True at real neighbour positions.
           E_u     [B,d]    source embeddings (the base chord channel; NOT a token).
           E_v     [B,C,d]  candidate embeddings.
           rec_v_log [B,C]  log1p(t_query − t_last[v]) candidate recency.
           -> logits [B, C].
        """
        ev = F.normalize(E_v, dim=-1)                               # [B, C, d]
        ew = F.normalize(tok_emb, dim=-1)                           # [B, n, d]
        eu = F.normalize(E_u, dim=-1).unsqueeze(1)                  # [B, 1, d]

        # --- attention channel: is v like u's NEIGHBOURS -----------------------
        a = self._attention(ev, ew, tok_dt, tok_mask)              # [B, C, n]
        r = self._pooled_logmap(ev, ew, a)                         # [B, C, d]
        logit = self.readout(r).squeeze(-1)                        # [B, C]

        # --- base channel: is v like u DIRECTLY (chord on the sphere) ----------
        # ‖a-b‖ = √(2-2⟨a,b⟩); closer => higher logit. u enters ONLY here.
        c = (eu * ev).sum(dim=-1).clamp(-1 + self.eps, 1 - self.eps)   # [B, C]
        chord = torch.sqrt(2.0 - 2.0 * c)                          # [B, C]
        logit = logit - self.logit_scale.clamp_min(1e-3) * chord

        # Proven query-time recency term (the walks are blind to query time).
        logit = logit + self.rec_head(self.t2v_rec(rec_v_log)).squeeze(-1)

        if self.use_pair_features:
            feat = torch.cat(
                [self.t2v_pair(pair_rec_log),
                 pair_ever.unsqueeze(-1), pair_count_log.unsqueeze(-1)],
                dim=-1)                                             # [B, C, d_time+2]
            logit = logit + self.pair_head(feat).squeeze(-1)

        return logit
