"""Link-prediction head v2 — walk-mediated per-position similarity.

Design (from analysis/REPORT.md §9, refined per design discussion):

For each (u, t, v_candidate):
    1. Sample n_walks walks for u (forward, backward, or both — config).
    2. For each (walk, position p) build:
         sim_vec  = primitives between E[v_cand] and E[walks_u[p]]
                    default: [Hadamard, |E_v - E_w|]            → 2·d_emb
                    cosine_only ablation: scalar cosine          → 1
         K_feat   = nn.Embedding(max_walk_len, d_K)[hop_distance]
                    forward direction:  hop = p
                    backward direction: hop = lens-1-p
                    (Seed slot gets hop=0 either way.)
         t_feat   = TimeEncoder(gap_norm)
                    non-seed: gap = (t_query - t_edge_p) / T_train
                    seed:     gap = (t_query - t_min)    / T_train
    3. Per-position MLP on concat[sim_vec, K_feat, t_feat] → d_pos
    4. Mask padded / sentinel positions to -inf / 0.
    5. Pool over positions: concat(max, mean) → 2·d_pos per walk.
    6. Pool over walks: mean → 2·d_pos per (u, v).
    7. Direct channel (bypass walks): MLP(Hadamard(E_u, E_v), |E_u-E_v|).
    8. Final MLP on concat[walk_features_per_direction, direct] → scalar logit.

The two directions (when direction="both") run through SEPARATE
WalkTower instances; their outputs are concatenated before the final
MLP. This preserves clean ablation semantics.

The (E[u], E[v]) inputs are EXPECTED to be detached upstream — this
head's gradients update only its own parameters; E is shaped by the
alignment loss alone.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------
# Time / position featurisers


class TimeEncoder(nn.Module):
    """Multi-scale sinusoidal + raw encoding of a normalised gap ∈ [0, 1].

    Output dim = 4 (raw: g, exp(-g), g², log1p(g)) + 2·n_omega (sin/cos).
    Default n_omega=4 → d_T = 12.
    """

    def __init__(self, n_omega: int = 4):
        super().__init__()
        self.n_omega = n_omega
        # Geometric base-2 frequencies up to 2^(n_omega-1) cycles per unit gap.
        omegas = 2.0 * math.pi * (2.0 ** torch.arange(n_omega).float())
        self.register_buffer("omegas", omegas, persistent=False)
        self.d_T = 4 + 2 * n_omega

    def forward(self, gap_norm: torch.Tensor) -> torch.Tensor:
        """gap_norm shape [...]. Output [..., d_T].

        Designed for gap_norm ∈ [0, 1] (the head's normaliser keeps it
        there). The clamp is defensive against any caller that passes
        a wider range.
        """
        g = gap_norm.clamp(0.0, 1.0)
        raw = torch.stack(
            [g, torch.exp(-g), g * g, torch.log1p(g)], dim=-1
        )
        ang = g.unsqueeze(-1) * self.omegas
        sin_cos = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return torch.cat([raw, sin_cos], dim=-1)


# ---------------------------------------------------------------------
# Walk tower


class WalkTower(nn.Module):
    """Process u-side walks into per-(u, v) pooled vectors.

    Inputs shapes (B = batch, C = num candidates per row,
                   W = num walks per seed, L = max walk length):
        E_walks   : [B, W, L, d_emb]  walk node embeddings (detached)
        mask      : [B, W, L]         True at valid positions (incl seed)
        K_idx     : [B, W, L]         long hop distance from seed
        t_feat    : [B, W, L, d_T]    pre-encoded gap features
        E_v       : [B, C, d_emb]     candidate embeddings (detached)

    Output: [B, C, 2·d_pos]  concat(max-pool, mean-pool) over positions,
                              averaged over walks.
    """

    def __init__(
        self,
        d_emb: int,
        max_walk_len: int,
        sim_primitives: str,
        use_K: bool,
        use_t: bool,
        d_K: int = 16,
        d_pos: int = 96,
        d_T: int = 12,
        chunk_C: int = 0,
    ):
        # chunk_C: candidates processed per inner loop. 0 (default) =
        # OFF, whole candidate dim runs in a single tensor (caller
        # must size the batch to fit GPU memory). When > 0 (and < C),
        # the forward chunks over C; intermediates per chunk are
        # wrapped in torch.utils.checkpoint so they're freed between
        # chunks and re-materialised at backward. Math (loss,
        # gradient, optimizer step) is identical to the un-chunked
        # path; only wall-clock changes (~2-3× slower per step).
        super().__init__()
        if sim_primitives == "hadamard_absdiff":
            self.sim_dim = 2 * d_emb
        elif sim_primitives == "cosine_only":
            self.sim_dim = 1
        else:
            raise ValueError(f"unknown sim_primitives: {sim_primitives}")
        self.sim_primitives = sim_primitives
        self.use_K = use_K
        self.use_t = use_t
        self.chunk_C = int(chunk_C)

        in_dim = self.sim_dim
        if use_K:
            self.K_emb = nn.Embedding(max_walk_len, d_K)
            in_dim += d_K
        if use_t:
            in_dim += d_T

        self.per_pos_mlp = nn.Sequential(
            nn.Linear(in_dim, d_pos),
            nn.GELU(),
            nn.Linear(d_pos, d_pos),
        )
        self.d_pos = d_pos
        self.out_dim = 2 * d_pos  # max + mean over positions

    def _chunk_sim(
        self, E_walks: torch.Tensor, E_v_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Per-(B, chunk_C, W, L) similarity primitives. Output last
        dim = sim_dim. Same math as a single broadcast over C; chunked
        for memory."""
        # E_walks [B, W, L, d]   → [B, 1, W, L, d]
        # E_v_chunk [B, chunk, d] → [B, chunk, 1, 1, d]
        Ew = E_walks.unsqueeze(1)
        Ev = E_v_chunk.unsqueeze(2).unsqueeze(3)
        if self.sim_primitives == "hadamard_absdiff":
            had = Ev * Ew
            absd = (Ev - Ew).abs()
            return torch.cat([had, absd], dim=-1)
        # cosine_only
        eps = 1e-6
        num = (Ev * Ew).sum(-1, keepdim=True)
        den = (
            Ev.norm(dim=-1, keepdim=True).clamp_min(eps)
            * Ew.norm(dim=-1, keepdim=True).clamp_min(eps)
        )
        return num / den

    def _process_chunk(
        self,
        E_walks: torch.Tensor,
        mask: torch.Tensor,
        K_idx: torch.Tensor,
        t_feat: torch.Tensor,
        E_v_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-(u, v) walk-pooled vector for a chunk of
        candidates. Returns [B, chunk_C, 2·d_pos]."""
        B, W, L, _ = E_walks.shape
        chunk = E_v_chunk.shape[1]

        sim = self._chunk_sim(E_walks, E_v_chunk)  # [B, chunk, W, L, sim_dim]
        feat_parts = [sim]
        if self.use_K:
            Ke = self.K_emb(K_idx)  # [B, W, L, d_K]
            feat_parts.append(
                Ke.unsqueeze(1).expand(B, chunk, W, L, Ke.shape[-1])
            )
        if self.use_t:
            feat_parts.append(
                t_feat.unsqueeze(1).expand(B, chunk, W, L, t_feat.shape[-1])
            )
        feat = torch.cat(feat_parts, dim=-1)
        h = self.per_pos_mlp(feat)  # [B, chunk, W, L, d_pos]
        m = mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, W, L, 1]

        h_neg = h.masked_fill(m == 0, float("-inf"))
        h_max = h_neg.max(dim=3).values  # [B, chunk, W, d_pos]
        h_max = torch.where(
            torch.isinf(h_max), torch.zeros_like(h_max), h_max,
        )
        h_sum = (h * m).sum(dim=3)
        m_sum = m.sum(dim=3).clamp_min(1.0)
        h_mean = h_sum / m_sum

        h_walk = torch.cat([h_max, h_mean], dim=-1)  # [B, chunk, W, 2·d_pos]
        return h_walk.mean(dim=2)  # [B, chunk, 2·d_pos]

    def forward(
        self,
        E_walks: torch.Tensor,
        mask: torch.Tensor,
        K_idx: torch.Tensor,
        t_feat: torch.Tensor,
        E_v: torch.Tensor,
    ) -> torch.Tensor:
        C = E_v.shape[1]
        if self.chunk_C <= 0 or self.chunk_C >= C:
            return self._process_chunk(E_walks, mask, K_idx, t_feat, E_v)
        # Chunked path. Each chunk's forward is wrapped in
        # torch.utils.checkpoint so its intermediates (sim cube, MLP
        # activations) are dropped after the chunk runs and
        # re-materialised on demand in backward. Without this, all
        # chunks' intermediates stay alive until the global backward
        # → defeats the memory benefit of chunking.
        out_parts = []
        for c0 in range(0, C, self.chunk_C):
            c1 = min(c0 + self.chunk_C, C)
            E_v_chunk = E_v[:, c0:c1]
            if self.training and torch.is_grad_enabled():
                out_parts.append(
                    checkpoint(
                        self._process_chunk,
                        E_walks, mask, K_idx, t_feat, E_v_chunk,
                        use_reentrant=False,
                    )
                )
            else:
                out_parts.append(
                    self._process_chunk(
                        E_walks, mask, K_idx, t_feat, E_v_chunk,
                    )
                )
        return torch.cat(out_parts, dim=1)  # [B, C, 2·d_pos]


# ---------------------------------------------------------------------
# Direct (E[u], E[v]) bypass


class DirectChannel(nn.Module):
    """Per-pair direct features from raw E. Acts as a residual when walks
    fail (empty W_u, inductive nodes) so the head doesn't lose all
    signal — it still has cosine-baseline-equivalent capacity."""

    def __init__(self, d_emb: int, d_direct: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_emb, d_direct),
            nn.GELU(),
            nn.Linear(d_direct, d_direct),
        )
        self.out_dim = d_direct

    def forward(
        self, E_u: torch.Tensor, E_v: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([E_u * E_v, (E_u - E_v).abs()], dim=-1)
        return self.mlp(x)


# ---------------------------------------------------------------------
# Top-level head


class LinkPredHeadV2(nn.Module):
    """Walk-mediated similarity head.

    Args:
        d_emb           : embedding width (model config).
        max_walk_len    : L; used to size the K embedding table.
        direction       : 'forward' | 'backward' | 'both'.
        sim_primitives  : 'hadamard_absdiff' (default) | 'cosine_only'.
        use_time_channel/use_K_channel : channel ablations.
        use_direct      : include the direct bypass (default True).
        direct_only     : drop walk towers entirely; head reduces to a
                          per-dim MLP on (E[u], E[v]).

    Inputs at forward():
        E_u           : [B, C, d_emb]  source embedding broadcast across
                                       candidates (detached upstream).
        E_v           : [B, C, d_emb]  candidate embeddings (detached).
        walks_fwd     : dict or None.  Required if direction in
                                       {'forward', 'both'}. Keys:
                                       E_walks [B,W,L,d], mask [B,W,L],
                                       K_idx [B,W,L], t_feat [B,W,L,d_T].
        walks_bwd     : dict or None.  Same shape for the backward side.

    Returns: [B, C] logits.
    """

    def __init__(
        self,
        d_emb: int,
        max_walk_len: int,
        direction: str = "both",
        sim_primitives: str = "hadamard_absdiff",
        use_time_channel: bool = True,
        use_K_channel: bool = True,
        use_direct: bool = True,
        direct_only: bool = False,
        d_K: int = 16,
        d_pos: int = 96,
        d_direct: int = 64,
        chunk_C: int = 0,
    ):
        super().__init__()
        if direction not in ("forward", "backward", "both"):
            raise ValueError(direction)
        self.direction = direction
        self.direct_only = direct_only
        self.use_direct = bool(use_direct or direct_only)

        self.time_encoder = TimeEncoder()  # d_T = 12

        final_in = 0
        if not direct_only:
            shared_kwargs = dict(
                d_emb=d_emb, max_walk_len=max_walk_len,
                sim_primitives=sim_primitives,
                use_K=use_K_channel, use_t=use_time_channel,
                d_K=d_K, d_pos=d_pos, d_T=self.time_encoder.d_T,
                chunk_C=chunk_C,
            )
            if direction in ("forward", "both"):
                self.tower_fwd = WalkTower(**shared_kwargs)
                final_in += self.tower_fwd.out_dim
            if direction in ("backward", "both"):
                self.tower_bwd = WalkTower(**shared_kwargs)
                final_in += self.tower_bwd.out_dim

        if self.use_direct:
            self.direct = DirectChannel(d_emb=d_emb, d_direct=d_direct)
            final_in += self.direct.out_dim

        if final_in == 0:
            raise ValueError(
                "head has no input — at least one of {walk_tower, direct} "
                "must be active."
            )

        self.final_mlp = nn.Sequential(
            nn.Linear(final_in, final_in),
            nn.GELU(),
            nn.Linear(final_in, 1),
        )

    def forward(
        self,
        E_u: torch.Tensor,
        E_v: torch.Tensor,
        walks_fwd: Optional[dict] = None,
        walks_bwd: Optional[dict] = None,
    ) -> torch.Tensor:
        parts = []
        if not self.direct_only:
            if self.direction in ("forward", "both"):
                assert walks_fwd is not None, "forward walks required"
                parts.append(self.tower_fwd(E_v=E_v, **walks_fwd))
            if self.direction in ("backward", "both"):
                assert walks_bwd is not None, "backward walks required"
                parts.append(self.tower_bwd(E_v=E_v, **walks_bwd))
        if self.use_direct:
            parts.append(self.direct(E_u, E_v))
        x = torch.cat(parts, dim=-1)
        return self.final_mlp(x).squeeze(-1)
