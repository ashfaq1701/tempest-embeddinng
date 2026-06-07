"""Link-prediction head v2 — walk-mediated per-position similarity.

Fixed architecture (sweep settled 2026-06-07 on tgbl-wiki):

For each (u, t, v_candidate):
    1. Sample K walks for u in ONE direction (undirected → backward,
       directed → forward; decided upstream from is_directed).
    2. For each (walk, position p) build:
         sim_vec  = [Hadamard(E[v], E[w]), |E[v] - E[w]|]      → 2·d_emb
         K_feat   = nn.Embedding(max_walk_len, d_K)[hop]       → d_K
                    forward:  hop = p; backward: hop = lens-1-p.
                    (Seed slot gets hop=0 either way.)
         t_feat   = TimeEncoder(gap_norm)                       → d_T
                    non-seed: gap = log1p(t_query - t_edge_p) /
                                    log1p(T_full)
                    seed:     gap = log1p(t_query - t_min)    /
                                    log1p(T_full)
    3. Per-position MLP on concat[sim_vec, K_feat, t_feat] → d_pos.
    4. Mask padded / sentinel positions to -inf / 0.
    5. Pool over positions: concat(max, mean) → 2·d_pos per walk.
    6. Pool over walks: mean → 2·d_pos per (u, v).
    7. Final MLP on the pooled walk features → scalar logit.

The single-tower decision is settled: Phase-0 + α-leak grid showed
"both" beats single-sided by ≤ 0.008 test (inside the wiki noise
band) at 2× compute, so the second tower is dropped. The choice of
direction is dictated by is_directed (TGB's default workload is
undirected, so backward is the operative mode).

The earlier standalone "direct channel" — a per-pair MLP on
(E[u], E[v]) concatenated to the walk features — was removed: the
walk seed slot IS node u (kept and compared with each candidate v at
hop=0), so the tower already carries the u-vs-v comparison the direct
channel duplicated. The walk-only ablation cost only ~0.03 val /
~0.05 test, attributable to that overlap, so the redundant channel
(and its E_u input) is dropped.

The E[v] candidate input is EXPECTED to be detached upstream — this
head's gradients update only its own parameters; E is shaped by the
alignment loss alone.
"""
import math

import torch
import torch.nn as nn
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

    Fixed architecture:
      - sim primitives: [Hadamard, |E_v - E_w|]   → 2·d_emb per position
      - K (hop) embedding: nn.Embedding(L, d_K)   → d_K   per position
      - time channel:  TimeEncoder(gap_norm)       → d_T   per position
      - per-position MLP, max + mean pool over positions, mean over walks.

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
        self.chunk_C = int(chunk_C)
        self.K_emb = nn.Embedding(max_walk_len, d_K)
        in_dim = 2 * d_emb + d_K + d_T
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
        """Per-(B, chunk_C, W, L) similarity primitives — [Hadamard,
        |E_v - E_w|]. Output last dim = 2·d_emb. Same math as a single
        broadcast over C; chunked for memory."""
        # E_walks [B, W, L, d]   → [B, 1, W, L, d]
        # E_v_chunk [B, chunk, d] → [B, chunk, 1, 1, d]
        Ew = E_walks.unsqueeze(1)
        Ev = E_v_chunk.unsqueeze(2).unsqueeze(3)
        had = Ev * Ew
        absd = (Ev - Ew).abs()
        return torch.cat([had, absd], dim=-1)

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

        sim = self._chunk_sim(E_walks, E_v_chunk)  # [B, chunk, W, L, 2·d_emb]
        Ke = self.K_emb(K_idx)                      # [B, W, L, d_K]
        feat = torch.cat([
            sim,
            Ke.unsqueeze(1).expand(B, chunk, W, L, Ke.shape[-1]),
            t_feat.unsqueeze(1).expand(B, chunk, W, L, t_feat.shape[-1]),
        ], dim=-1)
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
# Top-level head


class LinkPredHeadV2(nn.Module):
    """Walk-mediated similarity head — fixed architecture.

    A single WalkTower (per-position sim + K + time, pooled) feeds a
    final MLP that maps the pooled walk features to a scalar logit.
    The walk seed slot is node u itself (kept and compared with each
    candidate v), so the tower already carries the u-vs-v signal; the
    earlier standalone direct (E[u], E[v]) channel was removed as
    redundant.

    Args:
        d_emb           : embedding width (model config).
        max_walk_len    : L; used to size the K embedding table.
        d_K, d_pos      : tower hyperparameters.
        chunk_C         : candidate-dim memory chunking (see WalkTower).

    Inputs at forward():
        E_v           : [B, C, d_emb]  candidate embeddings (detached).
        walks         : dict. Required. Keys:
                              E_walks [B,W,L,d], mask [B,W,L],
                              K_idx [B,W,L], t_feat [B,W,L,d_T].

    Returns: [B, C] logits.
    """

    def __init__(
        self,
        d_emb: int,
        max_walk_len: int,
        d_K: int = 16,
        d_pos: int = 96,
        chunk_C: int = 0,
    ):
        super().__init__()
        self.time_encoder = TimeEncoder()  # d_T = 12
        self.tower = WalkTower(
            d_emb=d_emb, max_walk_len=max_walk_len,
            d_K=d_K, d_pos=d_pos, d_T=self.time_encoder.d_T,
            chunk_C=chunk_C,
        )
        final_in = self.tower.out_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(final_in, final_in),
            nn.GELU(),
            nn.Linear(final_in, 1),
        )

    def forward(
        self,
        E_v: torch.Tensor,
        walks: dict,
    ) -> torch.Tensor:
        walk_features = self.tower(E_v=E_v, **walks)
        return self.final_mlp(walk_features).squeeze(-1)
