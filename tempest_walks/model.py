"""Minimal production model for tempest-walks-v3.

Three modules:

  EmbeddingStore  — two identity tables E_target / E_context plus
                    optional node-feature residuals and an edge-feature
                    projection consumed by the alignment loss.

  TimeEncoder     — Component 0's functional Φ(Δt) (Xu et al. 2020).
                    Maps Δt → R^(2k) via learnable geometric ω_i.

  LinkPredictor   — 8-block cross-table + Component 0 + 3-layer GELU MLP
                    head. One scalar logit per (u, v, t) pair.

The architecture is FIXED — no head-mode variants, no link-MLP depth
knob, no dropout. Stage 2 (CLAUDE.md) confirmed all of those hurt.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class EmbeddingStore(nn.Module):
    """Dual identity tables (E_target, E_context) + optional feature residuals.

    target(u)  = E_target[u]                         (no node features)
                = target_final([E_target[u] || proj_t(nf[u])])   (with nf)
    context(u) = E_context[u]                        (no node features)
                = context_final([E_context[u] || proj_c(nf[u])]) (with nf)
    context_walk(u, ef) = context_walk_final([context(u) || edge_feat_proj(ef)])

    Alignment loss pulls E_target[seed] toward context_walk reads;
    uniformity spreads E_target over the unit hypersphere; normbrake
    clamps column magnitudes on both tables.
    """

    def __init__(
        self,
        n_nodes: int,
        d_emb: int,
        node_feat: Optional[np.ndarray] = None,
        edge_feat_dim: int = 0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_emb = d_emb

        self.E_target = nn.Embedding(n_nodes, d_emb)
        self.E_context = nn.Embedding(n_nodes, d_emb)
        nn.init.xavier_uniform_(self.E_target.weight)
        nn.init.xavier_uniform_(self.E_context.weight)

        # Node-feature residual (non-persistent buffer; can be hot-swapped).
        self.has_node_feat = node_feat is not None
        if self.has_node_feat:
            self.register_buffer(
                "node_feat",
                torch.from_numpy(np.asarray(node_feat, dtype=np.float32)),
                persistent=False,
            )
            d_nf = int(node_feat.shape[1])
            self.node_feat_proj_target = nn.Linear(d_nf, d_emb)
            self.node_feat_proj_context = nn.Linear(d_nf, d_emb)
        else:
            self.node_feat = None
            self.node_feat_proj_target = None
            self.node_feat_proj_context = None

        # Edge-feature projection (used by context_walk for the alignment loss).
        self.has_edge_feat = edge_feat_dim > 0
        if self.has_edge_feat:
            self.edge_feat_proj = nn.Linear(edge_feat_dim, d_emb)
        else:
            self.edge_feat_proj = None

        # Final fusion projections (only when node features are present).
        nf_extra = d_emb if self.has_node_feat else 0
        self.target_final = (
            nn.Linear(d_emb + nf_extra, d_emb) if nf_extra > 0 else None
        )
        self.context_final = (
            nn.Linear(d_emb + nf_extra, d_emb) if nf_extra > 0 else None
        )

        # context_walk site (only when edge features are present).
        ef_extra = d_emb if self.has_edge_feat else 0
        self.context_walk_final = (
            nn.Linear(d_emb + ef_extra, d_emb) if ef_extra > 0 else None
        )

    def target(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.E_target(ids)
        if not self.has_node_feat:
            return e
        nf_proj = self.node_feat_proj_target(self.node_feat[ids])
        return self.target_final(torch.cat([e, nf_proj], dim=-1))

    def context(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.E_context(ids)
        if not self.has_node_feat:
            return e
        nf_proj = self.node_feat_proj_context(self.node_feat[ids])
        return self.context_final(torch.cat([e, nf_proj], dim=-1))

    def context_walk(
        self,
        walk_nodes: torch.Tensor,                  # [N*K, L] long
        walk_edge_feats: Optional[torch.Tensor],   # [N*K, L-1, d_edge] or None
    ) -> torch.Tensor:
        """Per-position walk representation for the alignment loss."""
        c = self.context(walk_nodes)               # [N*K, L, d_emb]
        if walk_edge_feats is None or not self.has_edge_feat:
            return c
        ef_proj = self.edge_feat_proj(walk_edge_feats.float())   # [N*K, L-1, d_emb]
        # Right-pad to align edge_feats[p] with timestamps[p] at the same hop.
        ef_padded = torch.nn.functional.pad(ef_proj, (0, 0, 0, 1))
        return self.context_walk_final(torch.cat([c, ef_padded], dim=-1))


class TimeEncoder(nn.Module):
    """Φ(Δt) → R^(2k). Xu et al. 2020 functional time encoder.

    Φ(Δt) = [cos(ω_1·Δt), sin(ω_1·Δt), ..., cos(ω_k·Δt), sin(ω_k·Δt)]

    ω_i geometrically initialized covering periods from ~time_scale down
    to ~time_scale/1000. Frequencies are trainable.
    """

    def __init__(self, k: int, time_scale: float):
        super().__init__()
        self.k = int(k)
        i = torch.arange(k, dtype=torch.float32)
        init_omegas = (1.0 / time_scale) * (1000.0 ** (-i / max(k - 1, 1)))
        self.omegas = nn.Parameter(init_omegas)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        phases = dt.unsqueeze(-1) * self.omegas
        cos = torch.cos(phases)
        sin = torch.sin(phases)
        stacked = torch.stack([cos, sin], dim=-1)
        return stacked.flatten(start_dim=-2)


class LinkPredictor(nn.Module):
    """8-block cross-table head + Component 0 + 3-layer GELU MLP.

    Input concat:
      [ e_t_u, e_c_v, e_t_u ⊙ e_c_v, |e_t_u − e_c_v|,        # u→v direction
        e_t_v, e_c_u, e_t_v ⊙ e_c_u, |e_t_v − e_c_u|,        # v→u direction
        Φ(Δt_u), Φ(Δt_v), Φ(Δt_uv),                          # Component 0
        is_cold_u, is_cold_v, is_cold_uv ]                   # cold-start bits

    input_dim = 8·d + 3·d_time + 3
    """

    def __init__(self, d_emb: int, hidden: int, d_time: int):
        super().__init__()
        self.d_time = d_time
        in_d = 8 * d_emb + 3 * d_time + 3
        self.norm = nn.LayerNorm(in_d)
        # 3 layers: input proj + 1 hidden + output.
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        e_t_u: torch.Tensor, e_t_v: torch.Tensor,
        e_c_u: torch.Tensor, e_c_v: torch.Tensor,
        phi_dt_u: torch.Tensor,
        phi_dt_v: torch.Tensor,
        phi_dt_uv: torch.Tensor,
        is_cold_u: torch.Tensor,
        is_cold_v: torch.Tensor,
        is_cold_uv: torch.Tensor,
    ) -> torch.Tensor:
        x_ct = torch.cat([
            e_t_u, e_c_v, e_t_u * e_c_v, (e_t_u - e_c_v).abs(),
            e_t_v, e_c_u, e_t_v * e_c_u, (e_t_v - e_c_u).abs(),
        ], dim=-1)
        x = torch.cat([
            x_ct,
            phi_dt_u, phi_dt_v, phi_dt_uv,
            is_cold_u, is_cold_v, is_cold_uv,
        ], dim=-1)
        return self.net(self.norm(x)).squeeze(-1)
