"""Source-side walk encoder (v2.4 §14).

A 1-layer GRU that consumes Tempest walks seeded on `u` and produces a
time-aware, history-aware source representation `walk_repr[u]`. Replaces
the static `E_target[u]` slot in the link MLP's source-side input.

Per-step input (dual-table locked architecture):
  [E_target[node_i] if i == L-1 else E_context[node_i],
   Φ(t_seed - t_i),
   edge_features[i]]   # zero-padded at i = 0 (no incoming edge for the
                       # first walk node)

The GRU processes walks left-to-right (oldest → newest). Last hidden
state corresponds to the seed position. Mean-pool over K walks per
seed gives `walk_repr[u]`.

Gradient flow (Option α, default — alignment+normbrake stay on):
  ∂L_link / ∂walk_encoder.parameters     ← BCE backprop into GRU
  ∂L_link / ∂E[walk_node_i]              ← BCE *also* trains E via lookups
  ∂L_align / ∂E_{target,context}         ← alignment trains E separately

The walk encoder is jointly trained with link BCE; the embedding tables
get gradient from BOTH alignment (decoupled) AND link BCE (through the
encoder).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WalkEncoder(nn.Module):
    def __init__(
        self,
        d_emb: int,
        d_time: int = 32,
        has_edge_feat: bool = False,
    ):
        super().__init__()
        self.d_emb = d_emb
        self.d_time = d_time
        self.has_edge_feat = has_edge_feat
        d_step = d_emb + d_time + (d_emb if has_edge_feat else 0)
        self.gru = nn.GRU(
            input_size=d_step,
            hidden_size=d_emb,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward(
        self,
        walk_nodes: torch.Tensor,            # [N*K, L] long
        walk_timestamps: torch.Tensor,       # [N*K, L] long
        walk_lens: torch.Tensor,             # [N*K] long; valid positions = [0, lens-1]
        t_query: torch.Tensor,               # [N*K] long
        embedding_store,                     # access to E_target / E_context
        time_encoder,                        # Φ
        edge_feats_padded: Optional[torch.Tensor],  # [N*K, L, d_emb] projected + zero-padded
        K: int,
    ) -> torch.Tensor:
        """Returns walk_repr: [N, d_emb] — mean-pooled GRU output at seed position."""
        device = walk_nodes.device
        NK, L = walk_nodes.shape
        N = NK // K

        positions = torch.arange(L, device=device).unsqueeze(0)        # [1, L]
        seed_pos_idx = (walk_lens - 1).clamp_min(0).unsqueeze(1)       # [N*K, 1]
        is_seed_pos = positions == seed_pos_idx                        # [N*K, L]

        clamped_nodes = walk_nodes.clamp_min(0)
        e_t = embedding_store.E_target(clamped_nodes)                  # [N*K, L, d_emb]
        e_c = embedding_store.E_context(clamped_nodes)                 # [N*K, L, d_emb]
        # Seed gets target embedding (source role); walk-internal gets
        # context embedding (destination role, what u historically connected to).
        e_step = torch.where(is_seed_pos.unsqueeze(-1), e_t, e_c)      # [N*K, L, d_emb]

        # Φ(t_seed - t_i). Padding positions get t_i = -1 (sentinel); Δt
        # becomes very large negative which we clamp to 0 — encoder treats
        # padded steps as "right now" but their hidden-state contribution
        # is bounded since the GRU is read at seed_pos (always within valid).
        t_query_bc = t_query.unsqueeze(1).expand(-1, L)                # [N*K, L]
        dt = (t_query_bc - walk_timestamps).clamp_min(0).float()       # [N*K, L]
        dt_enc = time_encoder(dt)                                      # [N*K, L, d_time]

        step_input = torch.cat([e_step, dt_enc], dim=-1)
        if self.has_edge_feat and edge_feats_padded is not None:
            step_input = torch.cat([step_input, edge_feats_padded], dim=-1)

        # GRU over the full L (padded). We read the hidden state at the
        # SEED position (lens-1), so post-seed padding doesn't pollute the
        # representation we use. Pre-seed positions are all valid by
        # construction (Tempest never returns "holes" — only trailing
        # padding).
        gru_out, _ = self.gru(step_input)                              # [N*K, L, d_emb]
        # Gather hidden state at seed position for each walk.
        seed_gather = (walk_lens - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, self.d_emb)
        walk_repr_per_walk = gru_out.gather(1, seed_gather).squeeze(1)  # [N*K, d_emb]

        # Mean-pool over K walks per seed.
        walk_repr = walk_repr_per_walk.view(N, K, self.d_emb).mean(dim=1)  # [N, d_emb]
        return walk_repr
