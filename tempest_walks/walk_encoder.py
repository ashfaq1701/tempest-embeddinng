"""Source-side walk encoder — transition-pair design (Lesson 31).

A 1-layer GRU that consumes Tempest walks seeded on `u` and produces a
time-aware, history-aware source representation `walk_repr[u]`. Replaces
the static `E_target[u]` slot in the link MLP's source-side input.

Per-step input is a complete *transition tuple*. Each non-seed step
carries (source node, destination node, time, edge feature). The seed
step is a distinct *terminal* step — it has no outgoing edge so the
destination + edge slots are explicit zeros.

  For p ∈ [0, lens-2] (transition steps):
    step_input[p] = concat([
        context(node_p),         # source of the transition
        context(node_{p+1}),     # destination of the transition
        Φ(t_query - t_p),        # time of the transition (relative to query)
        ef_proj(ef[p]),          # edge feature of the transition  (if has_edge_feat)
    ])

  For p == lens-1 (seed step, terminal):
    step_input[lens-1] = concat([
        target(seed),            # the seed in source-role (target lookup)
        zeros(d_emb),            # no destination — terminal
        Φ(t_query - t_seed),     # Δt = 0 at scoring time
        zeros(d_edge_proj),      # no outgoing edge          (if has_edge_feat)
    ])

This contrasts with the pre-Lesson-31 design, where each step carried
only one node's identity AND that node's outgoing edge — forcing the
GRU to compose (n_p, ef_p) with (n_{p+1}) across two recurrent steps.
The new design hands the transition tuple to one step, freeing GRU
capacity. It also dissolves the seed-step edge-feat asymmetry (Issue 1
in the line-by-line audit): the seed step has no edge slot in the
transition sense, so there's no ambiguity over short-vs-full walks.

The GRU processes walks left-to-right (oldest → newest). Last hidden
state corresponds to the seed position. Mean-pool over K walks per
seed gives `walk_repr[u]`.

Gradient flow (Option α — alignment+uniformity train tables separately):
  ∂L_link / ∂walk_encoder.parameters     ← BCE backprop into GRU
  ∂L_link / ∂E[walk_node_i]              ← BCE also trains E via lookups
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
        # Per-step input dimensions (Lesson 31, transition-pair design):
        #   src (d_emb) + dst (d_emb) + Φ(Δt) (d_time) + ef_proj (d_emb if has_edge_feat).
        d_step = 2 * d_emb + d_time + (d_emb if has_edge_feat else 0)
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
        embedding_store,                     # access to target() / context() / E_target/E_context
        time_encoder,                        # Φ
        edge_feats_padded: Optional[torch.Tensor],  # [N*K, L, d_emb] projected + right-padded
        K: int,
    ) -> torch.Tensor:
        """Returns walk_repr: [N, d_emb] — mean-pooled GRU output at seed position."""
        device = walk_nodes.device
        NK, L = walk_nodes.shape
        N = NK // K

        positions = torch.arange(L, device=device).unsqueeze(0)        # [1, L]
        seed_pos_idx = (walk_lens - 1).clamp_min(0).unsqueeze(1)       # [N*K, 1]
        is_seed_pos = positions == seed_pos_idx                        # [N*K, L]
        not_seed_f = (~is_seed_pos).unsqueeze(-1).float()              # [N*K, L, 1]

        clamped_nodes = walk_nodes.clamp_min(0)
        e_t = embedding_store.E_target(clamped_nodes)                  # [N*K, L, d_emb]
        e_c = embedding_store.E_context(clamped_nodes)                 # [N*K, L, d_emb]
        # Source side at each step: TARGET at the seed, CONTEXT elsewhere.
        # (The seed in the link MLP's source slot is `target(u)`; the
        # walk-internal nodes are u's destinations — context() role.)
        src_step = torch.where(is_seed_pos.unsqueeze(-1), e_t, e_c)    # [N*K, L, d_emb]

        # Destination side at each non-seed step: CONTEXT of the next node
        # in the walk (n_{p+1}). For the seed step (p == lens-1), there
        # is no next node — zero out. For post-seed padding positions
        # (only matters for short walks at p > lens-1), `next_nodes`
        # picks up further padding (-1 → clamped to 0); these positions
        # are post-seed in the GRU's chronological read, never read at
        # the output, so the value doesn't matter.
        next_nodes = torch.cat(
            [clamped_nodes[:, 1:], torch.zeros_like(clamped_nodes[:, :1])],
            dim=1,
        )                                                              # [N*K, L]
        dst_step = embedding_store.E_context(next_nodes)               # [N*K, L, d_emb]
        dst_step = dst_step * not_seed_f                               # zero at seed

        # Φ(t_query - t_p). Padding positions have walk_timestamps=-1;
        # (t_query - (-1)) is some large positive number, but those
        # positions are post-seed and never read at the GRU output.
        t_query_bc = t_query.unsqueeze(1).expand(-1, L)                # [N*K, L]
        dt = (t_query_bc - walk_timestamps).clamp_min(0).float()       # [N*K, L]
        dt_enc = time_encoder(dt)                                      # [N*K, L, d_time]

        parts = [src_step, dst_step, dt_enc]
        if self.has_edge_feat and edge_feats_padded is not None:
            # edge_feats_padded is the trainer's projected+right-padded
            # ef tensor. Zero out at the seed step (no outgoing edge in
            # the transition sense). This also resolves the audit's
            # Issue 1: under the transition-pair design the seed-step
            # edge-feat slot is canonically zero regardless of
            # whether the walk reached max_walk_len.
            ef_for_step = edge_feats_padded * not_seed_f
            parts.append(ef_for_step)
        step_input = torch.cat(parts, dim=-1)

        # GRU over the full L (padded). We read the hidden state at the
        # seed position (lens-1), so post-seed padding doesn't pollute
        # the representation we use.
        gru_out, _ = self.gru(step_input)                              # [N*K, L, d_emb]
        # Gather hidden state at seed position for each walk.
        seed_gather = (walk_lens - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, self.d_emb)
        walk_repr_per_walk = gru_out.gather(1, seed_gather).squeeze(1)  # [N*K, d_emb]

        # Mean-pool over K walks per seed.
        walk_repr = walk_repr_per_walk.view(N, K, self.d_emb).mean(dim=1)  # [N, d_emb]
        return walk_repr
