"""Dual-table embedding store + 8-block link MLP.

Composition (concat + final projection — robust to differing feature scales).

Identity tables       E_target, E_context  ∈  ℝ^[n_nodes, d_emb]
                      Always Xavier-uniform init. No feature-based init —
                      that would freeze node features at construction
                      time, breaking streaming-feature datasets.

Node features         Learned at every lookup. The per-feature projection
                      brings raw features into d_emb scale; the per-site
                      final projection learns the channel weighting:
                        target(u)  = target_final(  [E_target[u]  || proj_t(nf[u])] )
                        context(u) = context_final( [E_context[u] || proj_c(nf[u])] )
                      target() and context() are the canonical primitives —
                      EVERY downstream site (link MLP, uniformity, walk
                      context) reads through them, so node-feature fusion
                      happens exactly once per role.

Walk context          Runtime, alignment-loss only. Each walk position
                      represents a node u in its CONTEXT role (someone
                      that has shown up in another node's recent past),
                      augmented by the feature of the hop that connects
                      u to the next position toward the seed:

                        context_walk[p] = context_walk_final(
                            [ context(node[p])        # u in context role
                            ‖ proj_e(edge_feat[p])    # edge (node[p], node[p+1])
                            ]
                        )

                      Edge index p (NOT p-1) matches the timestamp at the
                      same walk position: both describe the hop that
                      leaves position p toward position p+1 — the same
                      edge the alignment loss weights by recency.

                      The `target` table is touched only as the SEED side
                      of alignment, never at walk-internal positions —
                      that asymmetry is what makes the two tables earn
                      their keep (see top-level design note).

                      Edge features never reach the LinkPredictor — per
                      the no-leak rule (negatives don't have edges).

All projection modules are instantiated ONLY when the corresponding
feature is present. Zero params, zero compute on absent channels.
Gradients flow independently into E and each projection via the
optimizer; nothing is mutated in-place during the forward pass.
Streaming feature updates: overwrite the buffer with `update_node_feat`.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EmbeddingStore(nn.Module):
    """Two embedding tables (identity / context) + optional feature residuals.

    `node_feat`: optional numpy array [n_nodes, d_node_feat]. Registered as
                 a non-persistent buffer so .to(device) carries it along.
    `edge_feat_dim`: dim of the per-hop edge feature returned by Tempest's
                     walks (0 if dataset has none).
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

        # Identity tables: Xavier-uniform init, always.
        self.E_target = nn.Embedding(n_nodes, d_emb)
        self.E_context = nn.Embedding(n_nodes, d_emb)
        nn.init.xavier_uniform_(self.E_target.weight)
        nn.init.xavier_uniform_(self.E_context.weight)

        # ── Per-feature projections (bring raw features to d_emb scale) ──
        # Node features. Buffer is non-persistent so checkpoints don't
        # lock in a stale feature matrix; callers can swap the matrix
        # via `update_node_feat`.
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

        # Edge features.
        self.has_edge_feat = edge_feat_dim > 0
        if self.has_edge_feat:
            self.edge_feat_proj = nn.Linear(edge_feat_dim, d_emb)
        else:
            self.edge_feat_proj = None

        # ── Per-site final fusion projections (concat → d_emb) ──────────
        # target / context sites concatenate E with node-feat projection
        # (when present). When no node features, no fusion is needed.
        nf_extra = d_emb if self.has_node_feat else 0
        self.target_final = (
            nn.Linear(d_emb + nf_extra, d_emb) if nf_extra > 0 else None
        )
        self.context_final = (
            nn.Linear(d_emb + nf_extra, d_emb) if nf_extra > 0 else None
        )

        # context_walk site: concatenates context(u) ‖ proj_e(edge).
        # context(u) is d_emb on output (already fuses node features when
        # present). The walk-level final only earns its keep when an edge
        # feature is being mixed in — otherwise it's just context(u).
        ef_extra = d_emb if self.has_edge_feat else 0
        walk_in = d_emb + ef_extra
        self.context_walk_final = (
            nn.Linear(walk_in, d_emb) if ef_extra > 0 else None
        )

    @torch.no_grad()
    def update_node_feat(self, new_node_feat: np.ndarray) -> None:
        """Replace the static node-feature buffer with a fresh matrix.
        Use this on datasets where node features evolve in time —
        between batches/phases the new values are picked up automatically
        by the next `target(...)` / `context(...)` call. Shape must match
        the original (n_nodes, d_node_feat)."""
        if not self.has_node_feat:
            raise RuntimeError("update_node_feat called but EmbeddingStore was "
                               "constructed without node features.")
        new = torch.from_numpy(np.asarray(new_node_feat, dtype=np.float32)).to(
            self.node_feat.device,
        )
        if new.shape != self.node_feat.shape:
            raise ValueError(
                f"shape mismatch: existing {tuple(self.node_feat.shape)} vs "
                f"new {tuple(new.shape)}",
            )
        self.node_feat.copy_(new)

    # ------------------------------------------------------------------ #
    # Lookups (concat raw E with per-feature projections, then a learned
    # final Linear collapses back to d_emb. When no features are present
    # the final projection is None and we just return E directly.)
    # ------------------------------------------------------------------ #

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
        walk_nodes: torch.Tensor,                  # [N*K, L] long, padding-safe (≥0)
        walk_edge_feats: Optional[torch.Tensor],   # [N*K, L-1, d_edge] or None
    ) -> torch.Tensor:
        """Per-position walk representation: context role + hop's edge feature.

        At each walk position p ∈ [0, L):

            context(node[p])             # u in context role, node-feat fused
            ‖ edge_feat_proj(ef[p])      # edge (node[p], node[p+1]), same
                                         # hop as timestamps[p]
            → context_walk_final         # → d_emb

        Edge-feat at the seed position (p = lens-1) doesn't exist (no
        out-going hop) and is right-padded with zeros; the alignment
        loss masks the seed position anyway, so the value there never
        contributes. Padding positions are also masked downstream via
        `lens`.

        The `target` table is intentionally not consumed here — it gets
        gradient only as the seed side of the alignment loss, keeping
        the two-table asymmetry sharp.
        """
        c = self.context(walk_nodes)                # [N*K, L, d_emb]

        if walk_edge_feats is None or not self.has_edge_feat:
            return c

        ef_proj = self.edge_feat_proj(walk_edge_feats.float())             # [N*K, L-1, d_emb]
        # Right-pad: edge_feats[p] sits at position p of the padded tensor
        # so it aligns with timestamps[p] for the same hop (matches the
        # alignment loss's `ts[p]` indexing). Seed position p=lens-1 gets
        # zero — masked out downstream.
        ef_padded = torch.nn.functional.pad(ef_proj, (0, 0, 0, 1))         # [N*K, L,   d_emb]
        return self.context_walk_final(torch.cat([c, ef_padded], dim=-1))


class LinkPredictor(nn.Module):
    """4-channel MLP head: [E(u) ‖ E(v) ‖ W(u) ‖ W(v)].

    Simpler than the prior 12-block design — relies on CrossPairAttention
    (a separate module) to do all the interaction work *before* the head
    sees its inputs. The MLP itself is just a non-linear scorer on the
    concatenated raw inputs; no Hadamard / L1 / cross-table blocks.

    Per-channel contract:
      - E(u), E(v):  identity-table node embeddings (typically target()).
                     Stable per-node, independent of (u, v) pairing.
      - W(u), W(v):  walk-summary embeddings AFTER cross-pair attention —
                     each side has already attended to the other's walk,
                     so co-occurrence / shared-history signal is folded
                     into W before the MLP sees it.

    Why 4 raw channels instead of 12 hand-crafted interactions: the
    cross-pair attention is a learned interaction. Stacking it under
    Hadamard / L1 blocks would just give the MLP two stages of feature
    cross — wasteful and harder to train. The MLP becomes a clean
    "is this combination of identity + walk-context-aware-summaries
    consistent with a positive link?" classifier.
    """

    def __init__(self, d_emb: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        in_d = 4 * d_emb
        self.norm = nn.LayerNorm(in_d)
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        e_u: torch.Tensor,
        e_v: torch.Tensor,
        w_u: torch.Tensor,
        w_v: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([e_u, e_v, w_u, w_v], dim=-1)
        return self.net(self.norm(x)).squeeze(-1)


class CrossPairAttention(nn.Module):
    """DyGFormer-style cross-pair attention between two walk sequences.

    For each (u, v) pair, both walks (pre-pooled per-position encodings)
    attend to each other so the resulting per-position outputs carry
    co-occurrence / shared-history structure. Pooling those gives W(u)
    and W(v) — pair-conditioned walk summaries that feed the link MLP.

    Pre-LN residual block in both directions:
        h_u_out = h_u + MHA( LN(h_u), LN(h_v), LN(h_v) )
        h_v_out = h_v + MHA( LN(h_v), LN(h_u), LN(h_u) )

    Two separate MHAs (one per direction) — they have different roles
    even on undirected datasets: u attending to v looks for "what of u's
    history is supported by v's neighbours", v attending to u looks for
    the symmetric direction.

    `key_padding_mask` is `True` for positions that should be MASKED OUT
    (padding / cold-start beyond the walk's true length). Caller passes
    `(~valid_mask)` where valid_mask is True at real positions.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln_u = nn.LayerNorm(d_model)
        self.ln_v = nn.LayerNorm(d_model)
        self.u_attn_v = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.v_attn_u = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

    def forward(
        self,
        h_u_seq: torch.Tensor,   # [P, L, d]
        h_v_seq: torch.Tensor,   # [P, L, d]
        u_valid_mask: torch.Tensor,  # [P, L]  True where valid (real walk position)
        v_valid_mask: torch.Tensor,  # [P, L]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_u_out, h_v_out) — same shapes as inputs.

        Cold-start guarantee: if any row of u_valid_mask or v_valid_mask is
        all-False, the corresponding row's attention reduces to an identity
        (we override with the residual h_*_seq itself) to avoid NaN.
        """
        # MHA's key_padding_mask: True = MASK OUT
        u_kpm = ~u_valid_mask
        v_kpm = ~v_valid_mask

        # Detect all-masked rows on each side and force at least one valid
        # key for MHA to consume (otherwise softmax over empty produces NaN).
        # We blend the cold-start rows OUT of the attention output below.
        u_all_masked = u_kpm.all(dim=1, keepdim=True)   # [P, 1]
        v_all_masked = v_kpm.all(dim=1, keepdim=True)
        u_kpm_safe = u_kpm.clone()
        v_kpm_safe = v_kpm.clone()
        u_kpm_safe[:, 0] = u_kpm_safe[:, 0] & ~u_all_masked.squeeze(-1)
        v_kpm_safe[:, 0] = v_kpm_safe[:, 0] & ~v_all_masked.squeeze(-1)

        u_q = self.ln_u(h_u_seq)
        v_q = self.ln_v(h_v_seq)

        # u attends to v's walk
        u_attended, _ = self.u_attn_v(
            query=u_q, key=v_q, value=v_q,
            key_padding_mask=v_kpm_safe,
            need_weights=False,
        )
        # v attends to u's walk
        v_attended, _ = self.v_attn_u(
            query=v_q, key=u_q, value=u_q,
            key_padding_mask=u_kpm_safe,
            need_weights=False,
        )

        # Zero out attention contribution for cold-start rows (no info on
        # the other side to attend to). Residual carries them forward.
        u_attended = u_attended * (~v_all_masked).float().unsqueeze(-1)
        v_attended = v_attended * (~u_all_masked).float().unsqueeze(-1)

        return h_u_seq + u_attended, h_v_seq + v_attended


def masked_mean_pool(
    h_seq: torch.Tensor,        # [P, L, d]
    valid_mask: torch.Tensor,   # [P, L]
) -> torch.Tensor:
    """Mean-pool over the valid positions of each row. Rows that are
    all-invalid pool to zero (and the caller should substitute a fallback
    like target(node) before passing to the link MLP)."""
    mask_f = valid_mask.float().unsqueeze(-1)            # [P, L, 1]
    summed = (h_seq * mask_f).sum(dim=1)                  # [P, d]
    counts = mask_f.sum(dim=1).clamp_min(1e-6)            # [P, 1]
    return summed / counts


class WalkEncoder(nn.Module):
    """Single-layer GRU over per-position walk inputs.

    Replaces the per-position `context_walk` fusion in the alignment loss
    with a stateful representation that aggregates the entire walk-prefix
    into each position's hidden state.

    Per-position input at walk position p of walk w:

        x_{w,p} = [ context(u_p)                            # d_emb
                  ‖ role_embed(SEED if p==lens-1 else 0)    # d_role
                  ‖ time_embed(Δt = t_query − ts[p])        # d_time
                  ‖ proj_e(eps[p])  (right-padded at seed)  # d_emb  (if ef)
                  ]

    `context(u)` is the existing embedding primitive (fuses E_context with
    node-feature projection when present). `proj_e` is **shared** with
    EmbeddingStore's `edge_feat_proj` — same projection, two consumers.
    `time_embed` is a small MLP on the (normalised, log-transformed) Δt.

    Direction: the GRU runs forward in chronological order. Position 0 is
    the oldest reachable neighbour; position `lens-1` is the seed.
    `h_{w, lens-1}` therefore carries the full walk's accumulated context.

    Cold-start: walks with `lens == 0` (seeds with no past) get
    `lens.clamp_min(1)` before packing so `pack_padded_sequence` doesn't
    crash. The resulting single-position GRU output is on a junk node
    (the padding `-1` clamped to 0), but the alignment loss masks the
    whole walk out (no non-seed positions when lens ≤ 1) so it contributes
    nothing. Watch this when later phases feed h_seed into the link MLP.

    `d_gru` must equal `d_emb` because the alignment-loss cosine compares
    `target(seed) ∈ ℝ^{d_emb}` with the GRU's hidden state directly.
    """

    def __init__(
        self,
        embedding_store: "EmbeddingStore",
        d_gru: int = 128,
        d_time: int = 16,
        d_role: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_emb = embedding_store.d_emb
        if d_gru != d_emb:
            raise ValueError(
                f"d_gru ({d_gru}) must equal d_emb ({d_emb}) — the alignment "
                f"cosine compares target(seed) ∈ R^d_emb with GRU outputs."
            )
        # Hold a reference to EmbeddingStore WITHOUT registering it as a
        # submodule — that way `self.parameters()` does NOT double-count
        # E_target/E_context/proj_*/etc. (which the trainer already optimizes
        # via embedding_store.parameters()). Gradient flow through
        # `self.embedding_store.context(...)` still works because the
        # autograd graph is independent of module registration.
        object.__setattr__(self, "embedding_store", embedding_store)
        self.d_emb = d_emb
        self.d_gru = d_gru
        self.d_time = d_time
        self.d_role = d_role
        self.has_edge_feat = embedding_store.has_edge_feat

        # 2 role indices: 0 = INTERMEDIATE, 1 = SEED.
        self.role_embed = nn.Embedding(2, d_role)
        # Time MLP on [Δt_norm, log(1 + Δt_norm)]. Two-layer GELU.
        self.mlp_time = nn.Sequential(
            nn.Linear(2, d_time),
            nn.GELU(),
            nn.Linear(d_time, d_time),
        )

        d_in = d_emb + d_role + d_time
        if self.has_edge_feat:
            d_in += d_emb  # projected edge feat (shared proj_e)

        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=d_gru,
            num_layers=1,
            batch_first=True,
        )
        self.input_dropout = nn.Dropout(dropout)

    def forward(
        self,
        walk_nodes: torch.Tensor,         # [W, L] long, padding=clamped to 0
        walk_timestamps: torch.Tensor,    # [W, L] int64; seed slot = INT64_MAX sentinel
        lens: torch.Tensor,               # [W] long
        walk_edge_feats: Optional[torch.Tensor],  # [W, L-1, d_e] or None
        t_query: torch.Tensor,            # [W] int64 (per-walk query time)
        time_scale: float,
    ) -> torch.Tensor:
        """Returns h: [W, L, d_gru]. Positions ≥ lens_w are zero-padded
        (pad_packed_sequence default). The alignment-loss mask still
        excludes them via the same `lens` it already uses.
        """
        device = walk_nodes.device
        W, L = walk_nodes.shape

        # Identity + (optional) node-feat fusion through the canonical
        # primitive — proj_c and context_final are reused, not duplicated.
        c = self.embedding_store.context(walk_nodes)                # [W, L, d_emb]

        # Role embedding: SEED at lens-1, INTERMEDIATE elsewhere.
        positions = torch.arange(L, device=device).unsqueeze(0)     # [1, L]
        seed_pos = (lens - 1).clamp_min(0).unsqueeze(1)             # [W, 1]
        is_seed = (positions == seed_pos).long()                    # [W, L]
        role = self.role_embed(is_seed)                             # [W, L, d_role]

        # Time embedding. Δt = t_query − timestamps[p], clamped at 0.
        # The seed slot's timestamp is the INT64_MAX sentinel — replace
        # it with t_query so Δt at the seed is 0 (cleanly "now").
        t_q = t_query.unsqueeze(1)                                  # [W, 1]
        ts_for_dt = torch.where(positions == seed_pos, t_q, walk_timestamps)
        dt = (t_q - ts_for_dt).clamp_min(0).float()                 # [W, L]
        dt_norm = dt / max(time_scale, 1e-6)
        time_input = torch.stack(                                   # [W, L, 2]
            [dt_norm, torch.log1p(dt_norm)], dim=-1,
        )
        time_h = self.mlp_time(time_input)                          # [W, L, d_time]

        parts = [c, role, time_h]
        if self.has_edge_feat:
            if walk_edge_feats is not None:
                ef = self.embedding_store.edge_feat_proj(walk_edge_feats.float())  # [W, L-1, d_emb]
                # Right-pad to align edge_feats[p] with timestamps[p] at the
                # same walk position; seed slot gets zero.
                ef = F.pad(ef, (0, 0, 0, 1))                        # [W, L, d_emb]
            else:
                # Empty Tempest (first batch after reset) returns no edge
                # feats. The walks are also empty (lens=0) so the alignment
                # loss masks them anyway, but the GRU still needs an input
                # of the right width — feed zeros.
                ef = torch.zeros(W, L, self.d_emb, dtype=c.dtype, device=device)
            parts.append(ef)

        x = torch.cat(parts, dim=-1)                                # [W, L, d_in]
        x = self.input_dropout(x)

        # Pack and run GRU. Clamp lens ≥ 1 for cold-start safety; the
        # alignment loss masks lens ≤ 1 walks out anyway, so the junk
        # single-position computation doesn't contribute. enforce_sorted
        # False lets us skip the sort.
        safe_lens = lens.clamp_min(1).to("cpu")  # PackedSequence needs CPU lengths
        packed = pack_padded_sequence(x, safe_lens, batch_first=True, enforce_sorted=False)
        h_packed, _ = self.gru(packed)
        h, _ = pad_packed_sequence(h_packed, batch_first=True, total_length=L)
        return h                                                    # [W, L, d_gru]
