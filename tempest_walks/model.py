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
    """8-block MLP head with CROSS-TABLE interactions.

    The alignment loss trains the cosine geometry between target(seed) and
    context(walk-neighbour) — i.e. target ↔ context is the supervised
    interaction. The link MLP exposes exactly that interaction (in both
    directions) so the BCE signal can lean on it directly instead of
    re-learning it from scratch:

        input = concat([
          target(u),  context(v),  target(u)·context(v),  |target(u)−context(v)|,   ← u→v direction
          target(v),  context(u),  target(v)·context(u),  |target(v)−context(u)|,   ← v→u direction
        ])  ∈ ℝ^{8·d}

    Both directions are included because the head is called on ordered
    (u, v) pairs but the dataset's directionality isn't always clean —
    letting the MLP weight u→v vs v→u itself is cheaper than committing
    to one direction at the head and losing the signal on the other.

    The arguments are the FEATURE-AUGMENTED vectors coming from
    EmbeddingStore.target / .context — node-feature residuals are folded
    in upstream, so the link MLP gets them for free.
    """

    def __init__(self, d_emb: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        in_d = 8 * d_emb
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
        e_t_u: torch.Tensor, e_t_v: torch.Tensor,
        e_c_u: torch.Tensor, e_c_v: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                # u→v direction: target(u) ↔ context(v) is the trained pair
                e_t_u, e_c_v, e_t_u * e_c_v, (e_t_u - e_c_v).abs(),
                # v→u direction: target(v) ↔ context(u)
                e_t_v, e_c_u, e_t_v * e_c_u, (e_t_v - e_c_u).abs(),
            ],
            dim=-1,
        )
        return self.net(self.norm(x)).squeeze(-1)
