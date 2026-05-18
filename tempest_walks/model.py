"""Dual-table embedding store + 8-block link MLP + universal feature fusion.

Node and edge features, when the dataset provides them, are folded into
the embedding lookups as learned residuals:

  target(ids)        = E_target[ids]   (+ node_feat_proj_t(node_feat[ids]))
  context(ids)       = E_context[ids]  (+ node_feat_proj_c(node_feat[ids]))
  context_walk(...)  = context(walk_nodes) (+ edge_feat_proj(walk_edge_feats)
                                            at positions p ≥ 1)

The projections are ONLY instantiated when the dataset has the
corresponding feature — zero params, zero compute on absent channels.
Both training (alignment loss) and link MLP (via the same methods) pick
the features up automatically.
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

        self.E_target = nn.Embedding(n_nodes, d_emb)
        self.E_context = nn.Embedding(n_nodes, d_emb)
        nn.init.xavier_uniform_(self.E_target.weight)
        nn.init.xavier_uniform_(self.E_context.weight)

        # ── optional: static node features ──────────────────────────────
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

        # ── optional: per-hop edge features (used only in walk context) ─
        self.has_edge_feat = edge_feat_dim > 0
        if self.has_edge_feat:
            self.edge_feat_proj = nn.Linear(edge_feat_dim, d_emb)
        else:
            self.edge_feat_proj = None

    # ------------------------------------------------------------------ #
    # Lookups (feature-augmented when the dataset provides features).
    # ------------------------------------------------------------------ #

    def target(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.E_target(ids)
        if self.has_node_feat:
            x = x + self.node_feat_proj_target(self.node_feat[ids])
        return x

    def context(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.E_context(ids)
        if self.has_node_feat:
            x = x + self.node_feat_proj_context(self.node_feat[ids])
        return x

    def context_walk(
        self,
        walk_nodes: torch.Tensor,                  # [N*K, L] long, padding-safe (≥0)
        walk_edge_feats: Optional[torch.Tensor],   # [N*K, L-1, d_edge] or None
    ) -> torch.Tensor:
        """Per-position context vector for a batch of walks.

        Node-feature residuals apply per node (via `context`); edge-feature
        residuals are added at position p ≥ 1 (the edge between (p-1, p)
        lands on position p). Position 0 has no incoming edge → no residual.
        Padding positions (lens ≤ p < L) are NOT specially masked here —
        the downstream alignment loss masks them out via `lens`.
        """
        x = self.context(walk_nodes)                          # [N*K, L, d]
        if walk_edge_feats is not None and self.has_edge_feat:
            ef_proj = self.edge_feat_proj(walk_edge_feats.float())  # [N*K, L-1, d]
            # Pad the leading position with zeros to align with L.
            ef_padded = torch.nn.functional.pad(ef_proj, (0, 0, 1, 0))
            x = x + ef_padded
        return x


class LinkPredictor(nn.Module):
    """8-block MLP head:
        input = concat([
          E_t[u], E_t[v], E_t[u]·E_t[v], |E_t[u]−E_t[v]|,
          E_c[u], E_c[v], E_c[u]·E_c[v], |E_c[u]−E_c[v]|,
        ])  ∈ ℝ^{8·d}
    Returns raw logits (no sigmoid — paired with BCE-with-logits).

    The `target()` / `context()` arguments are the FEATURE-AUGMENTED vectors
    coming from EmbeddingStore.target / .context — node-feature residuals
    are folded in upstream, so the link MLP gets them for free.
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
                e_t_u, e_t_v, e_t_u * e_t_v, (e_t_u - e_t_v).abs(),
                e_c_u, e_c_v, e_c_u * e_c_v, (e_c_u - e_c_v).abs(),
            ],
            dim=-1,
        )
        return self.net(self.norm(x)).squeeze(-1)
