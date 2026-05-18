"""Dual-table embedding store + 8-block link MLP.

Node features  — INIT ONLY. When `node_feat` is supplied, a 1–2 layer
  MLP projects each node's raw feature vector to d_emb and that result
  is copy_()'d into E_target / E_context as the initial table values
  (separate projections for the two roles). The MLP weights are
  discarded after __init__; from training time onward the model treats
  the tables as ordinary learnable embeddings.
  When `node_feat` is None, Xavier-uniform init.

Edge features — runtime, alignment-loss only. `edge_feat_proj :
  Linear(d_ef → d_emb)` projects each per-hop edge feature, and the
  result is added as a residual to the corresponding walk-context
  position (the edge between (p-1, p) lands on position p; position 0
  is padded with zero). Edge features never reach the LinkPredictor.
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
        node_feat_init_layers: int = 2,           # 1 or 2 — depth of the init MLP
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_emb = d_emb

        self.E_target = nn.Embedding(n_nodes, d_emb)
        self.E_context = nn.Embedding(n_nodes, d_emb)

        # ── init: node-features (when present) project once, then the
        #         projection is discarded; the two tables are ordinary
        #         learnable embeddings from training time onward. ──────
        if node_feat is not None:
            self._init_from_node_features(
                np.asarray(node_feat, dtype=np.float32),
                num_layers=node_feat_init_layers,
            )
        else:
            nn.init.xavier_uniform_(self.E_target.weight)
            nn.init.xavier_uniform_(self.E_context.weight)

        # ── runtime: per-hop edge features → residual in walk context ──
        self.has_edge_feat = edge_feat_dim > 0
        if self.has_edge_feat:
            self.edge_feat_proj = nn.Linear(edge_feat_dim, d_emb)
        else:
            self.edge_feat_proj = None

    @torch.no_grad()
    def _init_from_node_features(self, node_feat: np.ndarray, num_layers: int) -> None:
        """Initialize E_target / E_context from a 1-2 layer projection of
        node_feat. Two SEPARATE projections (one per role) so the two
        tables don't start out identical. Projection modules are discarded
        once weights are copied — no node-feature pathway at runtime.
        """
        assert num_layers in (1, 2), f"node_feat_init_layers must be 1 or 2, got {num_layers}"
        d_nf = int(node_feat.shape[1])
        nf = torch.from_numpy(node_feat).float()                # [n_nodes, d_nf]

        def _build_proj() -> nn.Module:
            if num_layers == 1:
                return nn.Linear(d_nf, self.d_emb)
            return nn.Sequential(
                nn.Linear(d_nf, self.d_emb),
                nn.GELU(),
                nn.Linear(self.d_emb, self.d_emb),
            )

        proj_t = _build_proj()
        proj_c = _build_proj()
        # Xavier-style init on the projection layers so the projected
        # embeddings have unit-ish variance, comparable to xavier_uniform_
        # init for the no-features path.
        for p in list(proj_t.parameters()) + list(proj_c.parameters()):
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

        self.E_target.weight.copy_(proj_t(nf))                  # [n_nodes, d_emb]
        self.E_context.weight.copy_(proj_c(nf))
        # proj_t / proj_c are local — they go out of scope here and are
        # released. No persistent node-feature module remains on self.

    # ------------------------------------------------------------------ #
    # Lookups (raw — node features no longer enter at runtime).
    # ------------------------------------------------------------------ #

    def target(self, ids: torch.Tensor) -> torch.Tensor:
        return self.E_target(ids)

    def context(self, ids: torch.Tensor) -> torch.Tensor:
        return self.E_context(ids)

    def context_walk(
        self,
        walk_nodes: torch.Tensor,                  # [N*K, L] long, padding-safe (≥0)
        walk_edge_feats: Optional[torch.Tensor],   # [N*K, L-1, d_edge] or None
    ) -> torch.Tensor:
        """Per-position context vector for a batch of walks.

        Edge-feature residuals are added at position p ≥ 1 (the edge
        between (p-1, p) lands on position p). Position 0 has no incoming
        edge → no residual. Padding positions are NOT masked here —
        the downstream alignment loss masks them out via `lens`.
        """
        x = self.context(walk_nodes)                          # [N*K, L, d]
        if walk_edge_feats is not None and self.has_edge_feat:
            ef_proj = self.edge_feat_proj(walk_edge_feats.float())  # [N*K, L-1, d]
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
