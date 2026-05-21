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
        single_table: bool = False,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_emb = d_emb
        self.single_table = single_table

        # Identity tables: Xavier-uniform init, always.
        # Single-table mode (v2.4 §13 1T_asym): E_target and E_context
        # alias the same nn.Embedding. P_src/P_tgt asymmetry is provided
        # by target_final/context_final projections below, which are
        # always-present in single-table mode (even when no node_feat).
        if single_table:
            shared_E = nn.Embedding(n_nodes, d_emb)
            nn.init.xavier_uniform_(shared_E.weight)
            self.E_target = shared_E
            self.E_context = shared_E
        else:
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
        # SINGLE-TABLE mode: target_final / context_final are always
        # present (Linear(d, d) when no node_feat) — they act as
        # P_src / P_tgt projections providing the role asymmetry that
        # was previously encoded in two separate embedding tables.
        nf_extra = d_emb if self.has_node_feat else 0
        need_final = (nf_extra > 0) or single_table
        self.target_final = (
            nn.Linear(d_emb + nf_extra, d_emb) if need_final else None
        )
        self.context_final = (
            nn.Linear(d_emb + nf_extra, d_emb) if need_final else None
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
        if self.has_node_feat:
            nf_proj = self.node_feat_proj_target(self.node_feat[ids])
            e = torch.cat([e, nf_proj], dim=-1)
        # target_final is always present in single_table mode (acts as
        # P_src projection) or when node_feat is present (fusion). When
        # neither, we return the raw lookup.
        if self.target_final is not None:
            e = self.target_final(e)
        return e

    def context(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.E_context(ids)
        if self.has_node_feat:
            nf_proj = self.node_feat_proj_context(self.node_feat[ids])
            e = torch.cat([e, nf_proj], dim=-1)
        # context_final is always present in single_table mode (acts as
        # P_tgt projection) or when node_feat is present (fusion).
        if self.context_final is not None:
            e = self.context_final(e)
        return e

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


class TimeEncoder(nn.Module):
    """Functional time encoding (Xu et al. 2020) — Φ(Δt) ∈ ℝ^{2k}.

    Standard recipe used by TGAT / TGN / DyGFormer:

        Φ(Δt) = [cos(ω_1·Δt), sin(ω_1·Δt), ..., cos(ω_k·Δt), sin(ω_k·Δt)]

    The k frequencies ω_i are LEARNABLE parameters (initialised to a
    geometric schedule so the encoder starts as a Fourier-style basis).
    d_time = 2·k.

    Input is expected to be Δt PRE-CLAMPED to a sane range (the caller is
    responsible — Component 0 of the design clamps Δt to time_scale × 100
    so cold-start sentinel values don't make the learned ω_i explode).

    The buffer `inv_freq_init` keeps the geometric init in `state_dict` for
    diagnostics — the actual trainable values are stored in `self.omegas`.
    """

    def __init__(self, k: int = 16, time_scale: float = 1.0):
        super().__init__()
        self.k = int(k)
        # Initialise ω_i geometrically over a wide range (Xu et al. 2020).
        # The scale is normalised against time_scale so the encoder is
        # dataset-aware at init: ω_i covers periods from ~time_scale up to
        # ~time_scale / 1000 (i.e., very-recent variation is at high freq,
        # long-ago variation at low freq).
        i = torch.arange(k, dtype=torch.float32)
        init_omegas = (1.0 / time_scale) * (1000.0 ** (-i / max(k - 1, 1)))
        self.omegas = nn.Parameter(init_omegas)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """dt: [..., ] float tensor. Returns [..., 2k] tensor with
        [cos(ω_1·dt), sin(ω_1·dt), ..., cos(ω_k·dt), sin(ω_k·dt)] along the
        last axis."""
        # dt: [..., ], omegas: [k]
        phases = dt.unsqueeze(-1) * self.omegas                            # [..., k]
        # Stack cos and sin along last axis, then flatten so it's
        # [cos_1, sin_1, cos_2, sin_2, ...] in pairs.
        cos = torch.cos(phases)
        sin = torch.sin(phases)
        # Interleave so neighboring entries are (cos_i, sin_i):
        stacked = torch.stack([cos, sin], dim=-1)                          # [..., k, 2]
        return stacked.flatten(start_dim=-2)                                # [..., 2k]


class LinkPredictor(nn.Module):
    """Configurable head — 8-block cross-table and/or Component 0 time encoding.

    The alignment loss trains the cosine geometry between target(seed) and
    context(walk-neighbour) — i.e. target ↔ context is the supervised
    interaction. The link MLP exposes exactly that interaction (in both
    directions) so the BCE signal can lean on it directly instead of
    re-learning it from scratch.

    Heads (Phase S Group E):
      E.1 `head_mode="cross_table"` + `cross_table_dropout=0` (default; anchor):
          [ target(u), context(v), target(u)⊙context(v), |target(u)−context(v)|,   ← u→v
            target(v), context(u), target(v)⊙context(u), |target(v)−context(u)|,   ← v→u
            Φ(Δt_u), Φ(Δt_v), Φ(Δt_uv),                                            ← Component 0
            is_cold_start_u, is_cold_start_v, is_cold_start_uv ]                   ← bits
          input dim = 8·d + 3·d_time + 3

      E.2 `head_mode="component_0_only"`: drops the 8-block cross-table entirely.
          [ Φ(Δt_u), Φ(Δt_v), Φ(Δt_uv), is_cold_start_u, is_cold_start_v, is_cold_start_uv ]
          input dim = 3·d_time + 3
          Requires `use_time_encoding=True` (otherwise the head has no inputs).

      E.3 `head_mode="cross_table"` + `cross_table_dropout > 0`: same layout as E.1,
          but the 8·d cross-table block is passed through nn.Dropout(p) BEFORE
          concatenation with the Component 0 block.

    The Component 0 inputs are gated by `use_time_encoding=True` at construction.
    """

    def __init__(
        self,
        d_emb: int,
        hidden: int = 128,
        dropout: float = 0.0,
        use_time_encoding: bool = False,
        d_time: int = 32,
        head_mode: str = "cross_table",
        cross_table_dropout: float = 0.0,
        n_layers: int = 3,
    ):
        super().__init__()
        if head_mode not in ("cross_table", "component_0_only"):
            raise ValueError(
                f"head_mode must be 'cross_table' or 'component_0_only', got {head_mode!r}"
            )
        if head_mode == "component_0_only" and not use_time_encoding:
            raise ValueError(
                "head_mode='component_0_only' requires use_time_encoding=True "
                "(otherwise the head has no inputs)."
            )
        if n_layers < 2:
            raise ValueError(f"n_layers must be ≥ 2 (input projection + output), got {n_layers}")
        self.use_time_encoding = use_time_encoding
        self.d_time = d_time
        self.head_mode = head_mode
        self.cross_table_dropout = nn.Dropout(cross_table_dropout) if cross_table_dropout > 0 else None
        self.n_layers = n_layers

        in_d = 0
        if head_mode == "cross_table":
            in_d += 8 * d_emb
        if use_time_encoding:
            in_d += 3 * d_time + 3
        self.norm = nn.LayerNorm(in_d)
        # Build an n_layer MLP head: input projection (Linear in_d -> hidden) +
        # (n_layers - 2) GELU+Dropout+Linear(hidden,hidden) blocks + final
        # Linear(hidden, 1). Default n_layers=3 reproduces the original 3-layer
        # head (input proj + 1 hidden block + output). Sweep to 5 for §4.8.2.
        layers: list = [nn.Linear(in_d, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        e_t_u: torch.Tensor, e_t_v: torch.Tensor,
        e_c_u: torch.Tensor, e_c_v: torch.Tensor,
        phi_dt_u: Optional[torch.Tensor] = None,
        phi_dt_v: Optional[torch.Tensor] = None,
        phi_dt_uv: Optional[torch.Tensor] = None,
        is_cold_u: Optional[torch.Tensor] = None,
        is_cold_v: Optional[torch.Tensor] = None,
        is_cold_uv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = []
        if self.head_mode == "cross_table":
            x_ct = torch.cat([
                # u→v direction: target(u) ↔ context(v) is the trained pair
                e_t_u, e_c_v, e_t_u * e_c_v, (e_t_u - e_c_v).abs(),
                # v→u direction: target(v) ↔ context(u)
                e_t_v, e_c_u, e_t_v * e_c_u, (e_t_v - e_c_u).abs(),
            ], dim=-1)
            if self.cross_table_dropout is not None:
                x_ct = self.cross_table_dropout(x_ct)
            parts.append(x_ct)
        if self.use_time_encoding:
            if any(t is None for t in (phi_dt_u, phi_dt_v, phi_dt_uv, is_cold_u, is_cold_v, is_cold_uv)):
                raise ValueError(
                    "LinkPredictor was constructed with use_time_encoding=True "
                    "but one of phi_dt_*/is_cold_* was None at forward time."
                )
            parts.extend([phi_dt_u, phi_dt_v, phi_dt_uv])
            # Cold-start bits as 3 separate scalars (one per pair-row).
            # Shape: [P, 1] each.
            parts.extend([is_cold_u, is_cold_v, is_cold_uv])
        x = torch.cat(parts, dim=-1)
        return self.net(self.norm(x)).squeeze(-1)
