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

Walk-position features now come from `WalkEncoder` (a GRU over the
chronological walk sequence), not from the legacy `context_walk`
per-position fusion that lived here. The walk encoder consumes
`context(u)` at every hop, so `proj_c` / `context_final` are still
exercised — they just aren't bundled with `proj_e` inside this module.
`edge_feat_proj` is shared with the walk encoder so the per-edge
projection is defined exactly once.

All projection modules are instantiated ONLY when the corresponding
feature is present. Zero params, zero compute on absent channels.
Gradients flow independently into E and each projection via the
optimizer; nothing is mutated in-place during the forward pass.
Streaming feature updates: overwrite the buffer with `update_node_feat`.
"""

from typing import Optional, Tuple

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

        # NOTE: the legacy `context_walk` per-position fusion module that
        # used to live here has been removed — the WalkEncoder (in this
        # file, below) now does all walk-position encoding. We keep
        # `edge_feat_proj` because it's reused by the walk encoder and
        # by the DyGFormer-style NodeEncoder.

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


class LinkPredictor(nn.Module):
    """12-block link MLP (Phase 2's empirically-best structure) plus an
    OPTIONAL co-occurrence channel:

      Group A (8 cross-table interaction blocks):
        target(u), context(v), target(u)⊙context(v), |target(u)−context(v)|,
        target(v), context(u), target(v)⊙context(u), |target(v)−context(u)|
      Group B (4 walk-encoded blocks):
        W(u), W(v), W(u)⊙W(v), |W(u)−W(v)|
      Group C (1 optional co-occurrence block):
        co_feat   (projection of per-pair history overlap statistics)

    target(u) here may include the DyGFormer-style node-encoder residual
    (caller decides whether `target_in` is the raw lookup or the lookup
    plus node_h). Same for context.

    History note: Phase 3 tried collapsing this to 4 channels with cross-
    pair attention doing all the interaction work — that REGRESSED test
    MRR by ~0.05 on tgbl-wiki. The Hadamard/L1 cross-table interactions
    were doing more work than cross-pair attention added. Keeping the
    12-block structure is the empirically-validated choice.
    """

    def __init__(
        self,
        d_emb: int,
        hidden: int = 128,
        dropout: float = 0.0,
        use_co_feat: bool = False,
    ):
        super().__init__()
        self.use_co_feat = use_co_feat
        # 8 cross-table + 4 walk-encoded + (1 if co_feat) — all d_emb wide
        n_blocks = 8 + 4 + (1 if use_co_feat else 0)
        in_d = n_blocks * d_emb
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
        e_t_u: torch.Tensor,
        e_t_v: torch.Tensor,
        e_c_u: torch.Tensor,
        e_c_v: torch.Tensor,
        w_u: torch.Tensor,
        w_v: torch.Tensor,
        co_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [
            # u→v cross-table: target(u) interacts with context(v)
            e_t_u, e_c_v, e_t_u * e_c_v, (e_t_u - e_c_v).abs(),
            # v→u cross-table: target(v) interacts with context(u)
            e_t_v, e_c_u, e_t_v * e_c_u, (e_t_v - e_c_u).abs(),
            # walk-encoded interaction
            w_u, w_v, w_u * w_v, (w_u - w_v).abs(),
        ]
        if self.use_co_feat:
            if co_feat is None:
                raise ValueError(
                    "LinkPredictor was built with use_co_feat=True but "
                    "co_feat was not passed to forward()."
                )
            parts.append(co_feat)
        x = torch.cat(parts, dim=-1)
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


class CoOccurrenceEncoder(nn.Module):
    """Per-pair co-occurrence feature from per-node interaction histories.

    For each (u, v) pair, count how many distinct neighbours are shared
    between u's and v's K-most-recent-interaction sets. This is the
    recurrence signal EdgeBank scores with a hash table, made into a
    differentiable input feature for the link MLP.

    Three statistics computed and projected to d_emb together:
        1. raw overlap count             — |neighbors(u) ∩ neighbors(v)|
        2. overlap / min(vc_u, vc_v)     — normalized by the smaller history
        3. overlap / sqrt(vc_u · vc_v)   — Jaccard-ish geometric normalization

    All three are scalars; a 3 → d_emb projection lifts them to the link
    MLP's expected width. Doing all three is cheap (~20 FLOPs per pair)
    and lets the model choose which scale matters per dataset.
    """

    def __init__(self, d_emb: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
        )

    @staticmethod
    def compute_overlap(
        nb_u: torch.Tensor,      # [P, K] long — u's history neighbors
        nb_v: torch.Tensor,      # [P, K] long — v's history neighbors
        vc_u: torch.Tensor,      # [P] long — u's valid count
        vc_v: torch.Tensor,      # [P] long — v's valid count
    ) -> torch.Tensor:
        """Returns [P, 3]: raw overlap, /min(vc_u,vc_v), /sqrt(vc_u·vc_v)."""
        device = nb_u.device
        K = nb_u.shape[1]
        positions = torch.arange(K, device=device).unsqueeze(0)
        mask_u = positions < vc_u.unsqueeze(1)                    # [P, K]
        mask_v = positions < vc_v.unsqueeze(1)
        # Pairwise compare every u-slot with every v-slot
        match = (nb_u.unsqueeze(2) == nb_v.unsqueeze(1))          # [P, K_u, K_v]
        match = match & mask_u.unsqueeze(2) & mask_v.unsqueeze(1)
        # Count u-slots that have ≥1 v-match (approximation to |set ∩ set|;
        # equal to it when u-history has unique neighbors). This avoids the
        # cost of explicit set deduplication.
        overlap = match.any(dim=2).float().sum(dim=1)             # [P]
        vc_u_f = vc_u.float().clamp_min(1.0)
        vc_v_f = vc_v.float().clamp_min(1.0)
        denom_min = torch.minimum(vc_u_f, vc_v_f)
        denom_geo = torch.sqrt(vc_u_f * vc_v_f)
        return torch.stack([
            overlap,
            overlap / denom_min,
            overlap / denom_geo,
        ], dim=-1)                                                # [P, 3]

    def forward(
        self,
        nb_u: torch.Tensor,
        nb_v: torch.Tensor,
        vc_u: torch.Tensor,
        vc_v: torch.Tensor,
    ) -> torch.Tensor:
        feats = self.compute_overlap(nb_u, nb_v, vc_u, vc_v)      # [P, 3]
        return self.proj(feats)                                    # [P, d_emb]


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


class NodeEncoder(nn.Module):
    """DyGFormer-style dynamic node encoder.

    For each (node u, query time t), construct K_history tokens from u's
    most-recent interactions, run them through a small transformer encoder,
    and pool to a single d-dim node representation.

    Per-token features (concat → projection → d_model):
        neighbor identity  : E_target[neighbor_i]                 d_emb
        recency embedding  : time_embed(t_query − timestamp_i)    d_time
        edge feature       : proj_e(edge_feat_i)                  d_emb  (if ef)
        role               : role_embed(0 if u was src else 1)    d_role

    Cold-start handling: nodes with valid_cnt == 0 produce all-padding
    tokens; the transformer's key_padding_mask is all-True (mask out
    everything), which would NaN softmax → caller should detect cold
    rows and substitute a fallback (e.g. target(u)) before passing to
    the link MLP. The encoder itself just returns whatever PyTorch
    produces on the cold rows (typically zeros after our 0-out-cold
    safety post-step).

    `d_model` must equal `embedding_store.d_emb` so the encoder's output
    is in the same space as the rest of the system (link MLP, alignment).

    Shares `embedding_store.edge_feat_proj` (the same Linear used by the
    walk encoder + context_walk) — no duplicated parameter for edge feats.
    """

    def __init__(
        self,
        embedding_store: "EmbeddingStore",
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        d_time: int = 16,
        d_role: int = 8,
        dropout: float = 0.1,
        ff_dim: int = 256,
    ):
        super().__init__()
        d_emb = embedding_store.d_emb
        if d_model != d_emb:
            raise ValueError(
                f"NodeEncoder d_model ({d_model}) must equal d_emb ({d_emb}); "
                f"the link MLP / alignment / cross-pair all live in d_emb space."
            )
        # Don't register embedding_store as a submodule — avoids double-
        # counting in .parameters() (same trick as WalkEncoder).
        object.__setattr__(self, "embedding_store", embedding_store)
        self.d_emb = d_emb
        self.d_model = d_model
        self.has_edge_feat = embedding_store.has_edge_feat

        # Per-token feature heads.
        self.role_embed = nn.Embedding(2, d_role)
        self.mlp_time = nn.Sequential(
            nn.Linear(2, d_time),
            nn.GELU(),
            nn.Linear(d_time, d_time),
        )
        # Token concat width → d_model projection
        token_dim = d_emb + d_time + d_role + (d_emb if self.has_edge_feat else 0)
        self.token_proj = nn.Linear(token_dim, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable for short sequences
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        neighbors: torch.Tensor,       # [P, K] long, padding=-1 (clamp before lookup)
        timestamps: torch.Tensor,      # [P, K] long, padding=-1
        edge_feats: Optional[torch.Tensor],  # [P, K, d_e] float or None
        roles: torch.Tensor,           # [P, K] long {0, 1}, padding=-1
        valid_cnt: torch.Tensor,       # [P] long, in [0, K]
        t_query: torch.Tensor,         # [P] long — query times per row
        time_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (node_h, has_history):
            node_h:      [P, d_model] — encoded representation
            has_history: [P] bool     — True if the row had at least one
                                        real history entry (caller substitutes
                                        a fallback for the False ones)
        """
        device = neighbors.device
        P, K = neighbors.shape

        # Build valid mask from valid_cnt: position p of row i is valid iff p < vc[i].
        # Convention matches NodeHistory: right-padded, valid at [0, vc-1].
        positions = torch.arange(K, device=device).unsqueeze(0)        # [1, K]
        valid_mask = positions < valid_cnt.unsqueeze(1)                 # [P, K]
        has_history = valid_mask.any(dim=1)                             # [P]

        # Safe lookup tensors: clamp padding indices to 0 so embedding lookups
        # don't blow up. The mask zeros out their contribution anyway.
        nb_safe = neighbors.clamp_min(0)
        ro_safe = roles.clamp_min(0)

        # Neighbor identity via E_target (the persistent identity table —
        # NOT target() through target_final, because we want the raw lookup
        # for tokens; the per-feature-fusion is handled inside this encoder
        # via the token_proj).
        nb_emb = self.embedding_store.E_target(nb_safe)                 # [P, K, d_emb]

        # Recency embedding. t_query − timestamps[i]; clamp_min(0) defensively.
        # Replace padding -1 timestamps with t_query (Δt = 0) so they don't
        # blow up the time MLP — they're masked out downstream anyway.
        t_q = t_query.unsqueeze(1)                                       # [P, 1]
        ts_safe = torch.where(valid_mask, timestamps, t_q)
        dt = (t_q - ts_safe).clamp_min(0).float()
        dt_norm = dt / max(time_scale, 1e-6)
        time_in = torch.stack([dt_norm, torch.log1p(dt_norm)], dim=-1)
        time_h = self.mlp_time(time_in)                                  # [P, K, d_time]

        # Role embedding
        role_h = self.role_embed(ro_safe)                                # [P, K, d_role]

        parts = [nb_emb, time_h, role_h]
        if self.has_edge_feat and edge_feats is not None:
            ef_proj = self.embedding_store.edge_feat_proj(edge_feats.float())
            # Zero out padding slots' edge feats (mask out their contribution
            # since the model shouldn't see anything from padding rows).
            ef_proj = ef_proj * valid_mask.unsqueeze(-1).float()
            parts.append(ef_proj)

        tokens = torch.cat(parts, dim=-1)                                # [P, K, token_dim]
        tokens = self.token_proj(tokens)                                  # [P, K, d_model]

        # Transformer expects key_padding_mask: True = MASK (ignore).
        kpm = ~valid_mask                                                 # [P, K] True at padding

        # Cold-start rows (all padding): nn.TransformerEncoder NaNs on a
        # fully-masked sequence. Force one slot to be unmasked so softmax
        # has a valid key; we'll override the OUTPUT for those rows.
        all_padding = kpm.all(dim=1)                                     # [P]
        kpm_safe = kpm.clone()
        kpm_safe[all_padding, 0] = False

        enc_out = self.transformer(tokens, src_key_padding_mask=kpm_safe)
        enc_out = self.out_norm(enc_out)                                 # [P, K, d_model]

        # Pool: masked mean over the truly-valid positions
        node_h = masked_mean_pool(enc_out, valid_mask)                   # [P, d_model]

        # Zero out the cold rows (caller will substitute fallback)
        node_h = node_h * has_history.float().unsqueeze(-1)

        return node_h, has_history


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
