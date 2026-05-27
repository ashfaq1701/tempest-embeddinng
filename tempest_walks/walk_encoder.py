"""Edge-centric walk encoder for the link-prediction head.

Converts a batch of Tempest backward walks (K per seed) into a per-
seed representation `h_seed` of dim d_emb that the link head consumes
in place of E[seed] for BCE scoring.

Architecture:
  1. Per-edge embedding: each of the L-1 walk edges contributes
     (src_emb, tgt_emb, edge_feat, time2vec(Δt), hop_emb) → MLP_edge.
  2. GRU over the L-1 edges chronologically (oldest first). Final
     hidden state per walk = the walk representation.
  3. Mean-pool the K walks per seed.
  4. Concatenate with E[seed] and pass through MLP_seed → h_seed.

Strict separation of concerns (no coupling to InfoNCE):
  - InfoNCE alignment_loss is UNCHANGED — operates on E directly via
    p_target / p_context. The encoder is not a participant.
  - BCE link-prediction loss is the encoder's ONLY supervisor:
        BCE → link_head → encoder MLPs/GRU/time2vec/hop_emb/MLP_seed.
  - Encoder reads E in its forward, but ALL E lookups inside this
    module are .detach() so BCE does NOT reach E via the encoder.
  - E is trained only by InfoNCE (context-side AND seed-side, both
    via the projection heads — unchanged from baseline).

Conceptually: E carries the contrastively-shaped semantic geometry;
the encoder reads E + walk structure and produces a link-predictive
representation supervised by the BCE outcome.

h_seed output dim is fixed to d_emb so the existing LinkHead accepts
it without reshape.
"""

from typing import Optional

import torch
import torch.nn as nn

from .walks import WalkData


class Time2Vec(nn.Module):
    """Time2Vec (Kazemi et al. 2019). Scalar Δt → R^{d_te}.

    First component linear (w0·Δt + b0); the remaining d_te-1
    components are sinusoidal (sin(w_k·Δt + b_k)).
    """

    def __init__(self, d_te: int):
        super().__init__()
        assert d_te >= 1
        self.w = nn.Parameter(torch.randn(d_te))
        self.b = nn.Parameter(torch.randn(d_te))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        v = dt.unsqueeze(-1) * self.w + self.b
        v_lin = v[..., 0:1]
        v_sin = torch.sin(v[..., 1:])
        return torch.cat([v_lin, v_sin], dim=-1)


class WalkEncoder(nn.Module):
    """Edge-centric encoder producing h_seed of shape [N, d_emb] from a
    WalkData of N seeds × K walks each.

    Args:
        embedding_table: shared E (reference, not deep-copied).
        d_emb: node embedding dim. Output h_seed dim is also d_emb.
        d_ef: edge feature dim. 0 if dataset has no edge features.
        d_te, d_he, d_edge, d_walk: encoder hidden dims.
        max_walk_len: L from the walk sampler — sets hop_emb table size.
    """

    def __init__(
        self,
        embedding_table,
        d_emb: int,
        d_ef: int,
        d_te: int = 32,
        d_he: int = 16,
        d_edge: int = 128,
        d_walk: int = 128,
        max_walk_len: int = 20,
    ):
        super().__init__()
        self.E = embedding_table
        self.d_emb = d_emb
        self.d_ef = d_ef
        self.d_te = d_te
        self.d_he = d_he
        self.d_edge = d_edge
        self.d_walk = d_walk
        self.max_walk_len = max_walk_len

        self.time2vec = Time2Vec(d_te)

        # Hop indices range 1..L-1; index 0 unused. Allocate L+1 slots
        # so any clamp-to-(1, L) lookup is in range.
        self.hop_emb = nn.Embedding(max_walk_len + 1, d_he)

        d_edge_in = 2 * d_emb + d_ef + d_te + d_he
        self.mlp_edge = nn.Sequential(
            nn.Linear(d_edge_in, d_edge),
            nn.GELU(),
            nn.Linear(d_edge, d_edge),
        )

        self.gru = nn.GRU(
            input_size=d_edge,
            hidden_size=d_walk,
            num_layers=1,
            batch_first=True,
        )

        # Output dim pinned to d_emb so the link head accepts h_seed
        # in place of E[seed] without re-shaping.
        self.mlp_seed = nn.Sequential(
            nn.Linear(d_emb + d_walk, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
        )

    def forward(
        self,
        walks: WalkData,
        t_now: float,
        T_train: float,
    ) -> torch.Tensor:
        """Returns h_seed: [N, d_emb], row i = encoded rep of walks.seeds[i]."""
        device = self.E.E.weight.device
        nodes = walks.nodes.to(device).long()                # [NK, L]
        timestamps = walks.timestamps.to(device).long()      # [NK, L]
        edge_feats = (
            walks.edge_feats.to(device).float()
            if walks.edge_feats is not None
            else None
        )                                                    # [NK, L-1, d_ef] or None
        lens = walks.lens.to(device).long()                  # [NK]
        seeds = walks.seeds.to(device).long()                # [N]
        K = walks.K
        NK, L = nodes.shape
        N = NK // K

        # Per-edge endpoints (padding -1 clamped; padding rows are
        # masked downstream via packing on edges_per_walk).
        nodes_safe = nodes.clamp_min(0)
        src_ids = nodes_safe[:, :-1]                         # [NK, L-1]
        tgt_ids = nodes_safe[:, 1:]                          # [NK, L-1]

        # E lookups detached. BCE gradient stops here; E is trained
        # exclusively by InfoNCE in losses.py.
        src_embs = self.E(src_ids).detach()                  # [NK, L-1, d_emb]
        tgt_embs = self.E(tgt_ids).detach()                  # [NK, L-1, d_emb]

        # Per-edge Δt. timestamps[:, p] for p in [0, lens-2] is the
        # edge (nodes[p], nodes[p+1]) timestamp; position lens-1 is
        # the INT64_MAX seed sentinel (sliced out by [:, :-1]). Padding
        # positions (-1) yield a large dt; masked via packing below.
        delta_t = (float(t_now) - timestamps[:, :-1].float()).clamp_min(0.0)
        delta_t_norm = delta_t / max(T_train, 1.0)
        time_enc = self.time2vec(delta_t_norm)               # [NK, L-1, d_te]

        # Hop from seed: edge at position p has hop = lens-1 - p.
        # hop=1 → edge directly into seed; hop=L-1 → oldest edge.
        edge_positions = torch.arange(L - 1, device=device).unsqueeze(0)  # [1, L-1]
        hop_per_edge = (lens.unsqueeze(1) - 1 - edge_positions).clamp(
            min=1, max=self.max_walk_len,
        )
        hop_enc = self.hop_emb(hop_per_edge)                 # [NK, L-1, d_he]

        parts = [src_embs, tgt_embs]
        if self.d_ef > 0:
            if edge_feats is not None:
                parts.append(edge_feats)
            else:
                # Encoder was configured for edge features but the
                # dataset didn't supply any — substitute zeros so the
                # MLP_edge input shape stays stable.
                parts.append(torch.zeros(
                    NK, L - 1, self.d_ef, device=device,
                ))
        parts.extend([time_enc, hop_enc])
        edge_input = torch.cat(parts, dim=-1)                # [NK, L-1, d_edge_in]
        edge_repr = self.mlp_edge(edge_input)                # [NK, L-1, d_edge]

        # GRU over real edges (mask padding via packing on edges_per_walk).
        edges_per_walk = (lens - 1).clamp_min(0)
        nonempty = edges_per_walk > 0

        h_walk = torch.zeros(NK, self.d_walk, device=device)
        if bool(nonempty.any()):
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                edge_repr[nonempty],
                lengths=edges_per_walk[nonempty].cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h_final = self.gru(packed)                    # [1, NK_nonempty, d_walk]
            h_walk[nonempty] = h_final[-1]

        walk_aggregate = h_walk.view(N, K, self.d_walk).mean(dim=1)  # [N, d_walk]

        e_seed = self.E(seeds).detach()                      # [N, d_emb]

        h_seed = self.mlp_seed(
            torch.cat([e_seed, walk_aggregate], dim=-1),
        )                                                    # [N, d_emb]
        return h_seed


class AttentionWalkEncoder(nn.Module):
    """Self-attention variant of WalkEncoder. Replaces the GRU with a
    multi-head self-attention block over the L-1 edge tokens of each
    walk. Mean-pool over K walks; concat with E[seed] and MLP_seed —
    same output shape as WalkEncoder.

    Args mirror WalkEncoder; adds:
        n_heads: attention heads (default 4).
        n_layers: transformer-encoder layers (default 1).
        exclude_seed: if True, the encoder is rendered purely
            neighbourhood-derived — the seed's E[seed] is NOT folded
            into the last edge's tgt_emb (replaced by a learned
            [SEED] marker) and the final MLP_seed does not concat
            E[seed]. h_seed then depends only on walk-edge content.

    Gradient routing: same as WalkEncoder — E lookups detached, BCE
    is the only supervisor.
    """

    def __init__(
        self,
        embedding_table,
        d_emb: int,
        d_ef: int,
        d_te: int = 32,
        d_he: int = 16,
        d_edge: int = 128,
        d_walk: int = 128,
        max_walk_len: int = 20,
        n_heads: int = 4,
        n_layers: int = 1,
        exclude_seed: bool = False,
    ):
        super().__init__()
        # d_walk must equal d_edge for the attention path (the
        # transformer's residual stream needs a single width).
        assert d_walk == d_edge, (
            f"AttentionWalkEncoder requires d_walk == d_edge "
            f"(got {d_walk} vs {d_edge}); pass --d-edge equal to --d-walk."
        )
        self.E = embedding_table
        self.d_emb = d_emb
        self.d_ef = d_ef
        self.d_te = d_te
        self.d_he = d_he
        self.d_edge = d_edge
        self.d_walk = d_walk
        self.max_walk_len = max_walk_len
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.exclude_seed = exclude_seed

        self.time2vec = Time2Vec(d_te)
        self.hop_emb = nn.Embedding(max_walk_len + 1, d_he)

        d_edge_in = 2 * d_emb + d_ef + d_te + d_he
        self.mlp_edge = nn.Sequential(
            nn.Linear(d_edge_in, d_edge),
            nn.GELU(),
            nn.Linear(d_edge, d_edge),
        )

        # nn.TransformerEncoder with norm_first for stable training
        # (pre-LN; standard practice for small datasets).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_edge,
            nhead=n_heads,
            dim_feedforward=4 * d_edge,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if exclude_seed:
            # [SEED] marker replaces tgt_emb at the to-seed edge so the
            # seed's identity is opaque to the encoder. Output MLP gets
            # walk_aggregate only (no E[seed] concat).
            self.seed_marker = nn.Parameter(torch.randn(d_emb))
            self.mlp_seed = nn.Sequential(
                nn.Linear(d_walk, d_emb),
                nn.GELU(),
                nn.Linear(d_emb, d_emb),
            )
        else:
            self.seed_marker = None
            self.mlp_seed = nn.Sequential(
                nn.Linear(d_emb + d_walk, d_emb),
                nn.GELU(),
                nn.Linear(d_emb, d_emb),
            )

    def forward(
        self,
        walks: WalkData,
        t_now: float,
        T_train: float,
        return_tokens: bool = False,
    ):
        """Returns h_seed: [N, d_emb], row i = encoded rep of walks.seeds[i].

        If return_tokens=True, returns (h_seed, tokens, token_mask)
        where tokens is [N, K*(L-1), d_walk] (flattened per-edge
        attended reps across the K walks of each seed) and token_mask
        is [N, K*(L-1)] (True at valid edges, False at within-walk
        padding). Used by cross-attention-style link heads.
        """
        device = self.E.E.weight.device
        nodes = walks.nodes.to(device).long()
        timestamps = walks.timestamps.to(device).long()
        edge_feats = (
            walks.edge_feats.to(device).float()
            if walks.edge_feats is not None
            else None
        )
        lens = walks.lens.to(device).long()
        seeds = walks.seeds.to(device).long()
        K = walks.K
        NK, L = nodes.shape
        N = NK // K

        nodes_safe = nodes.clamp_min(0)
        src_ids = nodes_safe[:, :-1]                         # [NK, L-1]
        tgt_ids = nodes_safe[:, 1:]                          # [NK, L-1]

        src_embs = self.E(src_ids).detach()                  # [NK, L-1, d_emb]
        tgt_embs = self.E(tgt_ids).detach()                  # [NK, L-1, d_emb]

        # Exclude_seed: replace E[seed] EVERYWHERE it appears in the
        # walk (last-edge tgt by contract, but also any interior cycle
        # back to the seed) with a learned [SEED] marker. The encoder
        # then cannot see the seed's E[seed] value through ANY walk
        # position — its identity is conveyed only by the fact that
        # walks were sampled FROM the seed (purely neighbourhood-
        # derived h).
        edges_per_walk = (lens - 1).clamp_min(0)             # [NK]
        if self.exclude_seed:
            seeds_per_row = seeds.repeat_interleave(K)       # [NK]
            seed_mask_src = src_ids == seeds_per_row.unsqueeze(1)  # [NK, L-1]
            seed_mask_tgt = tgt_ids == seeds_per_row.unsqueeze(1)  # [NK, L-1]
            seed_marker_b = self.seed_marker.view(1, 1, -1).expand(NK, L - 1, -1)
            src_embs = torch.where(
                seed_mask_src.unsqueeze(-1), seed_marker_b, src_embs,
            )
            tgt_embs = torch.where(
                seed_mask_tgt.unsqueeze(-1), seed_marker_b, tgt_embs,
            )

        # Per-edge Δt / hop / edge-feat assembly (unchanged from GRU encoder).
        delta_t = (float(t_now) - timestamps[:, :-1].float()).clamp_min(0.0)
        delta_t_norm = delta_t / max(T_train, 1.0)
        time_enc = self.time2vec(delta_t_norm)               # [NK, L-1, d_te]

        edge_positions = torch.arange(L - 1, device=device).unsqueeze(0)
        hop_per_edge = (lens.unsqueeze(1) - 1 - edge_positions).clamp(
            min=1, max=self.max_walk_len,
        )
        hop_enc = self.hop_emb(hop_per_edge)                 # [NK, L-1, d_he]

        parts = [src_embs, tgt_embs]
        if self.d_ef > 0:
            if edge_feats is not None:
                parts.append(edge_feats)
            else:
                parts.append(torch.zeros(NK, L - 1, self.d_ef, device=device))
        parts.extend([time_enc, hop_enc])
        edge_input = torch.cat(parts, dim=-1)
        edge_repr = self.mlp_edge(edge_input)                # [NK, L-1, d_edge]

        # Padding mask for attention: True at positions to IGNORE.
        # Position p is valid iff p < edges_per_walk.
        positions = torch.arange(L - 1, device=device).unsqueeze(0)      # [1, L-1]
        key_padding_mask = positions >= edges_per_walk.unsqueeze(1)      # [NK, L-1]
        # Rows with zero edges have an all-True mask, which would make
        # softmax produce NaN. Run transformer only on nonempty rows.
        nonempty = edges_per_walk > 0
        attn_out = torch.zeros_like(edge_repr)
        if bool(nonempty.any()):
            ne_repr = edge_repr[nonempty]
            ne_mask = key_padding_mask[nonempty]
            attn_out[nonempty] = self.transformer(
                ne_repr, src_key_padding_mask=ne_mask,
            )

        # Per-walk h_walk = the LAST valid edge's attended representation.
        # Mirrors the GRU's "final hidden state after the most recent
        # edge into the seed" semantics.
        h_walk = torch.zeros(NK, self.d_walk, device=device)
        if bool(nonempty.any()):
            ne_idx = nonempty.nonzero(as_tuple=True)[0]
            last_idx = edges_per_walk[ne_idx] - 1                        # [NK_ne]
            h_walk[ne_idx] = attn_out[ne_idx, last_idx]                  # [NK_ne, d_walk]

        walk_aggregate = h_walk.view(N, K, self.d_walk).mean(dim=1)      # [N, d_walk]

        if self.exclude_seed:
            h_seed = self.mlp_seed(walk_aggregate)
        else:
            e_seed = self.E(seeds).detach()
            h_seed = self.mlp_seed(torch.cat([e_seed, walk_aggregate], dim=-1))

        if not return_tokens:
            return h_seed

        # Token bank for cross-attention link heads. Tokens are the
        # attended per-edge representations flattened across the K
        # walks of each seed. token_mask is True at valid edges only,
        # so downstream attention can mask within-walk padding.
        d = attn_out.shape[-1]
        tokens = attn_out.view(N, K, L - 1, d).reshape(N, K * (L - 1), d)
        positions_tok = torch.arange(L - 1, device=device).view(1, 1, -1)
        edges_per_walk_2d = edges_per_walk.view(N, K, 1)
        per_walk_valid = positions_tok < edges_per_walk_2d
        token_mask = per_walk_valid.reshape(N, K * (L - 1))
        return h_seed, tokens, token_mask


def lookup_h_seed(
    h_seed: torch.Tensor,
    seeds_sorted: torch.Tensor,
    node_ids: torch.Tensor,
) -> torch.Tensor:
    """Row-index h_seed by node id. seeds_sorted must be ascending
    (guaranteed when produced by np.unique). Caller is responsible for
    ensuring every node_id is present in seeds_sorted."""
    rows = torch.searchsorted(seeds_sorted, node_ids)
    return h_seed[rows]
