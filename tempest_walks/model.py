"""Model components: embedding table, projection heads, link head.

Three classes, no shared state:

EmbeddingTable
  - Single nn.Embedding(num_nodes, d_emb).
  - Lookup-only. Trained by InfoNCE contrastive alignment through
    the projection heads.

ProjectionHead
  - Conditional architecture based on node-feature availability:
      E only   → MLP on E
      E + NF   → MLP on E + MLP on NF, concat, merge MLP
  - Output is L2-normalised via F.normalize(..., p=2, dim=-1, eps=1e-12).
  - Two instances: P_target (for seed/downstream nodes) and
    P_context (for walk-internal/upstream nodes), each with its own
    parameters. Both heads have the same architecture.
  - Edge features were tested and consistently underperformed the
    no-EF baseline; the EF channel has been removed.

LinkHead
  - score(u, v) = bilinear(E(u), E(v)) + small_MLP(pair_features(u, v))
    bilinear  = E(u)^T W E(v) + b   (one learnable matrix)
    MLP input = 6-channel pair features
                [E(u), E(v), E(u)*E(v), |E(u)-E(v)|,
                 (E(u)-E(v))^2, E(u)+E(v)]
  - Inputs are raw E lookups. Stop-gradient on E is the CALLER's
    responsibility (call with e_u.detach(), e_v.detach() in
    trainer.py); LinkHead never detaches internally.
  - No node features, no edge features, no time features at scoring.
  - Asymmetric by construction. Undirected eval averages
    forward(u, v) and forward(v, u) at the caller.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingTable(nn.Module):
    """Lookup-only node embedding table."""

    def __init__(self, num_nodes: int, d_emb: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.E = nn.Embedding(num_nodes, d_emb)
        # Small Gaussian init; downstream L2-norm in projections handles scale.
        nn.init.normal_(self.E.weight, mean=0.0, std=0.02)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """node_ids: any shape of long; returns shape + [d_emb]."""
        return self.E(node_ids)


class ProjectionHead(nn.Module):
    """Loss-side projection with optional node-feature channel.

    Architecture:
      - One sub-MLP per active input channel (E always; NF if
        d_node_feat is not None).
      - Concat the active sub-MLP outputs along the last dim.
      - Merge MLP mixes back to d_proj.
      - Output is L2-normalised on the unit sphere.

    Two instances are typically constructed: P_target (for
    seed/downstream nodes) and P_context (for walk-internal/upstream
    nodes), each with its own parameters.

    Edge features were tested and consistently underperformed the
    no-EF baseline (val 0.397 no-EF vs ≤ 0.355 for every EF
    variant). The EF channel has been removed.
    """

    def __init__(
        self,
        d_emb: int,
        d_proj: int,
        d_node_feat: Optional[int] = None,
        d_hidden: Optional[int] = None,
    ):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_proj

        self.has_nf = d_node_feat is not None

        self.e_mlp = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
        )

        if self.has_nf:
            self.nf_mlp = nn.Sequential(
                nn.Linear(d_node_feat, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_proj),
            )

        n_branches = 1 + int(self.has_nf)
        self.merge = nn.Sequential(
            nn.Linear(n_branches * d_proj, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
        )

    def forward(
        self,
        e: torch.Tensor,
        node_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns L2-normalised projection, shape e.shape[:-1] + [d_proj]."""
        if self.has_nf and node_feat is None:
            raise ValueError("ProjectionHead has NF channel but no NF passed")
        if not self.has_nf and node_feat is not None:
            raise ValueError("ProjectionHead has no NF channel but NF was passed")

        branches = [self.e_mlp(e)]
        if self.has_nf:
            branches.append(self.nf_mlp(node_feat))

        z = torch.cat(branches, dim=-1)
        out = self.merge(z)
        return F.normalize(out, p=2, dim=-1, eps=1e-12)


class LinkHead(nn.Module):
    """Link-prediction scorer: bilinear + 6-channel pair MLP.

    Input: raw E[u], E[v]. The caller must pass `e_u.detach()` /
    `e_v.detach()` if stop-grad on E is desired (this is the
    documented contract for trainer.py — see CLAUDE.md). LinkHead
    NEVER detaches its inputs.

    Output: one logit per (u, v) pair. Asymmetric — undirected eval
    must average score(u, v) and score(v, u) at the caller.
    """

    def __init__(self, d_emb: int, d_hidden: Optional[int] = None):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_emb

        self.bilinear = nn.Bilinear(d_emb, d_emb, 1)

        # 6-channel pair features concatenated along last dim → 6*d_emb input.
        self.mlp = nn.Sequential(
            nn.Linear(6 * d_emb, 4 * d_hidden),
            nn.GELU(),
            nn.Linear(4 * d_hidden, 2 * d_hidden),
            nn.GELU(),
            nn.Linear(2 * d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, e_u: torch.Tensor, e_v: torch.Tensor) -> torch.Tensor:
        """Returns logits, shape e_u.shape[:-1] (no trailing dim)."""
        score_bilin = self.bilinear(e_u, e_v).squeeze(-1)

        pair_feats = torch.cat(
            [
                e_u,
                e_v,
                e_u * e_v,
                (e_u - e_v).abs(),
                (e_u - e_v).pow(2),
                e_u + e_v,
            ],
            dim=-1,
        )
        score_mlp = self.mlp(pair_feats).squeeze(-1)

        return score_bilin + score_mlp


class CrossAttentionLinkHead(nn.Module):
    """Cross-attention link head over per-seed walk-token banks.

    For each pair (u, v):
      - h_u (query, shape [1, d]) cross-attends to v's token bank
        (key/value, shape [T_v, d]) → u_reads_v [d].
      - Symmetric: h_v cross-attends to u's token bank → v_reads_u [d].
      - Concat [h_u, h_v, u_reads_v, v_reads_u] → MLP → score.

    The encoder produces (h_seed, tokens, token_mask). The trainer
    gathers per-pair (h_u, h_v) and (u_tokens, v_tokens, masks) via
    node-id indexing, then calls this head.

    Why single-query cross-attention (h_u as one query) rather than
    full token-to-token cross-attention: full T_u×T_v attention scales
    O(T²) in memory per pair, which OOMs on 8 GB GPUs at bs=2000
    (P=2200 pairs × T=95 tokens × 128 floats = ~10 GB just for the
    attention matrices). Single-query keeps it O(T) per pair while
    still answering the load-bearing question "what in v's walks is
    relevant to u (as summarised by h_u)?".

    Detach invariant: this head does not call .detach() internally.
    The caller controls whether h_u/h_v carry gradient (they should
    when the encoder is on; that's the BCE→encoder pathway).
    """

    def __init__(self, d_emb: int, n_heads: int = 4, d_hidden: Optional[int] = None):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_emb
        self.n_heads = n_heads
        self.cross_uv = nn.MultiheadAttention(
            d_emb, num_heads=n_heads, batch_first=True,
        )
        self.cross_vu = nn.MultiheadAttention(
            d_emb, num_heads=n_heads, batch_first=True,
        )
        # Score MLP over [h_u | h_v | u_reads_v | v_reads_u].
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_emb, 4 * d_hidden),
            nn.GELU(),
            nn.Linear(4 * d_hidden, 2 * d_hidden),
            nn.GELU(),
            nn.Linear(2 * d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        h_u: torch.Tensor,            # [P, d]
        h_v: torch.Tensor,            # [P, d]
        u_tokens: torch.Tensor,       # [P, T, d]
        u_mask: torch.Tensor,         # [P, T] bool, True at valid edges
        v_tokens: torch.Tensor,       # [P, T, d]
        v_mask: torch.Tensor,         # [P, T] bool, True at valid edges
    ) -> torch.Tensor:
        """Returns logits, shape [P]."""
        # MultiheadAttention's key_padding_mask: True at positions to IGNORE.
        u_pad = ~u_mask
        v_pad = ~v_mask
        # If a row has NO valid tokens (all padded), key_padding_mask
        # is all-True → softmax produces NaN. Substitute a zero output
        # for those rows and skip them in the attention call.
        u_any = u_mask.any(dim=1)     # [P]
        v_any = v_mask.any(dim=1)

        u_reads_v = torch.zeros_like(h_u)
        if bool(v_any.any()):
            sel = v_any.nonzero(as_tuple=True)[0]
            q = h_u[sel].unsqueeze(1)                          # [P', 1, d]
            kv = v_tokens[sel]                                 # [P', T, d]
            kv_pad = v_pad[sel]
            out, _ = self.cross_uv(q, kv, kv, key_padding_mask=kv_pad)
            u_reads_v[sel] = out.squeeze(1)

        v_reads_u = torch.zeros_like(h_v)
        if bool(u_any.any()):
            sel = u_any.nonzero(as_tuple=True)[0]
            q = h_v[sel].unsqueeze(1)
            kv = u_tokens[sel]
            kv_pad = u_pad[sel]
            out, _ = self.cross_vu(q, kv, kv, key_padding_mask=kv_pad)
            v_reads_u[sel] = out.squeeze(1)

        feats = torch.cat([h_u, h_v, u_reads_v, v_reads_u], dim=-1)
        return self.mlp(feats).squeeze(-1)


class HybridLinkHead(nn.Module):
    """Link-prediction scorer that consumes BOTH E[v] (identity) and
    h_v (walk-derived) for each endpoint. Forms wider 2*d_emb-dim
    inputs by concatenation, then runs the same bilinear + 6-channel
    MLP structure on those wider vectors. Tests whether augmenting
    h_v with E[v] at the link-head input helps (vs replacing E[v]
    with h_v as the StandardLinkHead encoder path does).

    Input contract:
        eh_u, eh_v: [..., 2 * d_emb] each, formed by the caller as
        torch.cat([E[u].detach(), h_u], dim=-1) and the same for v.
        Detach on the E side is the caller's responsibility — preserves
        the "BCE does not reach E" invariant. h_u / h_v carry full
        gradient through the encoder.
    """

    def __init__(self, d_emb: int, d_hidden: Optional[int] = None):
        super().__init__()
        d = 2 * d_emb
        if d_hidden is None:
            d_hidden = d

        # No bilinear here — at d=2*d_emb=256, nn.Bilinear's backward
        # materialises a [batch, 1, d, d] = [22000, 1, 256, 256] = 5.6 GB
        # intermediate that OOMs the 8 GB GPU. The 6-channel pair-MLP
        # below is expressive enough; LinkHead's bilinear was an
        # add-on inductive bias at d=128 where the [B, 1, 128, 128]
        # intermediate is 1.4 GB and still fits.

        self.mlp = nn.Sequential(
            nn.Linear(6 * d, 4 * d_hidden),
            nn.GELU(),
            nn.Linear(4 * d_hidden, 2 * d_hidden),
            nn.GELU(),
            nn.Linear(2 * d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, eh_u: torch.Tensor, eh_v: torch.Tensor) -> torch.Tensor:
        """Returns logits, shape eh_u.shape[:-1]."""
        pair_feats = torch.cat(
            [
                eh_u,
                eh_v,
                eh_u * eh_v,
                (eh_u - eh_v).abs(),
                (eh_u - eh_v).pow(2),
                eh_u + eh_v,
            ],
            dim=-1,
        )
        return self.mlp(pair_feats).squeeze(-1)
