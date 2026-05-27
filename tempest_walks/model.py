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
  - Asymmetric by construction AND used asymmetrically: training
    only ever feeds (src-role, dst-role), and TGB's eval ranks
    candidate dsts given fixed src. The caller MUST always pass
    (e_src, e_dst) in that order — do NOT average forward(u, v) and
    forward(v, u) for "undirected" datasets; the reverse direction
    is untrained input territory and adds noise to the ranking.
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

    Output: one logit per (u, v) pair. Asymmetric by construction
    (bilinear u^T W v + [e_u, e_v] concat channels carry directional
    information). For datasets where the underlying graph topology is
    symmetric/bipartite, the caller may symmetrise at eval by
    averaging score(u, v) + score(v, u) — see trainer's _score_pairs
    `if not self.config.is_directed` branch. The averaging acts as a
    correlated-ensemble test-time augmentation and empirically helps
    on bipartite-like data.

    No internal regularisation (no dropout, no LayerNorm). A
    dropout-0.1 + pre-MLP LayerNorm variant was A/B'd on a 50ep
    tgbl-wiki run and did not improve over plain (best val 0.4391
    vs 0.4933 baseline-with-symmetrize) — those layers were
    suspected to be hurting, so this is the no-regularisation
    head.
    """

    def __init__(
        self,
        d_emb: int,
        d_hidden: Optional[int] = None,
    ):
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
