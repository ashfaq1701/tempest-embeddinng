"""Model components: embedding table, projection heads, link head.

Three classes, no shared state:

EmbeddingTable
  - Single nn.Embedding(num_nodes, d_emb).
  - Lookup-only. Trained by alignment+uniformity through the
    projection heads.

ProjectionHead
  - Conditional architecture based on feature availability:
      E only          → MLP on E
      E + NF          → MLP on E + MLP on NF, concat, merge MLP
      E + EF          → MLP on E + MLP on EF, concat, merge MLP
      E + NF + EF     → MLP on E + MLP on NF + MLP on EF, concat,
                        merge MLP
  - Output is L2-normalised via F.normalize(..., p=2, dim=-1, eps=1e-12).
  - Two instances: P_target (for seed/downstream nodes) and
    P_context (for walk-internal/upstream nodes). EF channel only
    appears in P_context per convention β.

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
    """Loss-side projection with conditional NF / EF channels.

    Architecture:
      - One sub-MLP per active input channel (E always; NF if
        d_node_feat is not None; EF if d_edge_feat is not None).
      - Concat the active sub-MLP outputs along the last dim.
      - Merge MLP mixes back to d_proj.
      - Output is L2-normalised on the unit sphere.

    Two instances are typically constructed:
      P_target  — for seed/downstream nodes. EF channel disabled.
      P_context — for walk-internal/upstream nodes. EF channel
                  enabled iff the dataset has edge features and the
                  caller wants to consume them (convention β:
                  ef[p] = edge OUT of position p toward the seed).
    """

    def __init__(
        self,
        d_emb: int,
        d_proj: int,
        d_node_feat: Optional[int] = None,
        d_edge_feat: Optional[int] = None,
        d_hidden: Optional[int] = None,
    ):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_proj

        self.has_nf = d_node_feat is not None
        self.has_ef = d_edge_feat is not None

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

        if self.has_ef:
            self.ef_mlp = nn.Sequential(
                nn.Linear(d_edge_feat, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_proj),
            )

        n_branches = 1 + int(self.has_nf) + int(self.has_ef)
        self.merge = nn.Sequential(
            nn.Linear(n_branches * d_proj, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
        )

    def forward(
        self,
        e: torch.Tensor,
        node_feat: Optional[torch.Tensor] = None,
        edge_feat: Optional[torch.Tensor] = None,
        bypass_ef: bool = False,
    ) -> torch.Tensor:
        """Returns L2-normalised projection, shape e.shape[:-1] + [d_proj].

        bypass_ef=True with has_ef=True: skip the EF branch, inject
          zeros of shape [..., d_proj] at the EF slot of the merge
          concat. The merge MLP's EF-slot weights multiply zero,
          contributing nothing. Used by uniformity_loss which has no
          edge to feed (Task 12 Option γ).
        bypass_ef=True with has_ef=False: no-op (the head has no EF
          channel to bypass anyway).
        """
        if self.has_nf and node_feat is None:
            raise ValueError("ProjectionHead has NF channel but no NF passed")
        if not self.has_nf and node_feat is not None:
            raise ValueError("ProjectionHead has no NF channel but NF was passed")
        # Bypass overrides the "EF must be provided" requirement.
        if self.has_ef and edge_feat is None and not bypass_ef:
            raise ValueError(
                "ProjectionHead has EF channel but no EF passed "
                "(and bypass_ef=False)"
            )
        if not self.has_ef and edge_feat is not None:
            raise ValueError("ProjectionHead has no EF channel but EF was passed")

        branches = [self.e_mlp(e)]
        if self.has_nf:
            branches.append(self.nf_mlp(node_feat))
        if self.has_ef:
            if bypass_ef:
                # Option γ: hard-zero contribution at the merge concat
                # slot. ef_mlp is NOT called, so its bias can't leak
                # a constant DC offset (which was Option α's failure
                # mode in Task 12 C3).
                ef_branch = torch.zeros_like(branches[0])
            else:
                ef_branch = self.ef_mlp(edge_feat)
            branches.append(ef_branch)

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
