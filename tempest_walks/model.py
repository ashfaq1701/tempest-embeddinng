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
  - Output is L2-normalised onto the unit sphere via
    F.normalize(.., p=2, dim=-1). The alignment loss is squared L2
    distance, which on the sphere equals 2 - 2*cos.
  - Two instances: P_target (for seed/downstream nodes) and
    P_context (for walk-internal/upstream nodes), each with its own
    parameters. Both heads have the same architecture.
  - Edge features were tested and consistently underperformed the
    no-EF baseline; the EF channel has been removed.

LinkHead
  - score(u, v) = bilinear(E(u), E(v)) + small_MLP(pair_features(u, v))
    bilinear  = (W·E(u)) · E(v) + b   (Linear(d, d)·v inner product
                                        plus scalar bias — same
                                        expressivity as nn.Bilinear
                                        but with a backward graph
                                        that materialises [B, 1+K, d]
                                        instead of [B, 1+K, d, d];
                                        nn.Bilinear OOMs at our batch
                                        × K_train when E is not
                                        detached on the link path)
    MLP input = 6-channel pair features
                [E(u), E(v), E(u)*E(v), |E(u)-E(v)|,
                 (E(u)-E(v))^2, E(u)+E(v)]
  - Per-query batched: inputs are [B, 1+K, d_emb], output is
    [B, 1+K] logits. Column 0 holds the positive candidate at
    training; columns 1..K are negatives sharing the same query
    source.
  - E is NOT detached on the call site — L_link's gradient flows
    back through this head into E. Joint training of E by both
    L_align and L_link.
  - No node features, no edge features, no time features at scoring.
  - Asymmetric by construction. For undirected datasets the caller
    symmetrises by averaging forward(e_u, e_v) with
    forward(e_v, e_u); applied at both training and eval so the
    two paths share the exact same scoring rule.
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
      - Merge MLP mixes back to d_emb.
      - Output is L2-normalised onto the unit sphere via
        F.normalize(.., p=2, dim=-1).

    The alignment loss is squared L2 distance, which on the unit
    sphere equals 2 - 2*cos — a monotone transform of cosine, so
    L2-norm + l2_dist is equivalent to cosine sim up to a constant.

    Projection dim is fixed equal to the embedding dim. The earlier
    d_proj knob was always set equal to d_emb in practice and only
    added an unused degree of freedom; collapsing it removes the
    knob without changing behaviour for any past run.

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
        d_node_feat: Optional[int] = None,
        d_hidden: Optional[int] = None,
    ):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_emb

        self.has_nf = d_node_feat is not None

        self.e_mlp = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_emb),
        )

        if self.has_nf:
            self.nf_mlp = nn.Sequential(
                nn.Linear(d_node_feat, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_emb),
            )

        n_branches = 1 + int(self.has_nf)
        self.merge = nn.Sequential(
            nn.Linear(n_branches * d_emb, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_emb),
        )

    def forward(
        self,
        e: torch.Tensor,
        node_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns L2-normalised projection on the unit sphere."""
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

    Input contract: `e_u` and `e_v` of shape `[B, 1+K, d_emb]` where
    column 0 is the positive candidate and columns 1..K are the K
    negatives, all sharing the same query row's source. Returns
    logits of shape `[B, 1+K]`. nn.Bilinear and nn.Linear broadcast
    over the leading two dims; no flat-input mode.

    E is NOT detached at the call site. L_link's gradient flows back
    through this head into the embedding table — E is jointly trained
    by both L_align (alignment InfoNCE) and L_link (per-query softmax
    CE).

    Asymmetric by construction (bilinear u^T W v + concat channels
    carry directional info). For undirected datasets the caller
    symmetrises by averaging `link_head(e_u, e_v)` with
    `link_head(e_v, e_u)` — both at training and at eval. The two
    orderings read different rows of E so it's a genuine two-view
    average, not a TTA shim.

    No internal regularisation (no dropout, no LayerNorm) —
    dropout-0.1 + pre-MLP LayerNorm A/B on a 50ep tgbl-wiki run
    did not improve over plain.
    """

    def __init__(
        self,
        d_emb: int,
        d_hidden: Optional[int] = None,
    ):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_emb

        # Bilinear branch: u^T W v + b. Refactored from nn.Bilinear to
        # (Linear(d, d, bias=False)(u) * v).sum(-1) + scalar_b so the
        # backward pass materialises [B, 1+K, d] instead of the
        # [B, 1+K, d, d] outer product nn.Bilinear's autograd builds.
        # Same param count (d*d weight + 1 bias), same expressivity.
        # The outer-product form OOMs at B=500, K_train=300, d=128
        # when E is not detached on the link path (~10 GiB).
        self.bilinear_w = nn.Linear(d_emb, d_emb, bias=False)
        self.bilinear_b = nn.Parameter(torch.zeros(1))

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
        """e_u, e_v: [B, 1+K, d_emb]. Returns logits [B, 1+K]."""
        score_bilin = (self.bilinear_w(e_u) * e_v).sum(dim=-1) + self.bilinear_b

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
