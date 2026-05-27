"""Model components: embedding table, projection heads, link head.

Three classes, no shared state. Trainer instantiates the
EmbeddingTable twice (E_t target-role, E_c context-role) and
wires them per role into the loss and link head.

EmbeddingTable
  - Single nn.Embedding(num_nodes, d_emb).
  - Lookup-only. Used both for E_t and E_c (two independent
    instances on the Trainer). Trained by InfoNCE contrastive
    alignment through the projection heads.

ProjectionHead
  - Conditional architecture based on node-feature availability:
      E only   → MLP on E
      E + NF   → MLP on E + MLP on NF, concat, merge MLP
  - Output is L2-normalised via F.normalize(..., p=2, dim=-1, eps=1e-12).
  - Two instances: P_target (consumes E_t for seeds) and
    P_context (consumes E_c for walk-internal positives + sampled
    negatives), each with its own parameters. Both heads have the
    same architecture.
  - Edge features were tested and consistently underperformed the
    no-EF baseline; the EF channel has been removed.

LinkHead
  - score(u, v) = bilinear(E_c[u], E_t[v]) + small_MLP(pair_features)
    bilinear  = E_c[u]^T W E_t[v] + b   (one learnable matrix)
    MLP input = 6-channel pair features
                [E_c[u], E_t[v], E_c[u]·E_t[v], |E_c[u]-E_t[v]|,
                 (E_c[u]-E_t[v])², E_c[u]+E_t[v]]
  - Caller passes (e_src, e_dst) — i.e. the source-role row first
    (E_c lookup) and the destination-role row second (E_t lookup).
    Stop-gradient on the tables is the caller's responsibility
    (call .detach() in trainer.py); LinkHead never detaches
    internally.
  - No node features, no edge features, no time features at scoring.
  - Asymmetric by construction. On directed datasets the caller
    issues ONE call with (E_c[src], E_t[dst]). On undirected
    datasets the caller symmetrises by averaging that with
    (E_c[dst], E_t[src]) — both at training and at eval. The two
    orderings read different rows of E_t/E_c so this is a real
    two-view average, not a TTA shim.
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

    Input contract: the caller passes (e_src, e_dst) — i.e. the
    source-role embedding first, the destination-role embedding
    second. Under the two-table design, that means E_c[u] then
    E_t[v]. The caller is also responsible for stop-grad: pass
    `.detach()`'d inputs if E_t/E_c should not receive gradient
    from the BCE path. LinkHead never detaches.

    Output: one logit per (src, dst) pair. Asymmetric by
    construction (bilinear u^T W v + [e_u, e_v] concat channels
    carry directional information). For undirected datasets the
    caller symmetrises by averaging score(E_c[u], E_t[v]) and
    score(E_c[v], E_t[u]) — see trainer's _train_step (BCE) and
    _score_pairs (eval), which share the same rule. The two
    orderings read different rows of E_t and E_c, so this is a
    genuine two-view average, not a TTA shim.

    No internal regularisation (no dropout, no LayerNorm). A
    dropout-0.1 + pre-MLP LayerNorm variant was A/B'd on a 50ep
    tgbl-wiki single-table run and did not improve over plain
    (best val 0.4391 vs 0.4933 baseline-with-symmetrize) — those
    layers were suspected to be hurting, so this is the
    no-regularisation head.
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
