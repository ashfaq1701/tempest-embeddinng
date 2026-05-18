"""Dual-table embedding store + 8-block link MLP."""

import torch
import torch.nn as nn


class EmbeddingStore(nn.Module):
    """Two embedding tables: E_target (node as source) and E_context (node as
    walk-neighbour / target). The alignment loss pulls E_target[seed] toward
    E_context[walk-neighbours]; the link MLP reads both for u and v.
    """

    def __init__(self, n_nodes: int, d_emb: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_emb = d_emb
        self.target = nn.Embedding(n_nodes, d_emb)
        self.context = nn.Embedding(n_nodes, d_emb)
        nn.init.xavier_uniform_(self.target.weight)
        nn.init.xavier_uniform_(self.context.weight)


class LinkPredictor(nn.Module):
    """8-block MLP head:
        input = concat([
          E_t[u], E_t[v], E_t[u]·E_t[v], |E_t[u]−E_t[v]|,
          E_c[u], E_c[v], E_c[u]·E_c[v], |E_c[u]−E_c[v]|,
        ])  ∈ ℝ^{8·d}
    Returns raw logits (no sigmoid — paired with BCE-with-logits).
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
