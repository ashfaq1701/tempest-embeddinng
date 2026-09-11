import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


def euclidean_midpoint(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted arithmetic mean -- the flat-space midpoint (exact Frechet mean of the L2 metric)."""
    return (w.unsqueeze(-1) * x).sum(dim=-2)


class BagWeights(nn.Module):

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden = int(hidden_dim)
        self.n_feat = 3
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, tokens: WalkTokens, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        age = torch.log1p(tokens.ages.clamp_min(0).to(x.dtype)).unsqueeze(-1)
        pos = tokens.positions.unsqueeze(-1).to(x.dtype)
        norm = x.detach().norm(dim=-1).unsqueeze(-1)
        feat = torch.cat([age, pos, norm], dim=-1).to(x.dtype)
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        torch.manual_seed(seed)
        self.bag_weights = BagWeights(hidden_dim)

        # E is unconstrained (flat space imposes no constraint), initialised uniform inside the unit
        # ball: isotropic direction * U^(1/d) radius (uniform by volume). Only sets the starting
        # scale; nothing keeps E inside the ball once training starts.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            direction = F.normalize(torch.randn(self.num_nodes, self.d_emb), dim=-1)
            radius = torch.rand(self.num_nodes, 1).pow(1.0 / self.d_emb)
            self.E.weight.copy_(direction * radius)

        self.geo_temp = nn.Parameter(torch.tensor(1.0))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, emb)
        w = self.bag_weights(tokens, x, valid)
        return euclidean_midpoint(x, w)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)
        p_v = self.pool(cand_tokens, emb)
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)
        dist = (p_u.unsqueeze(1) - p_v).norm(dim=-1)
        return self.geo_temp * (-dist)
