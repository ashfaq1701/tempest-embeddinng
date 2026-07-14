"""Lightweight training-time probes on the learned embedding E.

CommunityProbe — per-epoch community-formation measure: cosine-kNN label purity of E against the
graph's Louvain communities, vs the random-neighbour null. Labels + null + anchor sample are
computed ONCE (the graph is fixed); per epoch only a sampled cosine-kNN + label lookup runs, so it
is cheap enough to call every epoch.
"""
import numpy as np
import torch
import torch.nn.functional as F


class CommunityProbe:
    """Per-epoch cosine-kNN label purity of E against the graph's Louvain communities.

        purity = mean fraction of each anchor's k nearest neighbours (cosine) sharing its community.
        null   = random-neighbour same-community probability Σ_c (n_c/N)²  (closed form, no permutation).
        lift   = purity / null   (>1 ⇒ E clusters by community; ≈1 ⇒ diffuse, no community in E).

    Rank-based + null-normalised, so it is robust to the high-dim distance concentration that makes a
    plain mean intra/inter-community distance ratio read ≈1 even when structure exists.
    """

    def __init__(self, src, dst, num_nodes, n_anchors=1000, k=10, seed=0):
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(range(num_nodes))
        g.add_edges_from(zip(np.asarray(src).tolist(), np.asarray(dst).tolist()))
        comms = nx.community.louvain_communities(g, seed=seed)

        self.label = np.full(num_nodes, -1, dtype=np.int64)
        for cid, nodes in enumerate(comms):
            for n in nodes:
                self.label[n] = cid

        counts = np.unique(self.label[self.label >= 0], return_counts=True)[1]
        p = counts / counts.sum()
        self.null = float((p ** 2).sum())               # random-neighbour same-community prob
        self.q = float(nx.community.modularity(g, comms))
        self.n_comms = len(comms)

        rng = np.random.default_rng(seed)
        cand = np.where(self.label >= 0)[0]
        anchors = rng.choice(cand, size=min(n_anchors, len(cand)), replace=False)
        self.anchors = torch.as_tensor(anchors, dtype=torch.long)
        self.label_t = torch.as_tensor(self.label)
        self.k = k

    @torch.no_grad()
    def measure(self, e: torch.Tensor) -> float:
        """e [N, d] embeddings on any device. Returns the cosine-kNN community purity ∈ [0, 1]."""
        a = self.anchors.to(e.device)
        lab = self.label_t.to(e.device)
        sim = F.normalize(e[a], dim=-1) @ F.normalize(e, dim=-1).T     # [n_anchor, N] cosine
        sim[torch.arange(a.numel(), device=e.device), a] = -2.0        # exclude the anchor itself
        knn = sim.topk(self.k, dim=-1).indices                         # [n_anchor, k]
        return (lab[knn] == lab[a][:, None]).float().mean().item()
