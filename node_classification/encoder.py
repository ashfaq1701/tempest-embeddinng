"""The frozen Run-1 encoder: trained E + pooler from a link-prediction checkpoint.

For an interaction (u, t) it returns 6 numbers, every one causal (EXCLUSIVE cutoff t, so only
edges strictly before t are seen). p_u is the frozen pooler's point for u's walk bag, x_i the
bag's token points, d the hyperbolic distance, d0 the radius.

  geometry (4)
    d0(p_u)                       radius of the pooled point
    d0(E[u])                      radius of u's own trained point
    sum_i w_i d(x_i, p_u)         spread of the bag around p_u under the pooler's weights w,
    -sum_i w_i log w_i              and the entropy of those weights (padding has w = 0)
  Tempest history (2), from Tempest's per-node event index
    log1p(#u's prior edges)       get_node_popularity
    log1p(t - u's last edge time) get_node_recency; no prior edge maps to
                                  log1p(time span of the whole graph), older than any real gap

These six were selected from a 22-column candidate set by group-drop then candidate pruning
on chronological CV over train ∪ val (reports/node_classification_2026-10-08.md); the spread
and entropy were later switched from a separate softmax(-d) attention to the pooler's weights.

Freezing happens once, in `from_checkpoint`: eval(), requires_grad_(False), and `encode`
runs under no_grad. Nothing here is ever handed to an optimiser.

Replay: Tempest's RNG advances with every walk call and has no per-call seed, so each
`encode` pass builds a FRESH walker with the same seed and re-ingests the graph. Every pass
over the same split therefore draws identical walks, so one pass is exactly what every
training epoch would see.
"""
import hashlib

import numpy as np
import torch

from link_property_prediction.data import SplitData, create_batches
from link_property_prediction.model import LinkPredHead
from link_property_prediction.walk_tokens import WalkTokens, build_query_walk_tokens
from link_property_prediction.walks import WalkGenerator


class FrozenEncoder:

    def __init__(self, model: LinkPredHead, walk_args: dict, graph: SplitData,
                 device: torch.device, use_gpu_tempest: bool, seed: int):
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.walk_args = walk_args
        self.graph = graph
        self.device = device
        self.use_gpu_tempest = bool(use_gpu_tempest)
        self.seed = int(seed)
        span = int(graph.timestamps.max()) - int(graph.timestamps.min()) + 1
        self.no_event_log_gap = float(np.log1p(span))

    @classmethod
    def from_checkpoint(cls, path: str, graph: SplitData, *, device: torch.device,
                        use_gpu_tempest: bool, seed: int,
                        num_walks_per_node: int = 0) -> "FrozenEncoder":
        """Rebuild the Run-1 model from the checkpoint's own args and load its weights.
        `graph` is the full stream (all splits) the walks run over. `num_walks_per_node`
        overrides the checkpoint's walk count (0 keeps it)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        args, state = ckpt["args"], ckpt["state_dict"]
        num_nodes, d_emb = state["E.weight"].shape
        model = LinkPredHead(num_nodes=num_nodes, d_emb=d_emb,
                             hidden_dim=args["hidden_dim"],
                             n_layers_pooler=args["n_layers_pooler"], seed=args["seed"])
        model.load_state_dict(state)
        walk_args = {k: args[k] for k in (
            "num_walks_per_node", "max_walk_len", "walk_bias", "start_bias", "t2nv_p", "t2nv_q")}
        if num_walks_per_node:
            walk_args["num_walks_per_node"] = int(num_walks_per_node)
        return cls(model, walk_args, graph, device, use_gpu_tempest, seed)

    def _fresh_walker(self) -> WalkGenerator:
        """Same construction as Trainer's walker, seeded, with the whole graph ingested."""
        a = self.walk_args
        walker = WalkGenerator(
            use_gpu=self.use_gpu_tempest,
            walk_bias=a["walk_bias"],
            start_bias=a["start_bias"],
            num_walks_per_node=a["num_walks_per_node"],
            max_walk_len=a["max_walk_len"],
            temporal_node2vec_p=a["t2nv_p"],
            temporal_node2vec_q=a["t2nv_q"],
            seed=self.seed,
        )
        g = self.graph
        walker.add_edges(g.sources, g.destinations, g.timestamps, g.edge_feat)
        return walker

    @torch.no_grad()
    def encode(self, split: SplitData, batch_size: int = 200) -> torch.Tensor:
        """One replayable pass over `split` in chronological batches -> features [N, 6]."""
        walker = self._fresh_walker()
        a = self.walk_args
        out = []
        for batch in create_batches(split, batch_size):
            src = torch.from_numpy(batch.src.astype(np.int64))
            ts = torch.from_numpy(batch.ts.astype(np.int64))
            tokens = build_query_walk_tokens(
                walker, self.device, src, ts,
                max_walk_len=a["max_walk_len"], num_walks_per_node=a["num_walks_per_node"],
                start_bias=a["start_bias"], walk_bias=a["walk_bias"])
            geometry = self._geometry(tokens, src.to(self.device))
            history = self._history(walker, batch.src, batch.ts)
            out.append(torch.cat([geometry, history.to(self.device)], dim=1))
        return torch.cat(out)

    def _geometry(self, tokens: WalkTokens, src: torch.Tensor) -> torch.Tensor:
        """The 4 geometric columns; see the module docstring."""
        geom = self.model.geom
        p_u, w, x = self.model.bag_weights.pool(tokens)                      # [Q,d], [Q,T], [Q,T,d]
        e_u = self.model.E.weight[src]                                       # [Q, d]

        d_tok = geom.dist(x, p_u.unsqueeze(-2))                              # [Q, T]
        geometry = [
            geom.dist0(p_u),
            geom.dist0(e_u),
            (w * d_tok).sum(-1),
            -(w * w.clamp_min(1e-12).log()).sum(-1),
        ]
        return torch.stack(geometry, dim=-1)                                 # [Q, 4]

    def _history(self, walker: WalkGenerator, src: np.ndarray, ts: np.ndarray) -> torch.Tensor:
        """The 2 Tempest history columns for queries (u, t); see the module docstring."""
        src = src.astype(np.int64)
        ts = ts.astype(np.int64)
        out = np.zeros((len(src), 2), dtype=np.float32)
        count = walker.get_node_popularity(src, ts)
        recency = walker.get_node_recency(src, ts)        # t - t_last; meaningless if count == 0
        out[:, 0] = np.log1p(count)
        out[:, 1] = np.where(count > 0, np.log1p(recency), self.no_event_log_gap)
        return torch.from_numpy(out)

    def state_hash(self) -> str:
        """Fingerprint of every encoder weight, for the before/after-training check."""
        h = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()
