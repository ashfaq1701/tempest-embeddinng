"""Frozen link-prediction encoder -> 5 causal features per interaction (u, t):
d0(p_u), d0(E[u]), mean_i d(x_i, p_u), log1p(#prior edges), log1p(t - last edge time)."""
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
        # Tempest's RNG has no per-call seed: only a fresh walker with the same seed replays
        # the same walks.
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
        return torch.cat(out)                                                # [N, 5]

    def _geometry(self, tokens: WalkTokens, src: torch.Tensor) -> torch.Tensor:
        geom = self.model.geom
        p_u, spread = self.model.bag_weights(tokens)                         # [Q, d], [Q]
        e_u = self.model.E.weight[src]                                       # [Q, d]
        geometry = [
            geom.dist0(p_u),
            geom.dist0(e_u),
            spread,
        ]
        return torch.stack(geometry, dim=-1)                                 # [Q, 3]

    def _history(self, walker: WalkGenerator, src: np.ndarray, ts: np.ndarray) -> torch.Tensor:
        src = src.astype(np.int64)
        ts = ts.astype(np.int64)
        out = np.zeros((len(src), 2), dtype=np.float32)
        count = walker.get_node_popularity(src, ts)
        recency = walker.get_node_recency(src, ts)
        out[:, 0] = np.log1p(count)
        # No prior edge: recency is undefined, use a gap older than any real one.
        out[:, 1] = np.where(count > 0, np.log1p(recency), self.no_event_log_gap)
        return torch.from_numpy(out)

    def state_hash(self) -> str:
        h = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()
