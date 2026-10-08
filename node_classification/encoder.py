"""The frozen Run-1 encoder: trained E + pooler from a link-prediction checkpoint.

For an interaction (u, t) it returns two geometric features of u's walk bag, from u's
backward walks with the EXCLUSIVE cutoff t, so only edges strictly before t are seen:

    d0(p_u)                  radius of the frozen pooler's point p_u
    sum_i w_i d(x_i, p_u)    pooling-weighted spread of the bag's token points around p_u

Freezing happens once, in `from_checkpoint`: eval(), requires_grad_(False), and `encode`
runs under no_grad. Nothing here is ever handed to an optimiser.

Replay: Tempest's RNG advances with every walk call and has no per-call seed, so each
`encode` pass builds a FRESH walker with the same seed and re-ingests the graph (~0.2 s on
tgbl-wiki). Every pass over the same split therefore draws identical walks.
"""
import hashlib
from typing import Iterator

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
        self.n_feat = 2

    @classmethod
    def from_checkpoint(cls, path: str, graph: SplitData, *, device: torch.device,
                        use_gpu_tempest: bool, seed: int) -> "FrozenEncoder":
        """Rebuild the Run-1 model from the checkpoint's own args and load its weights.
        `graph` is the full stream (all splits) the walks run over."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        args, state = ckpt["args"], ckpt["state_dict"]
        num_nodes, d_emb = state["E.weight"].shape
        model = LinkPredHead(num_nodes=num_nodes, d_emb=d_emb,
                             hidden_dim=args["hidden_dim"],
                             n_layers_pooler=args["n_layers_pooler"], seed=args["seed"])
        model.load_state_dict(state)
        walk_args = {k: args[k] for k in (
            "num_walks_per_node", "max_walk_len", "walk_bias", "start_bias", "t2nv_p", "t2nv_q")}
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
    def encode(self, split: SplitData, batch_size: int) -> Iterator[torch.Tensor]:
        """One replayable pass over `split` in chronological batches -> features [B, 2] per batch."""
        walker = self._fresh_walker()
        a = self.walk_args
        for batch in create_batches(split, batch_size):
            src = torch.from_numpy(batch.src.astype(np.int64))
            ts = torch.from_numpy(batch.ts.astype(np.int64))
            tokens = build_query_walk_tokens(
                walker, self.device, src, ts,
                max_walk_len=a["max_walk_len"], num_walks_per_node=a["num_walks_per_node"],
                start_bias=a["start_bias"], walk_bias=a["walk_bias"])
            yield self._features(tokens)

    def _features(self, tokens: WalkTokens) -> torch.Tensor:
        geom = self.model.geom
        p_u, w, x_tokens = self.model.bag_weights.pool(tokens)               # [Q, d], [Q, T], [Q, T, d]
        radius = geom.dist0(p_u)                                             # [Q]
        spread = (w * geom.dist(x_tokens, p_u.unsqueeze(-2))).sum(dim=-1)    # [Q]; w = 0 on padding
        return torch.stack([radius, spread], dim=-1)                         # [Q, 2]

    def state_hash(self) -> str:
        """Fingerprint of every encoder weight, for the before/after-training check."""
        h = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()
