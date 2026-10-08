"""The frozen Run-1 encoder: trained E + pooler from a link-prediction checkpoint.

For an interaction (u, t) it returns 22 numbers, every one causal (EXCLUSIVE cutoff t, so only
edges strictly before t are seen): 17 summarise u's backward walks, 5 are history read from
Tempest's own per-node event index. p_u is the frozen pooler's point, w its pooling
weights, x_i the bag's token points, d the hyperbolic distance, d0 the radius.

  geometry (12)
    d0(p_u)                     radius of the pooled point
    sum_i w_i d(x_i, p_u)       pooling-weighted spread of the bag around p_u
    mean_i d(x_i, p_u)          unweighted spread
    max_i / min_i d(x_i, p_u)   farthest / nearest token
    mean_i d0(x_i)              mean token radius
    d0(E[u])                    radius of u's own point
    d(E[u], p_u)                how far pooling moved u
    H(w), max_i w_i             pooling entropy / peak weight
    spread and entropy under softmax(-d(x_i, p_u)), i.e. attention with p_u as the key
  walk time (5)
    log1p(#real edges), min / mean / w-weighted log1p(age), #distinct nodes in the bag
  Tempest history (5), via get_node_popularity / get_node_recency / get_latest_events
    log1p(#u's prior edges)              log1p(t - u's last edge time)
    log1p(gap between u's last two edges, from a second lookup with cutoff t_last)
    log1p(#prior edges of u's last partner), log1p(t - that partner's last edge time)
  A missing event (no prior edge / no partner) maps to log1p(time span of the whole graph),
  older than any real gap.

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

N_GEOMETRY = 12                 # columns [0, 12)
N_WALK = 17                     # columns [0, 17): geometry + walk time
N_FEATURES = 22                 # columns [17, 22): Tempest history

# Named column subsets for ablations. Column 15 (w-weighted log-age) reads the frozen pooler's
# weights, so the "strict" walk-time set drops it: walk sampling only, no learned parameter.
_GEOMETRY = list(range(0, 12))
_WALK_TIME = list(range(12, 17))
_WALK_TIME_STRICT = [12, 13, 14, 16]
_HISTORY = list(range(17, 22))
FEATURE_SETS = {
    "all": _GEOMETRY + _WALK_TIME + _HISTORY,
    "geometry": _GEOMETRY,
    "geometry+walk": _GEOMETRY + _WALK_TIME,
    "history": _HISTORY,
    "walk+history": _WALK_TIME + _HISTORY,
    "walk-strict+history": _WALK_TIME_STRICT + _HISTORY,
    # Smallest set that holds the 22-column CV on both datasets (chronological CV over
    # train ∪ val, Wikipedia 0.902 vs 0.893, Reddit 0.814 vs 0.813): the three radii and
    # u's prior edge count and time since its last edge.
    "minimal": [0, 5, 6, 17, 18],
    # minimal + the attention pair (spread and entropy under softmax(-d(x_i, p_u))): the only
    # small set that held Reddit's latest CV fold (0.773 vs 0.776 for all 22).
    "minimal+attention": [0, 5, 6, 10, 11, 17, 18],
}
NO_EDGE_LOG_AGE = 16.0          # log1p(age) stand-in when u has no prior edge (> any real age)


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
        """One replayable pass over `split` in chronological batches -> features [N, 22]."""
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
            walk = self._features(tokens, src.to(self.device))
            history = self._history(walker, batch.src, batch.ts)
            out.append(torch.cat([walk, history.to(self.device)], dim=1))
        return torch.cat(out)

    def _history(self, walker: WalkGenerator, src: np.ndarray, ts: np.ndarray) -> torch.Tensor:
        """The 5 Tempest history columns for queries (u, t); see the module docstring."""
        src = src.astype(np.int64)
        ts = ts.astype(np.int64)
        missing = self.no_event_log_gap
        out = np.zeros((len(src), 5), dtype=np.float32)

        out[:, 0] = np.log1p(walker.get_node_popularity(src, ts))
        partner, t_last = walker.get_latest_events(src, ts)
        has_last = t_last >= 0
        out[:, 1] = np.where(has_last, np.log1p(np.maximum(ts - t_last, 0)), missing)

        # second lookup with cutoff t_last -> u's edge before its last one
        out[:, 2] = 0.0
        if has_last.any():
            _, t_prev = walker.get_latest_events(src[has_last], t_last[has_last])
            gap = np.where(t_prev >= 0, t_last[has_last] - t_prev, 0)
            out[has_last, 2] = np.log1p(gap)

        # u's last partner, as of t
        out[:, 3] = 0.0
        out[:, 4] = missing
        has_partner = partner >= 0
        if has_partner.any():
            p, t = partner[has_partner], ts[has_partner]
            out[has_partner, 3] = np.log1p(walker.get_node_popularity(p, t))
            _, p_last = walker.get_latest_events(p, t)
            out[has_partner, 4] = np.where(p_last >= 0, np.log1p(np.maximum(t - p_last, 0)), missing)
        return torch.from_numpy(out)

    def _features(self, tokens: WalkTokens, src: torch.Tensor) -> torch.Tensor:
        geom = self.model.geom
        p_u, w, x = self.model.bag_weights.pool(tokens)                      # [Q,d], [Q,T], [Q,T,d]
        valid = w > 0                                                        # padding has w = 0
        valid_f = valid.float()
        n_valid = valid_f.sum(-1).clamp_min(1)
        e_u = self.model.E.weight[src]                                       # [Q, d]

        d_tok = geom.dist(x, p_u.unsqueeze(-2))                              # [Q, T]
        attention = torch.softmax((-d_tok).masked_fill(~valid, float("-inf")), -1)
        geometry = [
            geom.dist0(p_u),
            (w * d_tok).sum(-1),
            (d_tok * valid_f).sum(-1) / n_valid,
            d_tok.masked_fill(~valid, -1).max(-1).values,
            d_tok.masked_fill(~valid, 1e9).min(-1).values,
            (geom.dist0(x) * valid_f).sum(-1) / n_valid,
            geom.dist0(e_u),
            geom.dist(e_u, p_u),
            -(w.clamp_min(1e-12).log() * w).sum(-1),
            w.max(-1).values,
            (attention * d_tok).sum(-1),
            -(attention.clamp_min(1e-12).log() * attention).sum(-1),
        ]

        real_edge = (tokens.mask & ~tokens.seed_mask).flatten(1)            # [Q, T]
        real_f = real_edge.float()
        n_real = real_f.sum(-1)
        log_age = torch.log1p(tokens.ages.flatten(1).clamp_min(0).float())
        min_log_age = log_age.masked_fill(~real_edge, 1e9).min(-1).values
        walk_time = [
            torch.log1p(n_real),
            min_log_age.where(n_real > 0, torch.full_like(n_real, NO_EDGE_LOG_AGE)),
            (log_age * real_f).sum(-1) / n_real.clamp_min(1),
            (log_age * w).sum(-1),
            self._distinct_nodes(tokens.nodes.flatten(1), valid),
        ]
        return torch.stack(geometry + walk_time, dim=-1)                     # [Q, 17]

    @staticmethod
    def _distinct_nodes(nodes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Number of distinct node ids among each row's valid tokens."""
        ids = torch.sort(nodes.masked_fill(~valid, -1), dim=-1).values
        new_id = torch.ones_like(ids, dtype=torch.bool)
        new_id[:, 1:] = ids[:, 1:] != ids[:, :-1]
        return (new_id & (ids >= 0)).sum(-1).float()

    def state_hash(self) -> str:
        """Fingerprint of every encoder weight, for the before/after-training check."""
        h = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()
