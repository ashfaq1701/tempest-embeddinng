"""Per-query-causal training + eval loop with a monotone metric head over walk bags.

The full graph (train + val + test) is ingested into Tempest once up front; causality is enforced
per query by the walk cutoff, not by ingestion order. A walk for (u, t) uses cutoff = t, which is
EXCLUSIVE: it traverses only edges with t_edge < t, so the target edge at t and any
simultaneous/future edge are never seen. TGB-Seq splits are chronological (train < val < test).

Training per batch: sample K_train uniform negatives, form candidates [pos | negs], score them
TWO-SIDED (walks for the source u and for every candidate v, each cut off at the query time t_i),
then cross-entropy with target 0 and a single optimizer step. E and the head train together under
the link CE with no detach.
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import geoopt
import numpy as np
import torch
import torch.nn.functional as F

from .data import Batch, SplitData
from .evaluator import Evaluator
from .model import LinkPredHead
from .negatives import UniformNegativeSampler
from .walk_tokens import build_query_walk_tokens
from .walks import WalkGenerator


@dataclass
class TrainerConfig:
    # Dataset-derived.
    num_nodes: int
    dst_pool: np.ndarray

    # Train-split span; sets the init of the recency scale.
    t_train: float = 1.0

    # Mean-field per-node inter-event time; AGE scale for the pooling recency weight.

    # Embedding dimension.
    d_emb: int = 64

    # Score a learned per-node popularity scalar (zero-init) alongside the distance, mixed by w.
    use_pop_bias: bool = False

    # Pooler MLP width: the net is 3 -> hidden -> 1.
    # Timestamp-grid resolution (data_stats.ts_quantum). Floors the TimeEncoding ladder so no
    # wavelength lands below Nyquist for the grid. 0.0 leaves the bare LAM_MIN in place.
    ts_quantum: float = 0.0

    # Pooler feature widths. Low-priority knobs; not swept.
    time_dim: int = 16
    pos_dim: int = 4
    hidden_dim: int = 32

    # Per-query training negatives ([B, 1+K_train]).
    K_train: int = 5

    # Walks: BACKWARD only, undirected; two-sided (source and every candidate).
    num_walks_per_node: int = 5
    max_walk_len: int = 5
    walk_bias: str = "ExponentialWeight"
    start_bias: str = "ExponentialWeight"
    t2nv_p: float = 4.0    # node2vec return param (used only when a bias is TemporalNode2Vec)
    t2nv_q: float = 0.25   # node2vec in-out param

    # Constant lr, no weight decay. One RiemannianAdam param group at a single `lr`: the
    # embedding tables, the distance temperature and the NN pooler all step at the same rate.
    lr: float = 1e-3

    # Run control.
    num_epochs: int = 50
    early_stop_patience: int = 5

    # System.
    seed: int = 42
    use_gpu: bool = False
    use_gpu_tempest: bool = False


class Trainer:
    def __init__(self, config: TrainerConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device(
            "cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu"
        )
        # Owns the Poincare-ball node embeddings and the monotone metric score.
        self.model = LinkPredHead(
            num_nodes=config.num_nodes,
            d_emb=int(config.d_emb),
            t_train=float(config.t_train),
            max_walk_len=int(config.max_walk_len),
            use_pop_bias=bool(config.use_pop_bias),
            time_dim=int(config.time_dim),
            pos_dim=int(config.pos_dim),
            hidden_dim=int(config.hidden_dim),
            ts_quantum=float(config.ts_quantum),
        ).to(self.device)

        self.walk_gen = WalkGenerator(
            use_gpu=config.use_gpu_tempest,
            walk_bias=config.walk_bias,
            start_bias=config.start_bias,
            num_walks_per_node=config.num_walks_per_node,
            max_walk_len=config.max_walk_len,
            temporal_node2vec_p=config.t2nv_p,
            temporal_node2vec_q=config.t2nv_q,
        )
        self.neg_sampler_train = UniformNegativeSampler(
            num_neg_per_pos=config.K_train, dst_pool=config.dst_pool, seed=config.seed,
        )

        # One param group at a single lr: Riemannian update for E, standard Adam for the rest.
        self.opt = geoopt.optim.RiemannianAdam(
            self.model.parameters(), lr=float(config.lr), stabilize=10,
        )

    # Full-graph ingestion (once, up front)

    def ingest_full_graph(self, src: np.ndarray, tgt: np.ndarray, ts: np.ndarray,
                          edge_feat: Optional[np.ndarray] = None) -> None:
        """Ingest the entire graph (all splits) into Tempest in one add_edges call, once before
        train()/eval. The per-query cutoff (t_edge < t_query, EXCLUSIVE) enforces causality.
        Capacity is unbounded."""
        self.walk_gen.add_edges(src, tgt, ts, edge_feat)
        print(f"  Ingested full graph into Tempest: {len(src):,} edges "
              f"(once; per-query cutoff enforces causality)")

    # Scoring — shared by train + eval

    def _score(self, src_t: torch.Tensor, cand_t: torch.Tensor,
               t_query_t: torch.Tensor):
        """src_t [B], cand_t [B, C], t_query_t [B] (all long) -> logits [B, C]. Two-sided per-query
        walks: K backward walks for the source and for every candidate, each cut off at t_i. Both
        bags go to the head; only the logits are returned."""
        device = self.device

        # Source side: K backward walks for each query (u_i, t_i), cutoff = t_i.
        src_tokens = build_query_walk_tokens(
            self.walk_gen, device, src_t, t_query_t,
            max_walk_len=self.config.max_walk_len,
            num_walks_per_node=self.config.num_walks_per_node,
            start_bias=self.config.start_bias,
            walk_bias=self.config.walk_bias)

        # Candidate side: walk every candidate v with its query's cutoff t_i. Flatten [B,C] → [B*C]
        # query-major; each candidate inherits its query's cutoff.
        b, c = cand_t.shape
        cand_seeds = cand_t.reshape(-1)                                  # [B*C]
        cand_cutoffs = t_query_t.unsqueeze(1).expand(b, c).reshape(-1)   # [B*C]
        cand_tokens = build_query_walk_tokens(
            self.walk_gen, device, cand_seeds, cand_cutoffs,
            max_walk_len=self.config.max_walk_len,
            num_walks_per_node=self.config.num_walks_per_node,
            start_bias=self.config.start_bias,
            walk_bias=self.config.walk_bias)

        return self.model(src_tokens, cand_tokens)

    # Per-batch training step

    def _train_step(self, batch: Batch) -> Dict[str, float]:
        device = self.device
        B = len(batch.src)

        # Full graph already ingested; each query walks with cutoff = t (EXCLUSIVE).
        _, neg_tgt = self.neg_sampler_train.sample(batch)              # [B, K_train]
        src_t = torch.from_numpy(batch.src.astype(np.int64)).to(device)
        cand_np = np.concatenate(
            [batch.tgt.astype(np.int64)[:, None],
             np.ascontiguousarray(neg_tgt, dtype=np.int64)], axis=1)   # [B, 1+K]
        cand_t = torch.from_numpy(cand_np).to(device)
        t_query_t = torch.from_numpy(batch.ts.astype(np.int64)).to(device)

        logits = self._score(src_t, cand_t, t_query_t)                            # [B, 1+K]
        target = torch.zeros(B, dtype=torch.long, device=device)
        link_loss = F.cross_entropy(logits, target)

        # loss = link CE only; E trained end-to-end through the monotone head (no detach).
        loss = link_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        return {
            "link": float(link_loss.detach()),
            "lr": float(self.opt.param_groups[0]["lr"]),
        }

    # Geometry probe

    @torch.no_grad()
    def _geometry_probe(self) -> Dict[str, float]:
        """Boundary watch: mean and max Euclidean norm of E (bulk vs tail radius)."""
        norms = self.model.E.weight.detach().norm(dim=-1)
        return {"max_norm": float(norms.max()), "mean_norm": float(norms.mean())}

    @torch.no_grad()
    def _head_probe(self) -> str:
        """The head's scalar parameters, for the epoch line. Read via hasattr so a head with a different
        set of knobs degrades to a shorter line rather than raising. w = [distance weight, radius-product
        weight] on this head, geo_temp the single scale on heads that carry one instead;
        the popularity channel (when on) rides at fixed unit weight and its per-node bias mean is a
        nuisance quantity (gauge + K-dependent negative-sampling offset), so it is not logged."""
        parts = []
        if hasattr(self.model, "temperature"):
            parts.append(f"temp={float(self.model.temperature):.3f}")
        if hasattr(self.model, "geo_temp"):
            parts.append(f"geo_temp={float(self.model.geo_temp):.3f}")
        if hasattr(self.model, "w"):
            wv = self.model.w.detach().flatten().tolist()
            parts.append("w=[" + ",".join(f"{x:.3f}" for x in wv) + "]")
        return ("  " + "  ".join(parts)) if parts else ""

    # Eval — strict-causal, no_grad

    def _eval(self, evaluator: Evaluator, batches: Iterable[Batch],
              recorder: Any = None) -> float:
        self.model.eval()
        # Rewind the fixed-negative cursor so every eval pass sees the same negatives.
        # Must precede the first sample_negatives call.
        evaluator.reset()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in batches:
                B = len(batch.src)
                if recorder is not None:
                    recorder.before_batch(batch)

                # Full graph already in Tempest; per-query cutoff keeps every walk causal.
                if B == 0:
                    if recorder is not None:
                        recorder.after_batch(batch)
                    continue

                _, neg_tgt_list = evaluator.sample_negatives(batch)
                counts = [int(arr.shape[0]) for arr in neg_tgt_list]
                max_K = max(counts) if counts else 0

                pos_v_np = batch.tgt.astype(np.int64)
                cand_v_np = np.tile(pos_v_np[:, None], (1, 1 + max_K))
                for i in range(B):
                    if counts[i] > 0:
                        cand_v_np[i, 1:1 + counts[i]] = neg_tgt_list[i].astype(np.int64)

                src_t = torch.from_numpy(batch.src.astype(np.int64)).to(self.device)
                cand_t = torch.from_numpy(cand_v_np).to(self.device)
                t_query_t = torch.from_numpy(batch.ts.astype(np.int64)).to(self.device)
                logits = self._score(src_t, cand_t, t_query_t)
                logits = logits.cpu().numpy()

                for i in range(B):
                    rr = evaluator.score_to_metric(
                        float(logits[i, 0]), logits[i, 1:1 + counts[i]])
                    total += rr
                    if recorder is not None:
                        recorder.on_positive(batch, i, rr)
                n += B

                if recorder is not None:
                    recorder.after_batch(batch)
        return total / max(n, 1)

    # Snapshot / restore (early-stop)

    @staticmethod
    def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "model": self._cpu_state_dict(self.model),
        }

    def _restore(self, snap: Dict[str, Any]) -> None:
        self.model.load_state_dict(snap["model"])

    # Train loop

    def train(
        self,
        train_batches_factory,
        full_graph: SplitData,
        val_evaluator: Optional[Evaluator] = None,
        val_batches_factory=None,
        test_evaluator: Optional[Evaluator] = None,
        test_batches_factory=None,
    ) -> Dict[str, Any]:
        # Ingest the full graph (train + val + test) into Tempest once, up front.
        self.ingest_full_graph(
            full_graph.sources, full_graph.destinations,
            full_graph.timestamps, full_graph.edge_feat)

        n_epochs = self.config.num_epochs
        patience = self.config.early_stop_patience

        best_val, best_test, best_epoch = -1.0, -1.0, -1
        best_snap: Optional[Dict[str, Any]] = None
        no_improve = 0
        per_epoch_val: List[float] = []
        per_epoch_test: List[float] = []

        for ep in range(1, n_epochs + 1):
            self.model.train()

            t0 = time.time()
            link_sum, n_batches = 0.0, 0
            for batch in train_batches_factory():
                m = self._train_step(batch)
                link_sum += m["link"]
                n_batches += 1
            train_dt = time.time() - t0

            line = (
                f"epoch {ep}/{n_epochs}  "
                f"link={link_sum / max(n_batches, 1):.4f}  "
                f"lr={self.opt.param_groups[0]['lr']:.0e}  "
                f"train {train_dt:.1f}s"
            )

            # Geometry watch: boundary radius (|E|mean vs |E|max).
            g = self._geometry_probe()
            line += f"  |E|mean={g['mean_norm']:.3f}  |E|max={g['max_norm']:.3f}"
            line += self._head_probe()

            if val_evaluator is not None and val_batches_factory is not None:
                t1 = time.time()
                val_metric = self._eval(val_evaluator, val_batches_factory())
                eval_dt = time.time() - t1
                per_epoch_val.append(val_metric)

                if val_metric > best_val:
                    best_val, best_epoch = val_metric, ep
                    best_snap = self._snapshot()
                    no_improve = 0
                    if test_evaluator is not None and test_batches_factory is not None:
                        best_test = self._eval(test_evaluator, test_batches_factory())
                        per_epoch_test.append(best_test)
                        line += f"  val {val_metric:.4f}  test {best_test:.4f} (new best)"
                    else:
                        line += f"  val {val_metric:.4f} (new best)"
                else:
                    no_improve += 1
                    line += f"  val {val_metric:.4f}  patience {no_improve}/{patience}"
                line += f"  eval {eval_dt:.1f}s"
            print(line)

            if patience > 0 and no_improve >= patience:
                break

        if best_snap is not None:
            self._restore(best_snap)
            print(
                f"  restored best weights from epoch {best_epoch} "
                f"(val {best_val:.4f}, test {best_test:.4f})")

        return {
            "stopped_at_epoch": best_epoch if best_snap is not None else n_epochs,
            "best_val_mrr": best_val,
            "best_test_mrr": best_test,
            "per_epoch_val_mrr": per_epoch_val,
            "per_epoch_test_mrr": per_epoch_test,
        }
