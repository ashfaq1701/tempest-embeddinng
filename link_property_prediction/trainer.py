"""Per-query-causal training + eval loop — stateless, geometry-free walk head.

Causality is enforced PER QUERY by Tempest's cutoff, not by ingestion order. The FULL graph
(train + val + test) is ingested into Tempest ONCE up front (`ingest_full_graph`, a single
`add_edges` call); there is no per-epoch reset and no per-batch ingestion. Per-batch ordering
(training):
  1. neg = neg_sampler.sample(batch)                 — [B, K_train] uniform negs
  2. candidates = [pos | negs]                       — [B, 1+K_train]
  3. logits = score(src, candidates)                 — TWO-SIDED: sample K cutoff=t_i backward
       walks for the source AND for every candidate, joint-encode both bags with the batch-local
       NodeEncoding, pool each into (x, h) and score with StatelessLinkHead.
  4. L = cross_entropy(logits, target=0)             — Bruch 2019, upper-bounds 1-MRR
  5. one backward + single optimizer step

Why one full graph is valid (and == TPNet): a walk for (u, t) with cutoff = t traverses only
edges with t_edge < t (EXCLUSIVE), so the target edge at t — and any simultaneous/future edge —
is never seen. Because the TGB splits are chronological (train < val < test), a TRAIN query at
time t sees only edges before t: every val/test edge is later and the cutoff excludes it, so
training is causally identical to having ingested train-only. VAL sees train + earlier val; TEST
sees everything before t. This is exactly TPNet's prebuilt-time-index queried strictly-before-t
per edge. The analysis-only stores (stratify) have no cutoff, so they are seeded explicitly over
the causal-past splits.

There is NO learned node-embedding state — the head derives all structure from the walks via the
batch-local, anonymized NodeEncoding — so only the head's few Euclidean params are trained (AdamW).

TOKEN PREP — both sides go through `walk_tokens.build_query_walk_tokens`: walks are generated PER
QUERY (no dedup — each row's (node, t) needs its own cutoff) and returned in the RAW per-walk
WalkTokens layout ([Q, K, L] nodes / nodes_mask / node-aligned timestamps, seeds + cutoffs). The
head jointly encodes the source + candidate bags and, per bag, flattens the walks to a [Q, K*L]
token bag (`walk_tokens.flatten_tokens`) for the attention pooling.
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import Batch, SplitData
from .evaluator import Evaluator
from .model import StatelessLinkHead
from .negatives import UniformNegativeSampler
from .walk_tokens import build_query_walk_tokens
from .walks import WalkGenerator


@dataclass
class TrainerConfig:
    # Dataset-derived.
    num_nodes: int
    dst_pool: np.ndarray

    # Frozen train-split span. Sets the log-spaced init of the μ recency λ (init ≈ 10/t_train).
    # Init only — never a per-step scaler.
    t_train: float = 1.0

    # Model — stateless NodeEncoding + walk-neighbourhood attention head.
    d_emb: int = 128          # NodeEncoding random-feature width per hop-block (JL basis dim).
    n_hops: int = 3           # diffusion depth: node_enc = [X0, ÂX0, …, Âⁿ X0], width (n_hops+1)*d_emb.
    d_ef: int = 0             # per-edge-feature dim (0 = dataset has no edge features); enters the
                             # attention logit as a stable per-token feature. Set from the loaded dataset.
    d_nf: int = 0             # per-node-feature dim (0 = dataset has no node features); enters the
                             # attention logit as a stable per-token feature. Set from the loaded dataset.
    node_feat: Optional[np.ndarray] = None   # [num_nodes, d_nf] static node-feature table (None if absent).
    t2v_dim: int = 16         # Time2Vec output dim for the per-token age feature.

    # Link loss / head.
    K_train: int = 100          # per-query training negatives ([B, 1+K_train])

    # Walks (BACKWARD only, undirected) for the source side (u → μ_u); the one-sided head samples
    # walks only for the source, each candidate v enters through its static embedding E[v].
    num_walks_per_node: int = 10
    max_walk_len: int = 5
    walk_bias: str = "ExponentialWeight"
    start_bias: str = "ExponentialWeight"
    t2nv_p: float = 4.0    # node2vec return param (used only when a bias is TemporalNode2Vec)
    t2nv_q: float = 0.25   # node2vec in-out param; low q/p = most diverse backward walks

    # Optimisation — plain AdamW at a constant LR (no scheduler / decay / warmup). The stateless head is
    # tiny (~few hundred params) and trains smoothly at a flat LR; weight_decay is the only regulariser.
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Run control.
    num_epochs: int = 25
    early_stop_patience: int = 10

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
        # Stateless, geometry-free head: batch-local NodeEncoding + walk-neighbourhood attention.
        self.model = StatelessLinkHead(
            num_nodes=config.num_nodes,
            d_emb=int(config.d_emb),
            n_hops=int(config.n_hops),
            t2v_dim=int(config.t2v_dim),
            d_ef=int(config.d_ef),
            d_nf=int(config.d_nf),
        ).to(self.device)

        # One generator, configured QUERY-side; only the source side samples walks.
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

        # Plain AdamW at a constant LR — no scheduler (like GraphMixer/TPNet, which both train temporal
        # link prediction at a flat LR with no decay/warmup).
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.lr), weight_decay=float(config.weight_decay),
        )

    # ──────────────────────────────────────────────────────────────────
    # Full-graph ingestion (once, up front)
    # ──────────────────────────────────────────────────────────────────

    def ingest_full_graph(self, src: np.ndarray, tgt: np.ndarray, ts: np.ndarray,
                          edge_feat: Optional[np.ndarray] = None) -> None:
        """Ingest the ENTIRE graph (all splits, concatenated) into Tempest in ONE add_edges call.
        The per-query cutoff (t_edge < t_query, EXCLUSIVE) then enforces causality: a train query at
        t sees only edges before t — every val/test edge is chronologically later (TGB splits are
        causal: train < val < test), so the cutoff excludes it; val sees train + earlier val; test
        sees everything before t. Call once before train()/eval; there is no per-epoch reset and no
        per-batch ingestion. Capacity is unbounded — the whole timeline must stay resident."""
        self.walk_gen.add_edges(src, tgt, ts, edge_feat)
        print(f"  Ingested full graph into Tempest: {len(src):,} edges "
              f"(once; per-query cutoff enforces causality)")

    # ──────────────────────────────────────────────────────────────────
    # Scoring — shared by train + eval
    # ──────────────────────────────────────────────────────────────────

    def _score(self, src_t: torch.Tensor, cand_t: torch.Tensor,
               t_query_t: torch.Tensor) -> torch.Tensor:
        """src_t [B] long, cand_t [B, C] long, t_query_t [B] long -> logits [B, C].

        TWO-SIDED per-query walks: the SOURCE side samples K backward walks for each query (u_i, t_i)
        with cutoff = t_i; the CANDIDATE side samples K backward walks for every candidate v_ij with the
        SAME cutoff t_i (so both sides are causal as of the query time). Both bags flow to the head.
        (Plumbing scaffold: the current geometric head is a placeholder — only the two-bag sampling
        matters. Cost: the candidate side is C queries per positive, ~C× the source walks.)"""
        device = self.device

        # SOURCE side: per-query (u_i, t_i) → K cutoff=t_i backward walks → raw [B,K,L] token bag.
        src_tokens = build_query_walk_tokens(
            self.walk_gen, device, src_t, t_query_t,
            max_walk_len=self.config.max_walk_len,
            num_walks_per_node=self.config.num_walks_per_node,
            start_bias=self.config.start_bias,
            walk_bias=self.config.walk_bias,
            node_feat=self.config.node_feat)

        # CANDIDATE side: walk every candidate v with its query's cutoff t_i. Flatten [B,C] → [B*C]
        # query-major; each candidate inherits its query's cutoff so its walk is causal.
        b, c = cand_t.shape
        cand_seeds = cand_t.reshape(-1)                                  # [B*C]
        cand_cutoffs = t_query_t.unsqueeze(1).expand(b, c).reshape(-1)   # [B*C]
        cand_tokens = build_query_walk_tokens(
            self.walk_gen, device, cand_seeds, cand_cutoffs,
            max_walk_len=self.config.max_walk_len,
            num_walks_per_node=self.config.num_walks_per_node,
            start_bias=self.config.start_bias,
            walk_bias=self.config.walk_bias,
            node_feat=self.config.node_feat)

        return self.model(src_tokens, cand_tokens)

    # ──────────────────────────────────────────────────────────────────
    # Per-batch training step
    # ──────────────────────────────────────────────────────────────────

    def _train_step(self, batch: Batch) -> Dict[str, float]:
        device = self.device
        B = len(batch.src)

        # No ingestion here: the full graph is already in Tempest (ingest_full_graph, once).
        # Each query (u, t) walks with cutoff = t (EXCLUSIVE), so it only traverses edges with
        # t_edge < t — every val/test edge is chronologically later and is never seen.
        _, neg_tgt = self.neg_sampler_train.sample(batch)              # [B, K_train]
        src_t = torch.from_numpy(batch.src.astype(np.int64)).to(device)
        cand_np = np.concatenate(
            [batch.tgt.astype(np.int64)[:, None],
             np.ascontiguousarray(neg_tgt, dtype=np.int64)], axis=1)   # [B, 1+K]
        cand_t = torch.from_numpy(cand_np).to(device)
        t_query_t = torch.from_numpy(batch.ts.astype(np.int64)).to(device)

        logits = self._score(src_t, cand_t, t_query_t)                 # [B, 1+K]
        target = torch.zeros(B, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        return {
            "link": float(loss.detach()),
            "lr": float(self.opt.param_groups[0]["lr"]),
        }

    # ──────────────────────────────────────────────────────────────────
    # Eval — strict-causal, no_grad
    # ──────────────────────────────────────────────────────────────────

    def _eval(self, evaluator: Evaluator, batches: Iterable[Batch],
              recorder: Any = None) -> float:
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in batches:
                B = len(batch.src)
                if recorder is not None:
                    recorder.before_batch(batch)

                # No ingestion: the full graph (incl. val/test) is already in Tempest. The
                # per-query cutoff keeps every walk causal (t_edge < t_query), so future eval
                # edges in the index never leak.
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
                logits = self._score(src_t, cand_t, t_query_t).cpu().numpy()

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

    # ──────────────────────────────────────────────────────────────────
    # Snapshot / restore (early-stop)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "model": self._cpu_state_dict(self.model),
        }

    def _restore(self, snap: Dict[str, Any]) -> None:
        self.model.load_state_dict(snap["model"])

    # ──────────────────────────────────────────────────────────────────
    # Train loop
    # ──────────────────────────────────────────────────────────────────

    def train(
        self,
        train_batches_factory,
        full_graph: SplitData,
        val_evaluator: Optional[Evaluator] = None,
        val_batches_factory=None,
        test_evaluator: Optional[Evaluator] = None,
        test_batches_factory=None,
    ) -> Dict[str, Any]:
        # Ingest the FULL graph (train + val + test) into Tempest ONCE, up front. Per-query cutoffs
        # then keep every walk causal (TGB splits are chronological: train < val < test), so there is
        # no per-epoch reset and no per-batch ingestion.
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
                f"lr={self.opt.param_groups[0]['lr']:.1e}  "
                f"train {train_dt:.1f}s"
            )
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
