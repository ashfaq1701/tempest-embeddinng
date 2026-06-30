"""Per-query-causal training + eval loop — link-supervised geometric walk head (REACH).

Causality is now enforced PER QUERY by Tempest's cutoff, not by ingestion order. Per-batch
ordering (training):
  1. walk_gen.add_edges(batch)                        — ingest the batch's true edges FIRST
  2. neg = neg_sampler.sample(batch)                 — [B, K_train] uniform negs
  3. candidates = [pos | negs]                       — [B, 1+K_train]
  4. logits = score(src, candidates)                 — for each query (u_i, t_i) sample K
       backward walks with cutoff = t_i (→ μ_u), pack to tokens, score with
       VelocityHead (identity + velocity-line). Candidate side samples no walks (static E[v]).
  5. L = cross_entropy(logits / tau_link, target=0)  — Bruch 2019, upper-bounds 1-MRR
  6. one backward + single optimizer step

Why ingest-first is valid (and == TPNet): a walk for (u, t) with cutoff = t traverses only
edges with t_edge < t (EXCLUSIVE), so the target edge at t — and any simultaneous/future
batch edge — is never seen, while same-batch-earlier edges (t' < t) legitimately are. This is
exactly TPNet's prebuilt-time-index queried strictly-before-t per edge, and it removes the
batch-size coupling of the old "walk the pre-ingest snapshot" scheme (which hid same-batch-
earlier edges). The stores have no per-query cutoff, so they keep a pre-batch snapshot
(updated last) — their strictly-causal form.

E (on the unit sphere) and the head are trained together by L (no alignment, no detach)
by a single RiemannianAdam.

TOKEN PREP — the source side (u → μ_u) goes through `walk_tokens.build_query_walk_tokens`:
walks are generated PER QUERY (no dedup — each row's (node, t) needs its own cutoff) and returned
in the RAW per-walk WalkTokens layout ([Q, K, L] nodes / nodes_mask / node-aligned timestamps,
seeds + cutoffs). This RAW layout is the SHARED walk contract for every head; VelocityHead
flattens it to a [Q, K*L] token bag and masks padding + the seed node u via
`walk_tokens.flatten_and_exclude_seed`, then builds μ_u with a per-row softmax (ages =
cutoffs − t_edge) and scores identity + velocity against E[v].
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import geoopt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .data import Batch
from .evaluator import Evaluator
from .link_pred_head import VelocityHead
from .model import EmbeddingTable
from .negatives import UniformNegativeSampler
from .utils import make_lr_lambda
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

    # Model.
    d_emb: int = 128

    # Link loss / head.
    tau_link: float = 1.0       # softmax-CE temperature
    K_train: int = 100          # per-query training negatives ([B, 1+K_train])

    # Walks (BACKWARD only, undirected). QUERY-side only: source u → μ_u tokens. The candidate
    # side samples no walks (reach compares against v's static embedding).
    num_walks_per_node_query_side: int = 5
    max_walk_len_query_side: int = 5
    walk_bias_query_side: str = "ExponentialWeight"
    start_bias_query_side: str = "ExponentialWeight"
    max_time_capacity: int = -1   # Tempest sliding-window eviction; -1 = unbounded

    # Optimisation.
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    warmup_steps_cap: int = 500
    decay_horizon_epochs: int = 50

    # Run control.
    num_epochs: int = 50
    early_stop_patience: int = 0

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
        self.embedding_table = EmbeddingTable(
            num_nodes=config.num_nodes, d_emb=config.d_emb,
        ).to(self.device)
        self.link_head = VelocityHead(
            d_emb=int(config.d_emb),
            t_train=float(config.t_train),
        ).to(self.device)

        # One generator, configured QUERY-side; only the source side samples walks.
        self.walk_gen = WalkGenerator(
            use_gpu=config.use_gpu_tempest,
            walk_bias=config.walk_bias_query_side,
            start_bias=config.start_bias_query_side,
            num_walks_per_node=config.num_walks_per_node_query_side,
            max_walk_len=config.max_walk_len_query_side,
            max_time_capacity=config.max_time_capacity,
        )
        self.neg_sampler_train = UniformNegativeSampler(
            num_neg_per_pos=config.K_train, dst_pool=config.dst_pool, seed=config.seed,
        )

        params = list(self.embedding_table.parameters()) + list(self.link_head.parameters())
        self.opt = geoopt.optim.RiemannianAdam(
            params, lr=config.lr, weight_decay=config.weight_decay, stabilize=10)

        self.sched: Optional[LambdaLR] = None
        self._global_step = 0

    # ──────────────────────────────────────────────────────────────────
    # LR schedule (warmup + cosine)
    # ──────────────────────────────────────────────────────────────────

    def _setup_lr_scheduler(self, batches_per_epoch: int) -> None:
        decay_steps = self.config.decay_horizon_epochs * max(batches_per_epoch, 1)
        warmup_steps = max(1, min(
            int(self.config.warmup_fraction * decay_steps), self.config.warmup_steps_cap))
        peak = float(self.config.lr)
        floor = float(self.config.lr_min)
        for pg in self.opt.param_groups:
            pg["lr"] = peak
            pg["initial_lr"] = peak
        ratio = floor / peak if peak > 0 else 0.0
        self.sched = LambdaLR(
            self.opt, lr_lambda=make_lr_lambda(warmup_steps, decay_steps, ratio))
        self._global_step = 0
        print(
            f"  LR schedule (warmup+cosine): peak={peak:.2e} floor={floor:.2e}; "
            f"warmup={warmup_steps} decay_horizon={decay_steps} "
            f"({self.config.decay_horizon_epochs}ep x {batches_per_epoch} batches)"
        )

    # ──────────────────────────────────────────────────────────────────
    # Scoring — shared by train + eval
    # ──────────────────────────────────────────────────────────────────

    def _score(self, src_t: torch.Tensor, cand_t: torch.Tensor,
               t_query_t: torch.Tensor) -> torch.Tensor:
        """src_t [B] long, cand_t [B, C] long, t_query_t [B] long -> logits [B, C].

        PER-QUERY walks: the source side samples K backward walks for each query (u_i, t_i)
        with cutoff = t_i, so every token has t_edge < t_i (strict causal past of that query).
        No dedup — each batch row is its own query. `build_query_walk_tokens` returns a
        self-contained raw WalkTokens (seeds + cutoffs + [Q,K,L] walks); the head flattens + masks
        the seed and builds μ_u with a per-row softmax over the token bag (ages = cutoffs − t_edge,
        all > 0 by the cutoff). The candidate side samples no walks — it enters only through its static
        embedding E[v] (identity + reach). Strict causality comes from the per-query cutoff, NOT
        from ingestion order, so the batch may already be in Tempest."""
        device = self.device

        # --- SOURCE side: per-query (u_i, t_i) → K cutoff=t_i backward walks → raw [Q,K,L]
        # token bag. The head flattens + masks the seed. ---
        src_tokens = build_query_walk_tokens(
            self.walk_gen, device, src_t, t_query_t,
            max_walk_len=self.config.max_walk_len_query_side,
            num_walks_per_node=self.config.num_walks_per_node_query_side,
            start_bias=self.config.start_bias_query_side,
            walk_bias=self.config.walk_bias_query_side)

        return self.link_head(
            self.embedding_table.E.weight,   # the whole table; head indexes E_u / E_v / tokens
            src_tokens,                      # raw source walk tokens (self-contained: seeds+cutoffs)
            cand_t)                          # candidate node ids

    # ──────────────────────────────────────────────────────────────────
    # Per-batch training step
    # ──────────────────────────────────────────────────────────────────

    def _train_step(self, batch: Batch) -> Dict[str, float]:
        device = self.device
        B = len(batch.src)

        # Ingest the batch's true edges into Tempest BEFORE scoring. This is causally
        # valid — and exactly TPNet's protocol (prebuilt time-sorted index queried
        # strictly-before-t per edge) — because every per-query walk is bounded by
        # cutoff = t_query (EXCLUSIVE): the walk for (u, t) only traverses edges with
        # t_edge < t, so the target edge at t (and any simultaneous/future batch edge) is
        # never visible, while same-batch-earlier edges (t' < t) legitimately are. The
        # stores below have NO cutoff mechanism, so they stay strictly causal the only way
        # they can — a pre-batch snapshot, updated AFTER scoring.
        self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)

        _, neg_tgt = self.neg_sampler_train.sample(batch)              # [B, K_train]
        src_t = torch.from_numpy(batch.src.astype(np.int64)).to(device)
        cand_np = np.concatenate(
            [batch.tgt.astype(np.int64)[:, None],
             np.ascontiguousarray(neg_tgt, dtype=np.int64)], axis=1)   # [B, 1+K]
        cand_t = torch.from_numpy(cand_np).to(device)
        t_query_t = torch.from_numpy(batch.ts.astype(np.int64)).to(device)

        logits = self._score(src_t, cand_t, t_query_t)                 # [B, 1+K]
        target = torch.zeros(B, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits / self.config.tau_link, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        if self.sched is not None:
            self.sched.step()
            self._global_step += 1

        return {
            "link": float(loss.detach()),
            "lr": float(self.opt.param_groups[0]["lr"]),
        }

    # ──────────────────────────────────────────────────────────────────
    # Eval — strict-causal, no_grad
    # ──────────────────────────────────────────────────────────────────

    def _eval(self, evaluator: Evaluator, batches: Iterable[Batch],
              recorder: Any = None) -> float:
        self.embedding_table.eval()
        self.link_head.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in batches:
                B = len(batch.src)
                if recorder is not None:
                    recorder.before_batch(batch)

                # Ingest into Tempest BEFORE scoring — the per-query cutoff keeps every
                # walk causal (t_edge < t_query), so eval edges in the index never leak.
                self.walk_gen.add_edges(
                    batch.src, batch.tgt, batch.ts, batch.edge_feat)

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
            "embedding_table": self._cpu_state_dict(self.embedding_table),
            "link_head": self._cpu_state_dict(self.link_head),
        }

    def _restore(self, snap: Dict[str, Any]) -> None:
        self.embedding_table.load_state_dict(snap["embedding_table"])
        self.link_head.load_state_dict(snap["link_head"])

    # ──────────────────────────────────────────────────────────────────
    # Train loop
    # ──────────────────────────────────────────────────────────────────

    def train(
        self,
        train_batches_factory,
        val_evaluator: Optional[Evaluator] = None,
        val_batches_factory=None,
        test_evaluator: Optional[Evaluator] = None,
        test_batches_factory=None,
    ) -> Dict[str, Any]:
        n_epochs = self.config.num_epochs
        patience = self.config.early_stop_patience

        batches_per_epoch = sum(1 for _ in train_batches_factory())
        self._setup_lr_scheduler(batches_per_epoch)

        best_val, best_test, best_epoch = -1.0, -1.0, -1
        best_snap: Optional[Dict[str, Any]] = None
        no_improve = 0
        per_epoch_val: List[float] = []
        per_epoch_test: List[float] = []

        for ep in range(1, n_epochs + 1):
            self.walk_gen.reset()
            self.embedding_table.train()
            self.link_head.train()

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
                f"lr={self.opt.param_groups[0]['lr']:.2e}  "
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
