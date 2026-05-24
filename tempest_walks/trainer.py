"""Strict-causal training + eval loop.

Single Trainer class. Per-batch ordering:

  TRAINING:
    1. walks = walk_gen.walks_for_nodes(seeds)       ← pre-ingest state
       seeds = unique(B_src ∪ B_tgt) (undirected) or unique(B_tgt) (directed)
    2. unif_a, unif_b = sample uniformity pairs (M pairs)
    3. L_table = L_align(walks) + η · L_unif(unif_a, unif_b)
    4. neg = neg_sampler.sample(batch)               ← pre-observe state
    5. logits = link_head(E[u].detach(), E[v].detach()) for pos + neg
       L_bce = BCE(logits, [1×B, 0×B*K])
    6. L_total = L_table + L_bce
    7. L_total.backward(); optimizer.step(); optimizer.zero_grad()
    8. neg_sampler.observe(B_src, B_tgt)             ← post-scoring
    9. walk_gen.add_edges(B_src, B_tgt, B_ts, B_ef)  ← post-scoring, last

  EVAL (within torch.no_grad()):
    1. neg_dst_list = tgb_neg_sampler.sample(batch)  ← per-positive negs
    2. Score positives and negatives via link_head(E[u], E[v]).
       Undirected eval: average forward(u, v) + forward(v, u).
    3. evaluator.score_to_metric(pos, neg) per positive (TGB MRR).
    4. walk_gen.add_edges(batch)                     ← Tempest state
                                                       carries forward.
    NOTE: reservoir not updated, walks not sampled at eval.
          Model parameters frozen.

Epoch boundary:
  - walk_gen.reset()
  - neg_sampler.reset() (if Historical)
  - Model parameters and optimiser state are NOT reset.

Final test eval:
  - walk_gen.reset() then re-ingest training edges so Tempest reflects
    "all of training" before val/test scoring begins.

Early stop:
  - Snapshot best-val model + projection + head state_dicts.
  - Restore before final eval.
  - Optimiser state not snapshotted (not needed after training stops).
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import Batch
from .evaluator import Evaluator
from .losses import alignment_loss, uniformity_loss
from .model import (
    EFPredictor,
    EFWeightModulator,
    EmbeddingTable,
    LinkHead,
    ProjectionHead,
)
from .negatives import (
    HistoricalNegativeSampler,
    NegativeSampler,
    UniformNegativeSampler,
)
from .walks import WalkGenerator


@dataclass
class TrainerConfig:
    # Dataset-derived (passed in by train.py).
    num_nodes: int
    is_directed: bool
    is_bipartite: bool
    dst_pool: np.ndarray
    t_train_span: float
    d_node_feat: Optional[int] = None
    d_edge_feat: Optional[int] = None    # NEW (Task 6.7)
    # Task 14: decouple "EF in projection head" from "EF in aux modules".
    # Default True preserves master (Task 6.7). Set False for Task 14
    # runs to get the C1-base (no EF in p_context) plus an aux module.
    ef_in_p_context: bool = True
    # Task 14a: EF modulates alignment weight via tanh modulator.
    # EF never enters projection or LinkHead — training-time signal only.
    ef_modulate_weight: bool = False
    # Task 14b: weight on auxiliary EF prediction loss (predict EF from
    # endpoint E embeddings). 0 = disabled. EF never enters inference path.
    ef_aux_lambda: float = 0.0

    # Model.
    d_emb: int = 128
    d_proj: int = 128

    # Loss-formulation.
    eta_uniform: float = 1.0
    uniform_temperature: float = 2.0
    uniform_pairs: int = 5000
    beta_time: float = 1.0

    # Walks.
    num_walks_per_node: int = 5
    max_walk_len: int = 20
    walk_bias: str = "ExponentialWeight"
    start_bias: str = "Uniform"

    # Negatives (training).
    num_neg_per_pos: int = 10
    hist_neg_ratio: float = 0.5
    reservoir_size: int = 32

    # Optimisation.
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 50
    early_stop_patience: int = 0

    # System.
    seed: int = 42
    use_gpu: bool = False
    use_gpu_tempest: bool = False    # independent from use_gpu

    # Eval.
    monitor_sample_pct: float = 1.0
    skip_final_full_eval: bool = False


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        node_feat: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device(
            "cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu"
        )

        if node_feat is not None:
            assert config.d_node_feat is not None, (
                "node_feat passed but config.d_node_feat is None"
            )
            assert node_feat.shape == (config.num_nodes, config.d_node_feat), (
                f"node_feat shape {node_feat.shape} != "
                f"({config.num_nodes}, {config.d_node_feat})"
            )
            self.node_feat = torch.from_numpy(node_feat).float().to(self.device)
        else:
            self.node_feat = None

        # Model.
        self.embedding_table = EmbeddingTable(
            num_nodes=config.num_nodes,
            d_emb=config.d_emb,
        ).to(self.device)
        self.p_target = ProjectionHead(
            d_emb=config.d_emb,
            d_proj=config.d_proj,
            d_node_feat=config.d_node_feat,
        ).to(self.device)
        # Task 14: ef_in_p_context flag decouples "EF channel in
        # p_context" from d_edge_feat (which the aux modules still need).
        d_ef_for_p_context = (
            config.d_edge_feat if config.ef_in_p_context else None
        )
        self.p_context = ProjectionHead(
            d_emb=config.d_emb,
            d_proj=config.d_proj,
            d_node_feat=config.d_node_feat,
            d_edge_feat=d_ef_for_p_context,   # Task 6.7 / Task 14
        ).to(self.device)
        self.link_head = LinkHead(d_emb=config.d_emb).to(self.device)

        # Task 14a/14b: optional EF auxiliary modules. Both are gated
        # on having edge features in the dataset.
        if config.ef_modulate_weight and config.d_edge_feat is not None:
            self.ef_weight_mod = EFWeightModulator(
                d_edge_feat=config.d_edge_feat,
            ).to(self.device)
        else:
            self.ef_weight_mod = None
        if config.ef_aux_lambda > 0 and config.d_edge_feat is not None:
            self.ef_predictor = EFPredictor(
                d_emb=config.d_emb,
                d_edge_feat=config.d_edge_feat,
            ).to(self.device)
        else:
            self.ef_predictor = None

        # Walk sampler.
        self.walk_gen = WalkGenerator(
            is_directed=config.is_directed,
            use_gpu=config.use_gpu_tempest,
            walk_bias=config.walk_bias,
            start_bias=config.start_bias,
            max_walk_len=config.max_walk_len,
            num_walks_per_node=config.num_walks_per_node,
        )

        # Negative samplers (training).
        if config.hist_neg_ratio > 0:
            self.neg_sampler_train: NegativeSampler = HistoricalNegativeSampler(
                num_nodes=config.num_nodes,
                num_neg_per_pos=config.num_neg_per_pos,
                hist_ratio=config.hist_neg_ratio,
                reservoir_size=config.reservoir_size,
                dst_pool=config.dst_pool,
                seed=config.seed,
            )
        else:
            self.neg_sampler_train = UniformNegativeSampler(
                num_neg_per_pos=config.num_neg_per_pos,
                dst_pool=config.dst_pool,
                seed=config.seed,
            )

        # Uniformity-pair RNG — separate stream so toggling uniform_pairs
        # doesn't perturb the negative-sampling sequence.
        self.unif_rng = np.random.default_rng(config.seed + 1)

        # Single optimiser over all trainable parameters. The decoupling
        # between E-training (alignment + uniformity) and link-head-
        # training (BCE) comes from the .detach() at the BCE call site,
        # NOT from separate optimisers.
        params = (
            list(self.embedding_table.parameters())
            + list(self.p_target.parameters())
            + list(self.p_context.parameters())
            + list(self.link_head.parameters())
        )
        if self.ef_weight_mod is not None:
            params.extend(list(self.ef_weight_mod.parameters()))
        if self.ef_predictor is not None:
            params.extend(list(self.ef_predictor.parameters()))
        self.optimizer = torch.optim.Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _sample_uniformity_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """M independent pairs. Bipartite → both endpoints from dst_pool;
        unipartite → both from the full node range."""
        M = self.config.uniform_pairs
        if self.config.is_bipartite:
            pool = self.config.dst_pool
            idx_a = self.unif_rng.integers(0, pool.shape[0], size=M)
            idx_b = self.unif_rng.integers(0, pool.shape[0], size=M)
            a = torch.from_numpy(pool[idx_a].astype(np.int64))
            b = torch.from_numpy(pool[idx_b].astype(np.int64))
        else:
            a = torch.from_numpy(
                self.unif_rng.integers(0, self.config.num_nodes, size=M)
            ).long()
            b = torch.from_numpy(
                self.unif_rng.integers(0, self.config.num_nodes, size=M)
            ).long()
        return a, b

    def _score_pairs(self, u_ids: torch.Tensor, v_ids: torch.Tensor) -> torch.Tensor:
        """Score [P] (u, v) pairs through E + link_head. Undirected eval
        symmetrises by averaging forward(u, v) + forward(v, u). Used at
        eval (no_grad context); training has its own inline path with
        .detach()."""
        e_u = self.embedding_table(u_ids)
        e_v = self.embedding_table(v_ids)
        logits = self.link_head(e_u, e_v)
        if not self.config.is_directed:
            logits = 0.5 * (logits + self.link_head(e_v, e_u))
        return logits

    # ──────────────────────────────────────────────────────────────────
    # Per-batch training step
    # ──────────────────────────────────────────────────────────────────

    def _train_step(self, batch: Batch) -> Dict[str, float]:
        device = self.device

        # Step 1: walks from PRE-INGEST state.
        if self.config.is_directed:
            seeds_np = np.unique(batch.tgt)
        else:
            seeds_np = np.unique(np.concatenate([batch.src, batch.tgt]))
        walks = self.walk_gen.walks_for_nodes(seeds_np)

        # Step 1b: edge features for alignment loss (convention β).
        # If the dataset has EF but Tempest returned None (cold-start:
        # no edges ingested yet in this epoch), fill with zeros —
        # alignment is masked to zero at cold-start anyway (all
        # walks have lens<=1, no context positions).
        if self.config.d_edge_feat is not None:
            if walks.edge_feats is not None:
                walk_ef = walks.edge_feats.to(device).float()
            else:
                NK, L = walks.nodes.shape
                walk_ef = torch.zeros(
                    NK, L - 1, self.config.d_edge_feat,
                    device=device, dtype=torch.float32,
                )
        else:
            walk_ef = None

        # Step 2: uniformity pairs.
        unif_a, unif_b = self._sample_uniformity_pairs()

        # Step 3: table loss = alignment + η · uniformity.
        t_now = int(batch.t_max)
        l_align = alignment_loss(
            embedding_table=self.embedding_table,
            p_target=self.p_target,
            p_context=self.p_context,
            walks=walks,
            t_now=t_now,
            T_train=self.config.t_train_span,
            beta=self.config.beta_time,
            node_feat=self.node_feat,
            edge_feat=walk_ef,
            ef_weight_mod=self.ef_weight_mod,    # Task 14a
        )
        # Apply uniformity to BOTH heads (Task 12 fix). Each head's
        # EF channel (if present) is bypassed via Option γ. The two
        # head losses are SUMMED (not averaged): heads have disjoint
        # parameter sets, so averaging would halve each head's
        # anti-collapse gradient — empirically verified to cause
        # collapse on C1 smoke in Task 12.
        l_unif_target = uniformity_loss(
            embedding_table=self.embedding_table,
            head=self.p_target,
            sample_idx_a=unif_a,
            sample_idx_b=unif_b,
            t=self.config.uniform_temperature,
            node_feat=self.node_feat,
            bypass_ef=self.p_target.has_ef,
        )
        l_unif_context = uniformity_loss(
            embedding_table=self.embedding_table,
            head=self.p_context,
            sample_idx_a=unif_a,
            sample_idx_b=unif_b,
            t=self.config.uniform_temperature,
            node_feat=self.node_feat,
            bypass_ef=self.p_context.has_ef,
        )
        l_unif = l_unif_target + l_unif_context
        l_table = l_align + self.config.eta_uniform * l_unif

        # Step 4: link-pred negatives from PRE-OBSERVE reservoir.
        neg_src, neg_tgt = self.neg_sampler_train.sample(batch)
        B = len(batch.src)
        K = neg_src.shape[1]
        all_u = np.concatenate([batch.src, neg_src.reshape(-1).astype(np.int64)])
        all_v = np.concatenate([batch.tgt, neg_tgt.reshape(-1).astype(np.int64)])
        u_t = torch.from_numpy(all_u).long().to(device)
        v_t = torch.from_numpy(all_v).long().to(device)

        # Step 5: link logits with DETACHED E (stop-grad on E for BCE).
        e_u = self.embedding_table(u_t).detach()
        e_v = self.embedding_table(v_t).detach()
        logits = self.link_head(e_u, e_v)
        labels = torch.cat([
            torch.ones(B, device=device),
            torch.zeros(B * K, device=device),
        ])
        l_bce = F.binary_cross_entropy_with_logits(logits, labels)

        # Task 14b: optional auxiliary EF prediction loss. Predicts
        # EF from (E[u], E[v]) of POSITIVE edges only (negatives have
        # no EF in TGB). E is NOT detached here — the aux loss gives
        # gradient to E so the table encodes EF-relevant structure.
        # ef_predictor is NOT used at eval (info-symmetric scoring).
        if (
            self.ef_predictor is not None
            and batch.edge_feat is not None
        ):
            pos_u = torch.from_numpy(batch.src.astype(np.int64)).to(device)
            pos_v = torch.from_numpy(batch.tgt.astype(np.int64)).to(device)
            pos_ef = torch.from_numpy(batch.edge_feat).float().to(device)
            e_u_aux = self.embedding_table(pos_u)
            e_v_aux = self.embedding_table(pos_v)
            ef_pred = self.ef_predictor(e_u_aux, e_v_aux)
            l_aux = F.mse_loss(ef_pred, pos_ef)
        else:
            l_aux = torch.tensor(0.0, device=device)

        # Step 6 + 7: total loss + single backward + step.
        l_total = l_table + l_bce + self.config.ef_aux_lambda * l_aux
        self.optimizer.zero_grad(set_to_none=True)
        l_total.backward()
        self.optimizer.step()

        # Step 8: observe positives into reservoir (post-scoring).
        if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
            self.neg_sampler_train.observe(batch.src, batch.tgt)

        # Step 9: ingest into Tempest (LAST).
        self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)

        return {
            "align": float(l_align.detach()),
            "uniform": float(l_unif.detach()),
            "bce": float(l_bce.detach()),
            "aux": float(l_aux.detach()),
            "total": float(l_total.detach()),
        }

    # ──────────────────────────────────────────────────────────────────
    # Eval — strict-causal, no_grad
    # ──────────────────────────────────────────────────────────────────

    def _eval(
        self,
        evaluator: Evaluator,
        batches: Iterable[Batch],
        sample_pct: float = 1.0,
    ) -> float:
        """Streaming eval. Walks NOT sampled; reservoir NOT updated.
        Tempest state advances via post-scoring add_edges."""
        self.embedding_table.eval()
        self.p_target.eval()
        self.p_context.eval()
        self.link_head.eval()

        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in batches:
                B_full = len(batch.src)
                if 0.0 < sample_pct < 1.0:
                    n_keep = max(1, int(B_full * sample_pct))
                    keep_idx = np.random.choice(B_full, size=n_keep, replace=False)
                    score_batch = Batch(
                        src=batch.src[keep_idx],
                        tgt=batch.tgt[keep_idx],
                        ts=batch.ts[keep_idx],
                        edge_feat=(
                            batch.edge_feat[keep_idx]
                            if batch.edge_feat is not None
                            else None
                        ),
                        t_max=(
                            int(batch.ts[keep_idx].max())
                            if len(keep_idx) > 0
                            else batch.t_max
                        ),
                    )
                else:
                    score_batch = batch

                B = len(score_batch.src)
                if B == 0:
                    # Still advance Tempest state with the FULL batch (strict-causal).
                    self.walk_gen.add_edges(
                        batch.src, batch.tgt, batch.ts, batch.edge_feat,
                    )
                    continue

                # TGB-supplied per-positive negative destinations.
                neg_src_list, neg_tgt_list = evaluator.sample_negatives(score_batch)

                # Positive scores in one shot.
                pos_u = torch.from_numpy(score_batch.src.astype(np.int64)).to(self.device)
                pos_v = torch.from_numpy(score_batch.tgt.astype(np.int64)).to(self.device)
                pos_logits = self._score_pairs(pos_u, pos_v).cpu().numpy()

                # Flatten all negatives across positives, score in one shot.
                counts = [int(arr.shape[0]) for arr in neg_tgt_list]
                if sum(counts) > 0:
                    flat_neg_u = np.concatenate(
                        [
                            np.full(counts[i], int(score_batch.src[i]), dtype=np.int64)
                            for i in range(B)
                        ]
                    )
                    flat_neg_v = np.concatenate(
                        [nt.astype(np.int64) for nt in neg_tgt_list]
                    )
                    neg_u_t = torch.from_numpy(flat_neg_u).to(self.device)
                    neg_v_t = torch.from_numpy(flat_neg_v).to(self.device)
                    neg_logits = self._score_pairs(neg_u_t, neg_v_t).cpu().numpy()
                else:
                    neg_logits = np.empty(0, dtype=np.float64)

                cursor = 0
                for i in range(B):
                    K_i = counts[i]
                    m = evaluator.score_to_metric(
                        float(pos_logits[i]),
                        neg_logits[cursor : cursor + K_i],
                    )
                    total += m
                    cursor += K_i
                n += B

                # Strict-causal: advance Tempest with the FULL batch.
                self.walk_gen.add_edges(
                    batch.src, batch.tgt, batch.ts, batch.edge_feat,
                )

        return total / max(n, 1)

    # ──────────────────────────────────────────────────────────────────
    # Snapshot / restore (early-stop)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().to("cpu", copy=True)
            for k, v in module.state_dict().items()
        }

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "embedding_table": self._cpu_state_dict(self.embedding_table),
            "p_target":        self._cpu_state_dict(self.p_target),
            "p_context":       self._cpu_state_dict(self.p_context),
            "link_head":       self._cpu_state_dict(self.link_head),
        }

    def _restore(self, snap: Dict[str, Any]) -> None:
        self.embedding_table.load_state_dict(snap["embedding_table"])
        self.p_target.load_state_dict(snap["p_target"])
        self.p_context.load_state_dict(snap["p_context"])
        self.link_head.load_state_dict(snap["link_head"])

    def _re_ingest_train(self, train_batches_factory) -> None:
        """Reset Tempest, re-ingest all training edges. Used before
        final val/test eval so Tempest state matches post-training."""
        self.walk_gen.reset()
        for batch in train_batches_factory():
            self.walk_gen.add_edges(
                batch.src, batch.tgt, batch.ts, batch.edge_feat,
            )

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
        sample_pct = self.config.monitor_sample_pct

        best_val = -1.0
        best_test = -1.0
        best_epoch = -1
        best_snap: Optional[Dict[str, Any]] = None
        no_improve = 0
        per_epoch_val: List[float] = []
        per_epoch_test: List[float] = []

        for ep in range(1, n_epochs + 1):
            # Epoch boundary: walks + reservoir reset only.
            self.walk_gen.reset()
            if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
                self.neg_sampler_train.reset()

            self.embedding_table.train()
            self.p_target.train()
            self.p_context.train()
            self.link_head.train()

            t0 = time.time()
            sums = {"align": 0.0, "uniform": 0.0, "bce": 0.0, "aux": 0.0, "total": 0.0}
            n_batches = 0
            for batch in train_batches_factory():
                metrics = self._train_step(batch)
                for k in sums:
                    sums[k] += metrics[k]
                n_batches += 1
            train_dt = time.time() - t0

            line = (
                f"epoch {ep}/{n_epochs}  "
                f"align={sums['align']/max(n_batches,1):.4f}  "
                f"unif={sums['uniform']/max(n_batches,1):.4f}  "
                f"bce={sums['bce']/max(n_batches,1):.4f}  "
                f"aux={sums['aux']/max(n_batches,1):.4f}  "
                f"train {train_dt:.1f}s"
            )

            if val_evaluator is not None and val_batches_factory is not None:
                t1 = time.time()
                val_metric = self._eval(
                    val_evaluator, val_batches_factory(), sample_pct,
                )
                eval_dt = time.time() - t1
                per_epoch_val.append(val_metric)

                test_metric = -1.0
                if val_metric > best_val:
                    best_val = val_metric
                    best_epoch = ep
                    best_snap = self._snapshot()
                    no_improve = 0
                    if (
                        test_evaluator is not None
                        and test_batches_factory is not None
                    ):
                        test_metric = self._eval(
                            test_evaluator, test_batches_factory(), sample_pct,
                        )
                        best_test = test_metric
                        per_epoch_test.append(test_metric)
                        line += f"  val {val_metric:.4f}  test {test_metric:.4f} (new best)"
                    else:
                        line += f"  val {val_metric:.4f} (new best)"
                else:
                    no_improve += 1
                    line += (
                        f"  val {val_metric:.4f}  "
                        f"patience {no_improve}/{patience}"
                    )
                line += f"  eval {eval_dt:.1f}s"
            print(line)

            if patience > 0 and no_improve >= patience:
                break

        if best_snap is not None:
            self._restore(best_snap)
            print(
                f"  restored best weights from epoch {best_epoch} "
                f"(val {best_val:.4f}, test {best_test:.4f})"
            )

        if (
            not self.config.skip_final_full_eval
            and val_evaluator is not None
            and val_batches_factory is not None
        ):
            print("=== Final full eval ===")
            self._re_ingest_train(train_batches_factory)
            final_val = self._eval(val_evaluator, val_batches_factory(), 1.0)
            print(f"  val MRR: {final_val:.4f}")
            if test_evaluator is not None and test_batches_factory is not None:
                final_test = self._eval(test_evaluator, test_batches_factory(), 1.0)
                print(f"  test MRR: {final_test:.4f}")
                best_val, best_test = final_val, final_test

        return {
            "stopped_at_epoch": best_epoch if best_snap is not None else n_epochs,
            "best_val_mrr": best_val,
            "best_test_mrr": best_test,
            "per_epoch_val_mrr": per_epoch_val,
            "per_epoch_test_mrr": per_epoch_test,
        }
