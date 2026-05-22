"""Strict-causal training + eval loop for tempest-walks-v3 (minimal production).

Per batch — IN THIS EXACT ORDER (training and eval both):

  1. walks = walk_gen.walks_for_nodes(seeds = unique(batch.src ∪ batch.tgt))
            ← PRE-ingest Tempest state (events ≤ batch B-1).
  2. l_align + η·l_uniform → emb_optimizer.step()
  3. neg = neg_sampler.sample(batch)        ← reservoir ≤ batch B-1.
  4. score = link_predictor(walk_repr_u, e_t_v, e_c_u, e_c_v, Component_0)
            ← e_t_u replaced by walk encoder's per-source GRU output.
  5. (train) BCE → link_optimizer.step()
     (eval)  TGB Evaluator.eval(...)
  6. reservoir.observe(batch.src, batch.tgt)   ← for batch B+1.
  7. time_state.observe(batch)                  ← for batch B+1.
  8. walk_gen.add_edges(batch)                  ← LAST.

The architecture is FIXED: alignment+uniformity primary loss,
weight_decay on link MLP, walk encoder MANDATORY on the source side,
Component 0 always on. See CLAUDE.md.
"""

import copy
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .config import Config
from .data import Batch
from .evaluator import Evaluator
from .losses import alignment_loss, link_bce, uniformity_loss
from .model import EmbeddingStore, LinkPredictor, TimeEncoder
from .negatives import HistoricalNegativeSampler, UniformNegativeSampler, NegativeSampler
from .timestate import NodeTimeState
from .walk_encoder import WalkEncoder
from .walks import WalkGenerator


class Trainer:
    def __init__(
        self,
        config: Config,
        train_dst_pool: np.ndarray,
        node_feat: Optional[np.ndarray] = None,
        edge_feat_dim: int = 0,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device(
            "cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu",
        )

        self.embedding_store = EmbeddingStore(
            n_nodes=config.max_node_count,
            d_emb=config.d_emb,
            node_feat=node_feat,
            edge_feat_dim=edge_feat_dim,
        ).to(self.device)

        d_time = 2 * config.time_enc_k
        self.link_predictor = LinkPredictor(
            d_emb=config.d_emb,
            hidden=config.d_hidden_link,
            d_time=d_time,
        ).to(self.device)

        # Component 0 wiring — time_scale set after dataset load.
        self.time_encoder = TimeEncoder(k=config.time_enc_k, time_scale=1.0).to(self.device)
        self.time_state = NodeTimeState(n_nodes=config.max_node_count)

        self.walk_gen = WalkGenerator(
            is_directed=config.is_directed,
            use_gpu=False,                    # Tempest CPU — preserves VRAM
            walk_bias=config.walk_bias,
            max_walk_len=config.max_walk_len,
            num_walks_per_node=config.num_walks_per_node,
            seed=config.seed,                 # Lesson 33: cross-run reproducibility
        )

        # Walk encoder MANDATORY (no flag — locked architecture).
        has_edge = (edge_feat_dim > 0)
        self.walk_encoder = WalkEncoder(
            d_emb=config.d_emb,
            d_time=d_time,
            has_edge_feat=has_edge,
        ).to(self.device)

        # Training negatives — TGB-protocol-matched mix.
        if config.hist_neg_ratio > 0:
            self.neg_sampler_train: NegativeSampler = HistoricalNegativeSampler(
                num_nodes=config.max_node_count,
                num_neg_per_pos=config.num_neg_per_pos,
                hist_ratio=config.hist_neg_ratio,
                reservoir_size=config.reservoir_size,
                dst_pool=train_dst_pool,
                seed=config.seed,
            )
        else:
            self.neg_sampler_train = UniformNegativeSampler(
                num_neg_per_pos=config.num_neg_per_pos,
                dst_pool=train_dst_pool,
                seed=config.seed,
            )

        # Ablation: freeze identity tables (Step 3 sanity cell). Auxiliary
        # heads (node_feat projections, edge_feat_proj, final fusion
        # projections) are NOT frozen — only the identity tables E_target
        # and E_context that alignment loss directly trains.
        if config.freeze_tables:
            self.embedding_store.E_target.weight.requires_grad_(False)
            self.embedding_store.E_context.weight.requires_grad_(False)

        # Two optimizers — decoupled supervision.
        self.emb_optimizer = torch.optim.Adam(
            [p for p in self.embedding_store.parameters() if p.requires_grad],
            lr=config.emb_lr,
        )
        # Walk encoder + time encoder live in the link-side param group
        # (BCE-trained). Link MLP gets weight_decay (cliff fix).
        link_params = (
            list(self.link_predictor.parameters())
            + list(self.time_encoder.parameters())
            + list(self.walk_encoder.parameters())
        )
        self.link_optimizer = torch.optim.Adam(
            link_params,
            lr=config.link_lr,
            weight_decay=config.weight_decay_link,
        )

        self._time_scale = config.alignment_time_scale  # overridden after dataset load

    def set_time_scale(self, ts: float) -> None:
        """Called by entry script after dataset load (derives ts from train span)."""
        self._time_scale = float(ts)
        # Re-init TimeEncoder's ω_i with the calibrated scale.
        with torch.no_grad():
            k = self.time_encoder.k
            i = torch.arange(k, dtype=torch.float32, device=self.time_encoder.omegas.device)
            self.time_encoder.omegas.copy_(
                (1.0 / ts) * (1000.0 ** (-i / max(k - 1, 1)))
            )

    # ------------------------------------------------------------------ #
    # Per-batch step (strict-causal: ingest LAST).
    # ------------------------------------------------------------------ #

    def _embedding_step(self, batch: Batch) -> Tuple[float, float]:
        """alignment + η·uniformity. Returns (l_align, l_unif)."""
        seeds_np = np.unique(np.concatenate([batch.src, batch.tgt]))
        walks = self.walk_gen.walks_for_nodes(seeds_np)
        nodes = walks.nodes.to(self.device).long().clamp_min(0)
        edge_feats = (
            walks.edge_feats.to(self.device) if walks.edge_feats is not None else None
        )
        e_target_seed = self.embedding_store.target(walks.seeds.to(self.device))   # [N, d]
        e_context_all = self.embedding_store.context_walk(nodes, edge_feats)       # [N*K, L, d]

        t_query = torch.full(
            (walks.seeds.shape[0],), int(batch.t_max),
            dtype=torch.long, device=self.device,
        )
        l_align = alignment_loss(
            e_target_seed=e_target_seed,
            e_context_all=e_context_all,
            walks=walks,
            t_query=t_query,
            beta=self.config.temporal_decay_exp,
            time_scale=self._time_scale,
        )

        unique_batch_nodes = np.unique(np.concatenate([batch.src, batch.tgt]))
        ub = torch.from_numpy(unique_batch_nodes).long().to(self.device)
        l_uniform = uniformity_loss(
            self.embedding_store.target(ub),
            temperature=self.config.uniformity_temperature,
            cap=self.config.uniformity_cap,
        )

        l_total = l_align + self.config.eta_uniform * l_uniform

        # Guard against the all-frozen case: when freeze_tables=True AND
        # walks.edge_feats is None (cold-start before any ingestion on a
        # dataset with edge features, OR datasets without edge features),
        # every operand of l_total has no grad path through any trainable
        # parameter, and backward() raises "element 0 of tensors does not
        # require grad and does not have a grad_fn". Skipping the step is
        # the right semantics — there's nothing to update.
        if l_total.requires_grad:
            self.emb_optimizer.zero_grad(set_to_none=True)
            l_total.backward()
            self.emb_optimizer.step()
        return float(l_align.detach()), float(l_uniform.detach())

    def _e_t_u_for(self, node_ids: np.ndarray, t_query: int) -> torch.Tensor:
        """Source-side e_t_u dispatcher honoring `config.use_walk_encoder`.

        Encoder ON  → GRU walk_repr (default; locked production).
        Encoder OFF → static E_target lookup (W_off ablation cell for
        Lesson 28 Step 3). Same signature as `_compute_walk_repr_for`
        so the evaluator can bind to a single hook.
        """
        if self.config.use_walk_encoder:
            return self._compute_walk_repr_for(node_ids, t_query)
        u_t = torch.from_numpy(node_ids.astype(np.int64)).to(self.device)
        return self.embedding_store.target(u_t)

    def _compute_walk_repr_for(self, node_ids: np.ndarray, t_query: int) -> torch.Tensor:
        """Compute walk_repr[u] for the given node IDs at query timestamp.

        Separate Tempest walks call seeded on unique(node_ids); GRU encodes
        each walk; mean-pool over K walks per seed. Returns a tensor with
        one walk_repr per input node, in input order (repeats look up the
        same row).
        """
        unique_ids = np.unique(node_ids)
        walks = self.walk_gen.walks_for_nodes(unique_ids)
        wn = walks.nodes.to(self.device).long().clamp_min(0)
        wt = walks.timestamps.to(self.device).long()
        wl = walks.lens.to(self.device).long()
        NK = wn.shape[0]
        tq = torch.full(
            (NK,), int(t_query), dtype=torch.long, device=self.device,
        )

        edge_feats_padded = None
        if self.embedding_store.has_edge_feat:
            L = wn.shape[1]
            d_emb = self.embedding_store.d_emb
            if walks.edge_feats is not None:
                ef = walks.edge_feats.to(self.device)
                ef_proj = self.embedding_store.edge_feat_proj(ef.float())   # [NK, L-1, d_emb]
                edge_feats_padded = torch.nn.functional.pad(ef_proj, (0, 0, 0, 1))
            else:
                # Cold-start (epoch-1 first batches): Tempest state empty. Pad zeros.
                edge_feats_padded = torch.zeros(
                    (NK, L, d_emb), dtype=torch.float32, device=self.device,
                )

        walk_repr_unique = self.walk_encoder(
            walk_nodes=wn,
            walk_timestamps=wt,
            walk_lens=wl,
            t_query=tq,
            embedding_store=self.embedding_store,
            time_encoder=self.time_encoder,
            edge_feats_padded=edge_feats_padded,
            K=walks.K,
        )  # [N_unique, d_emb]

        seeds_np = (
            walks.seeds.cpu().numpy()
            if isinstance(walks.seeds, torch.Tensor)
            else np.asarray(walks.seeds)
        )
        idx_map = {int(n): i for i, n in enumerate(seeds_np)}
        row_idx = np.fromiter(
            (idx_map[int(u)] for u in node_ids),
            dtype=np.int64,
            count=len(node_ids),
        )
        return walk_repr_unique[torch.from_numpy(row_idx).long().to(self.device)]

    def _time_features(
        self, all_u: np.ndarray, all_v: np.ndarray, t_query: int,
    ) -> Tuple[torch.Tensor, ...]:
        """Component 0 inputs for the link MLP: Φ(Δt_*) + cold-start bits."""
        last_u, last_v, last_uv = self.time_state.query(all_u, all_v)
        clamp_to = float(self._time_scale) * float(self.config.cold_start_dt_clamp_factor)
        dt_u = np.clip((float(t_query) - last_u).astype(np.float32), 0.0, clamp_to)
        dt_v = np.clip((float(t_query) - last_v).astype(np.float32), 0.0, clamp_to)
        dt_uv = np.clip((float(t_query) - last_uv).astype(np.float32), 0.0, clamp_to)
        cold_u = (last_u == 0).astype(np.float32)
        cold_v = (last_v == 0).astype(np.float32)
        cold_uv = (last_uv == 0).astype(np.float32)
        dev = self.device
        phi_u = self.time_encoder(torch.from_numpy(dt_u).to(dev))
        phi_v = self.time_encoder(torch.from_numpy(dt_v).to(dev))
        phi_uv = self.time_encoder(torch.from_numpy(dt_uv).to(dev))
        cold_u_t = torch.from_numpy(cold_u).to(dev).unsqueeze(-1)
        cold_v_t = torch.from_numpy(cold_v).to(dev).unsqueeze(-1)
        cold_uv_t = torch.from_numpy(cold_uv).to(dev).unsqueeze(-1)
        return phi_u, phi_v, phi_uv, cold_u_t, cold_v_t, cold_uv_t

    def _link_step(self, batch: Batch) -> float:
        neg_src, neg_tgt = self.neg_sampler_train.sample(batch)
        B = len(batch.src)
        K = neg_src.shape[1]

        all_u = np.concatenate([batch.src, neg_src.reshape(-1).astype(np.int64)])
        all_v = np.concatenate([batch.tgt, neg_tgt.reshape(-1).astype(np.int64)])
        u_t = torch.from_numpy(all_u).long().to(self.device)
        v_t = torch.from_numpy(all_v).long().to(self.device)

        # Source-side: GRU walk_repr (default) or static target (W_off ablation).
        e_t_u = self._e_t_u_for(all_u, int(batch.t_max))
        # Other slots: static table lookups.
        e_t_v = self.embedding_store.target(v_t)
        e_c_u = self.embedding_store.context(u_t)
        e_c_v = self.embedding_store.context(v_t)

        phi_u, phi_v, phi_uv, cold_u, cold_v, cold_uv = self._time_features(
            all_u, all_v, int(batch.t_max),
        )
        logits = self.link_predictor(
            e_t_u, e_t_v, e_c_u, e_c_v,
            phi_u, phi_v, phi_uv,
            cold_u, cold_v, cold_uv,
        )
        labels = torch.cat(
            [torch.ones(B, device=self.device), torch.zeros(B * K, device=self.device)],
        )
        loss = link_bce(logits, labels)
        self.link_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.link_optimizer.step()
        return float(loss.detach())

    # ------------------------------------------------------------------ #
    # Snapshot / restore (for early-stop best-weight restoration).
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Detach + move state_dict to CPU. Keeps snapshots off the GPU so
        large embedding tables (e.g., review's 352K-node E_target / E_context
        ≈ 360 MB) don't double the GPU footprint after a best-weight save.
        """
        return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "embedding_store": self._cpu_state_dict(self.embedding_store),
            "link_predictor": self._cpu_state_dict(self.link_predictor),
            "time_encoder": self._cpu_state_dict(self.time_encoder),
            "walk_encoder": self._cpu_state_dict(self.walk_encoder),
        }

    def _restore(self, snap: Dict[str, Any]) -> None:
        # load_state_dict copies values into existing parameters in-place;
        # CPU → GPU transfer happens implicitly via the existing tensors'
        # devices.
        self.embedding_store.load_state_dict(snap["embedding_store"])
        self.link_predictor.load_state_dict(snap["link_predictor"])
        self.time_encoder.load_state_dict(snap["time_encoder"])
        self.walk_encoder.load_state_dict(snap["walk_encoder"])

    # ------------------------------------------------------------------ #
    # Train loop with early-stop snapshot/restore and optional sampled eval.
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_batches_factory,                 # callable → iterable of Batches
        val_evaluator: Optional[Evaluator] = None,
        val_batches_factory=None,              # callable → iterable of Batches
        test_evaluator: Optional[Evaluator] = None,
        test_batches_factory=None,
    ) -> Dict[str, Any]:
        """Training loop. Early-stops on val MRR if patience>0; otherwise runs
        all epochs and reports the final state.

        Per epoch:
          - Reset Tempest + time_state + reservoir.
          - Iterate training batches (strict-causal step).
          - If val_evaluator given: run sampled val eval (full at end).
        """
        n_epochs = self.config.num_epochs
        patience = self.config.early_stop_patience
        sample_pct = self.config.monitor_sample_pct

        best_val = -1.0
        best_test = -1.0
        best_epoch = -1
        best_snap = None
        no_improve = 0

        per_epoch_val: List[float] = []
        per_epoch_test: List[float] = []

        for ep in range(1, n_epochs + 1):
            self.walk_gen.reset()
            self.time_state.reset()
            # Drop the historical-negative reservoir at epoch boundaries.
            # Without this, epoch 1's full chronological pass leaves every
            # source's reservoir reflecting the entire training set, so
            # epoch 2's "historical negatives" can include destinations
            # the source will positively interact with later in the same
            # epoch. Strict-causal violation — see Lesson 28.
            self.neg_sampler_train.reset()
            # Defragment GPU between epochs — review-scale runs accumulate
            # cached allocator fragments during ep 1 that can cause
            # multi-GiB OOMs at ep 2's backward pass on an 8 GB GPU.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.embedding_store.train()
            self.link_predictor.train()
            self.time_encoder.train()
            self.walk_encoder.train()

            t0 = time.time()
            sum_a = sum_u = sum_link = 0.0
            n = 0
            for batch in train_batches_factory():
                la, lu = self._embedding_step(batch)
                ll = self._link_step(batch)
                sum_a += la; sum_u += lu; sum_link += ll
                n += 1
                # Post-scoring block — strict-causal:
                if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
                    self.neg_sampler_train.observe(batch.src, batch.tgt)
                self.time_state.update(batch.src, batch.tgt, batch.ts)
                self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
            train_dt = time.time() - t0

            line = (
                f"  epoch {ep}/{n_epochs}  "
                f"align={sum_a/n:.4f}  uniform={sum_u/n:.4f}  "
                f"link={sum_link/n:.4f}  "
                f"train {train_dt:.1f}s"
            )

            if val_evaluator is not None and val_batches_factory is not None:
                t1 = time.time()
                val_metric = self._eval(val_evaluator, val_batches_factory(), sample_pct)
                eval_dt = time.time() - t1
                per_epoch_val.append(val_metric)
                test_metric = -1.0
                if val_metric > best_val:
                    best_val = val_metric
                    best_epoch = ep
                    best_snap = self._snapshot()
                    no_improve = 0
                    if test_evaluator is not None and test_batches_factory is not None:
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
                    line += f"  val {val_metric:.4f}  patience {no_improve}/{patience}"
                line += f"  eval {eval_dt:.1f}s"
            print(line)

            if patience > 0 and no_improve >= patience:
                break

        if best_snap is not None:
            self._restore(best_snap)
            print(f"  restored best weights from epoch {best_epoch} "
                  f"(val {best_val:.4f}, test {best_test:.4f})")

        # Final full-precision eval (if not skipping).
        if (
            not self.config.skip_final_full_eval
            and val_evaluator is not None
            and val_batches_factory is not None
        ):
            print("=== Final full eval ===")
            self.walk_gen.reset()
            self.time_state.reset()
            self._re_ingest_train(train_batches_factory)
            final_val = self._eval(val_evaluator, val_batches_factory(), 1.0)
            print(f"  val {self.config.tgb_name} mrr: {final_val:.4f}")
            if test_evaluator is not None and test_batches_factory is not None:
                final_test = self._eval(test_evaluator, test_batches_factory(), 1.0)
                print(f"  test {self.config.tgb_name} mrr: {final_test:.4f}")
                best_val, best_test = final_val, final_test

        return {
            "stopped_at_epoch": best_epoch if best_snap is not None else n_epochs,
            "best_val_mrr": best_val,
            "best_test_mrr": best_test,
            "per_epoch_val_mrr": per_epoch_val,
            "per_epoch_test_mrr": per_epoch_test,
        }

    def _eval(self, evaluator: Evaluator, batches: Iterable[Batch], sample_pct: float) -> float:
        """Streaming evaluation. State (walk_gen + time_state + reservoir)
        accumulates as eval proceeds — TGB convention. Model is frozen.
        """
        self.embedding_store.eval()
        self.link_predictor.eval()
        self.time_encoder.eval()
        self.walk_encoder.eval()
        total = 0.0
        n = 0
        for batch in batches:
            m, b = evaluator.evaluate_batch(batch, sample_pct=sample_pct)
            total += m
            n += b
            # Post-scoring: ingest into Tempest + time_state (eval is also strict-causal).
            self.time_state.update(batch.src, batch.tgt, batch.ts)
            self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
        return total / max(n, 1)

    def _re_ingest_train(self, train_batches_factory) -> None:
        """Re-ingest training edges into a fresh Tempest + time_state.
        Used before final full-precision eval to restore the post-training state.
        """
        for batch in train_batches_factory():
            self.time_state.update(batch.src, batch.tgt, batch.ts)
            self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
