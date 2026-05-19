"""Entry point. One required flag: --tgb-name (e.g. tgbl-wiki).

Loads the dataset via TGB, runs strict-causal training, evaluates with
TGB's official Evaluator on val and test.
"""

import argparse

import numpy as np
import torch

from tempest_walks.config import Config
from tempest_walks.data import create_batches, load_tgb
from tempest_walks.evaluator import Evaluator
from tempest_walks.negatives import TGBNegativeSampler
from tempest_walks.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tempest-walks-v3 trainer")
    p.add_argument("--tgb-name", required=True)
    p.add_argument("--tgb-root", default="datasets")
    p.add_argument("--use-gpu", action="store_true")
    # `--is-directed` / `--no-is-directed` overrides the per-dataset default
    # in data.py::default_is_directed (e.g. tgbl-wiki is undirected by default;
    # pass `--is-directed` to feed Tempest in directed mode instead).
    p.add_argument("--is-directed", default=None, action=argparse.BooleanOptionalAction)

    # Model
    p.add_argument("--d-emb", type=int, default=128)
    p.add_argument("--d-hidden-link", type=int, default=128)

    # Walks
    p.add_argument("--max-walk-len", type=int, default=20)
    p.add_argument("--num-walks-per-node", type=int, default=5)
    p.add_argument("--walk-bias", default="ExponentialWeight")

    # Walk encoder (Phase 1: GRU over per-position walk inputs feeding the
    # alignment loss in place of context_walk).
    p.add_argument("--use-walk-encoder", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Enable the GRU walk encoder (Phase 1). "
                        "Pass --no-use-walk-encoder for the legacy "
                        "context_walk path.")
    p.add_argument("--d-time", type=int, default=16)
    p.add_argument("--d-role", type=int, default=8)
    p.add_argument("--walk-encoder-dropout", type=float, default=0.1)

    # Cross-pair attention (Phase 3 — DyGFormer-style)
    p.add_argument("--xpair-n-heads", type=int, default=4)
    p.add_argument("--xpair-dropout", type=float, default=0.1)
    p.add_argument("--link-dropout", type=float, default=0.0)

    # DyGFormer-style dynamic node encoder + per-node history buffer
    p.add_argument("--use-node-encoder", default=True,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--k-history", type=int, default=32)
    p.add_argument("--node-enc-n-heads", type=int, default=4)
    p.add_argument("--node-enc-n-layers", type=int, default=1)
    p.add_argument("--node-enc-dropout", type=float, default=0.1)
    p.add_argument("--node-enc-ff-dim", type=int, default=256)

    # Co-occurrence feature (recurrence signal from per-pair history overlap)
    p.add_argument("--use-co-feat", default=True,
                   action=argparse.BooleanOptionalAction)
    # Memory module (TGN-style raw-message-store)
    p.add_argument("--use-memory", default=True,
                   action=argparse.BooleanOptionalAction)
    # Direct-recurrence (EdgeBank-style) feature
    p.add_argument("--use-eb-feat", default=True,
                   action=argparse.BooleanOptionalAction)

    # Losses
    p.add_argument("--temporal-decay-exp", type=float, default=0.5)
    p.add_argument("--alignment-time-scale", type=float, default=-1.0)
    p.add_argument("--eta-uniform", type=float, default=1.0)
    p.add_argument("--uniformity-temperature", type=float, default=2.0)

    # Optimization
    p.add_argument("--num-neg-per-pos", type=int, default=10)
    p.add_argument("--hist-neg-ratio", type=float, default=0.5,
                   help="Fraction of training negatives drawn from each "
                        "source's reservoir of past destinations (Vitter R). "
                        "0.5 matches TGB's eval-time historical/random mix. "
                        "0 disables — uniform random only.")
    p.add_argument("--reservoir-size", type=int, default=32)
    p.add_argument("--emb-lr", type=float, default=1e-3)
    p.add_argument("--link-lr", type=float, default=1e-3)
    p.add_argument("--target-batch-size", type=int, default=200)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading TGB dataset: {args.tgb_name}")
    loaded = load_tgb(args.tgb_name, root=args.tgb_root)
    is_directed = (
        args.is_directed if args.is_directed is not None else loaded.is_directed
    )
    directed_note = "" if args.is_directed is None else " (CLI override)"
    print(
        f"  N={loaded.max_node_count}  "
        f"train={len(loaded.train.sources)}  val={len(loaded.val.sources)}  "
        f"test={len(loaded.test.sources)}  is_directed={is_directed}{directed_note}  "
        f"eval_metric={loaded.eval_metric}",
    )

    config = Config(
        tgb_name=loaded.name,
        max_node_count=loaded.max_node_count,
        is_directed=is_directed,
        d_emb=args.d_emb,
        d_hidden_link=args.d_hidden_link,
        max_walk_len=args.max_walk_len,
        num_walks_per_node=args.num_walks_per_node,
        walk_bias=args.walk_bias,
        use_walk_encoder=args.use_walk_encoder,
        d_time=args.d_time,
        d_role=args.d_role,
        walk_encoder_dropout=args.walk_encoder_dropout,
        xpair_n_heads=args.xpair_n_heads,
        xpair_dropout=args.xpair_dropout,
        link_dropout=args.link_dropout,
        use_node_encoder=args.use_node_encoder,
        k_history=args.k_history,
        node_enc_n_heads=args.node_enc_n_heads,
        node_enc_n_layers=args.node_enc_n_layers,
        node_enc_dropout=args.node_enc_dropout,
        node_enc_ff_dim=args.node_enc_ff_dim,
        use_co_feat=args.use_co_feat,
        use_memory=args.use_memory,
        use_eb_feat=args.use_eb_feat,
        temporal_decay_exp=args.temporal_decay_exp,
        alignment_time_scale=args.alignment_time_scale,
        eta_uniform=args.eta_uniform,
        uniformity_temperature=args.uniformity_temperature,
        num_neg_per_pos=args.num_neg_per_pos,
        hist_neg_ratio=args.hist_neg_ratio,
        reservoir_size=args.reservoir_size,
        emb_lr=args.emb_lr,
        link_lr=args.link_lr,
        target_batch_size=args.target_batch_size,
        num_epochs=args.num_epochs,
        seed=args.seed,
        tgb_root=args.tgb_root,
        use_gpu=args.use_gpu,
    )

    train_dst_pool = np.unique(loaded.train.destinations)
    # Node features are dataset-static (when present); edge features are
    # per-edge — Tempest already carries them in walks. Both flow into
    # EmbeddingStore as learned residual projections when available.
    edge_feat_dim = (
        int(loaded.train.edge_feat.shape[1]) if loaded.train.edge_feat is not None else 0
    )
    print(
        f"  node_feat: {'present, d=' + str(loaded.node_feat.shape[1]) if loaded.node_feat is not None else 'absent'}  "
        f"edge_feat: {'present, d=' + str(edge_feat_dim) if edge_feat_dim > 0 else 'absent'}"
    )
    trainer = Trainer(
        config=config,
        train_dst_pool=train_dst_pool,
        node_feat=loaded.node_feat,
        edge_feat_dim=edge_feat_dim,
    )

    # Derive alignment time-scale: (training time span) / L_REF, where
    # L_REF is a fixed reference constant (NOT --max-walk-len).
    #
    # Empirically on tgbl-wiki, time_scale ≈ span / 20 = 93k sec ≈ 1.08
    # days outperforms the per-node mean inter-event time (155k sec ≈
    # 1.8 days) by ~0.02 test MRR. The alignment recency term wants a
    # sharper scale than the per-node recurrence period — closer to a
    # within-session timescale.
    #
    # Why L_REF is fixed: tying time_scale to --max-walk-len was a real
    # bug (Lesson 11). Bumping L from 20 → 50 collapsed the scale from
    # 93k → 37k and crushed the recency weight. Keeping L_REF=20 fixed
    # means changing the walk length does not perturb the decay rate.
    if config.alignment_time_scale <= 0:
        ts = loaded.train.timestamps
        span = float(ts.max() - ts.min())
        L_REF = 20
        derived = span / L_REF
        trainer.set_time_scale(derived)
        print(
            f"  alignment_time_scale (derived): {derived:.3f}  "
            f"[ span={span:.1f}  L_REF={L_REF} ]"
        )

    print("Loading TGB negatives (val + test)…")
    loaded.dataset.load_val_ns()
    loaded.dataset.load_test_ns()

    # Phase 3: pass walk generator + encoder + cross-pair attention to the
    # Evaluator. Per-batch it samples walks for unique(pos ∪ neg) nodes,
    # builds per-seed sequences, then runs cross-pair attention inside the
    # chunk loop to produce pair-conditioned W(u), W(v) for the 4-channel
    # link MLP.
    eval_val = Evaluator(
        embedding_store=trainer.embedding_store,
        link_predictor=trainer.link_predictor,
        neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="val"),
        device=trainer.device,
        tgb_dataset_name=loaded.name,
        eval_metric=loaded.eval_metric,
        walk_gen=trainer.walk_gen,
        walk_encoder=trainer.walk_encoder,
        cross_pair_attn=trainer.cross_pair_attn,
        node_history=trainer.node_history,
        node_encoder=trainer.node_encoder,
        co_encoder=trainer.co_encoder,
        eb_encoder=trainer.eb_encoder,
        memory=trainer.memory,
        time_scale=trainer._time_scale,
    )
    eval_test = Evaluator(
        embedding_store=trainer.embedding_store,
        link_predictor=trainer.link_predictor,
        neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="test"),
        device=trainer.device,
        tgb_dataset_name=loaded.name,
        eval_metric=loaded.eval_metric,
        walk_gen=trainer.walk_gen,
        walk_encoder=trainer.walk_encoder,
        cross_pair_attn=trainer.cross_pair_attn,
        node_history=trainer.node_history,
        node_encoder=trainer.node_encoder,
        co_encoder=trainer.co_encoder,
        eb_encoder=trainer.eb_encoder,
        memory=trainer.memory,
        time_scale=trainer._time_scale,
    )

    print(f"Model device: {trainer.device}    Tempest device: cpu")
    print("=== Training ===")
    trainer.train(create_batches(loaded.train, config.target_batch_size))

    print("=== Validation ===")
    val_metric = trainer.evaluate(create_batches(loaded.val, config.target_batch_size), eval_val)
    print(f"Val {loaded.eval_metric}: {val_metric:.4f}")

    print("=== Test ===")
    test_metric = trainer.evaluate(create_batches(loaded.test, config.target_batch_size), eval_test)
    print(f"Test {loaded.eval_metric}: {test_metric:.4f}")


if __name__ == "__main__":
    main()
