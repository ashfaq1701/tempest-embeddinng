"""Top-level run config. TGB-only."""

from dataclasses import dataclass


@dataclass
class Config:
    # Required (set per-dataset by the entry script after load_tgb).
    tgb_name: str
    max_node_count: int
    is_directed: bool

    # Model
    d_emb: int = 128
    d_hidden_link: int = 128

    # v2.4 §14 source walk encoder. When True, replaces e_t_u (the
    # source-side embedding lookup at the link MLP) with the GRU's
    # output applied to walks seeded on u. Jointly trained with link
    # BCE — gradient flows through encoder INTO E_target/E_context.
    # Default False = locked-v2 baseline (no encoder).
    use_walk_encoder: bool = False

    # Component 0: time encoding at the link MLP (Phase 0.5 of the
    # walk-distribution-matched design).
    # Inputs Δt_u, Δt_v, Δt_uv go through a learned functional time
    # encoder Φ (Xu et al. 2020) with `time_enc_k` frequencies. Three
    # binary cold-start flags are concatenated as scalars alongside.
    use_time_encoding: bool = True
    time_enc_k: int = 16                       # → d_time = 2·k = 32
    cold_start_dt_clamp_factor: float = 100.0  # Δt clamped to factor × time_scale

    # Walks (Tempest)
    max_walk_len: int = 20
    num_walks_per_node: int = 5
    walk_bias: str = "ExponentialWeight"

    # Alignment loss
    temporal_decay_exp: float = 0.5      # β in (1 + Δt/time_scale)^(-β)
    alignment_time_scale: float = -1.0   # ≤ 0 ⇒ derive from training time range
    # Phase 1 ablation: per-position weighting variant in alignment_loss.
    #   "A" = current 1/K · (1+Δt/τ)^(-β) [control]
    #   "B" = 1/K only (sampler does temporal decay)
    #   "C" = uniform α=1 (sampler does everything)
    align_weighting: str = "A"

    # Phase S Group A2 — alignment on/off (v2.2 §4.1).
    # 1.0 = standard walks-supervision (the anchor configuration).
    # 0.0 = alignment loss is computed but contributes zero gradient;
    #       cross-table embeddings only see uniformity. Phase S's first
    #       test of "does walks-supervision help on this dataset?"
    lambda_align: float = 1.0

    # Phase S Group E — link MLP head structure (v2.2 §4.1 / §6.4).
    # "cross_table"      = E.1 (8-block cross-table + Component 0; anchor).
    # "component_0_only" = E.2 (drop cross-table reads entirely; head sees
    #                          only Φ(Δt_u/v/uv) + 3 cold-start bits).
    head_mode: str = "cross_table"
    # E.3 (when head_mode="cross_table"): dropout applied to the 8·d
    # cross-table block before concatenation with Component 0. 0 = no
    # dropout (E.1).
    cross_table_dropout: float = 0.0
    # §4.8.2 architectural sweep — link MLP depth and dropout inside the
    # hidden layers. n_layers defaults to 3 (original v2.3 head). Increase to
    # 5 to test deeper-head fix. link_mlp_dropout applies BETWEEN hidden
    # layers (not on the cross-table block — that's cross_table_dropout).
    link_mlp_n_layers: int = 3
    link_mlp_dropout: float = 0.0
    # Embedding-side dropout — randomly zero entire rows of the embedding
    # lookups after target()/context() retrieve them. Acts as a stochastic
    # regulariser on the embedding tables; reduces the link MLP's reliance
    # on any single row. Default 0 (off).
    embedding_dropout: float = 0.0

    # Phase S Group C — joint training scalar (λ_link).
    # When > 0, link BCE also backprops into the embedding tables, scaled
    # by this value. λ_link=0 keeps the v2.2 decoupled training (embeddings
    # see ONLY the primary loss; link MLP sees ONLY BCE).
    # Under E.2 head this is a no-op (embeddings unread at scoring); under
    # E.1 head it couples the two optimisation paths — candidate fix for
    # the InfoNCE / SGNS rapid-breakdown observed on wiki.
    lambda_link: float = 0.0

    # Phase S §4.7 — loss-family search.
    # "alignment" = v2.2 default (alignment + uniformity, controlled by
    #               lambda_align and eta_uniform).
    # "triplet"   = A3.2 cosine margin loss with semi-hard mining.
    # "infonce"   = A3.1 multi-positive InfoNCE with positional weighting.
    # "sgns"      = A3.3 Skip-gram with negative sampling (unigram^0.75).
    # All three new primaries DROP uniformity (eta_uniform=0 enforced in Trainer).
    primary_loss: str = "alignment"
    # A3.2 triplet (cosine, semi-hard) hyperparameters — literature defaults
    # per v2.3 §4.7.0 generalization guard. No per-dataset tuning.
    triplet_margin: float = 0.5
    # Literature default 1e-4 (amendment v1.3 §4.2) — supplies the norm
    # control that uniformity used to provide. Only APPLIED when
    # primary_loss="triplet"; ignored for alignment / infonce / sgns paths.
    weight_decay_emb: float = 1e-4
    # v2.4 §8 cliff-fix: L2 on the link MLP weights. Stage 2 showed
    # link_w_norm runs away 0.28 → 1.83 (6.5×) even with normbrake clamping
    # the embedding magnitude — this is the residual cliff driver.
    # Always-applied (no loss gating).
    weight_decay_link: float = 0.0
    # A3.1 InfoNCE hyperparameters — literature defaults.
    infonce_tau: float = 0.1
    infonce_num_neg_in_batch: int = 256
    infonce_num_neg_unif: int = 256
    # A3.3 SGNS hyperparameters — Mikolov 2013 defaults.
    sgns_k_neg: int = 5
    sgns_subsample_t: float = 1e-5
    # Linear lr decay 0.025 → 1e-3 over the first `sgns_lr_decay_epochs` epochs
    # then constant at 1e-3 (Mikolov schedule). Only applied to the embedding
    # optimizer when primary_loss="sgns".
    sgns_lr_init: float = 0.025
    sgns_lr_final: float = 1e-3
    sgns_lr_decay_epochs: int = 5

    # Phase S §4.4 — norm-brake auxiliary (composable with any primary).
    # threshold = 1.5 × anchor mean col-norm (calibrated once per dataset).
    # lambda > 0 enables the regularizer.
    lambda_normbrake: float = 0.0
    normbrake_threshold: float = 0.54  # wiki default; recalibrate per dataset

    # Uniformity loss
    eta_uniform: float = 1.0
    uniformity_temperature: float = 2.0
    uniformity_cap: int = 20_000

    # Link prediction
    num_neg_per_pos: int = 10            # K negatives per positive
    # Mixture of TGB-style negatives at TRAINING time. hist_neg_ratio=0.5
    # matches TGB's eval-time 50/50 historical/random mix — keeping train
    # and eval distributions aligned (see CLAUDE.md negatives section).
    # 0 disables the historical channel (uniform random only).
    hist_neg_ratio: float = 0.5
    reservoir_size: int = 32             # per-source Vitter-R reservoir (M)

    # Optimization
    emb_lr: float = 1e-3
    link_lr: float = 1e-3
    target_batch_size: int = 200
    num_epochs: int = 50

    # System
    tgb_root: str = "datasets"
    use_gpu: bool = False
    seed: int = 42
