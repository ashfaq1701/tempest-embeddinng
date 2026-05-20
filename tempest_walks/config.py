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

    # Walks (Tempest)
    max_walk_len: int = 20
    num_walks_per_node: int = 5
    walk_bias: str = "ExponentialWeight"

    # Alignment loss
    temporal_decay_exp: float = 0.5      # β in (1 + Δt/time_scale)^(-β)
    alignment_time_scale: float = -1.0   # ≤ 0 ⇒ derive from training time range

    # Uniformity loss
    eta_uniform: float = 1.0
    uniformity_temperature: float = 2.0
    uniformity_cap: int = 20_000

    # Normbrake — per-column L2 hinge that caps embedding magnitudes above
    # `normbrake_threshold` (CLAUDE.md Lesson 18). Halves the 50-epoch
    # over-training cliff observed in alignment+uniformity. Default OFF
    # (lambda_normbrake=0); enable with CLI flags. Threshold needs
    # per-dataset calibration at 1.5× measured col_norm at ep 1–2.
    # Reference values: wiki 3.87, review 31.32.
    lambda_normbrake: float = 0.0
    normbrake_threshold: float = 0.0

    # weight_decay on the link MLP optimizer (Stage 3 §8 cliff fix). With
    # normbrake clamping embedding col_norms, the residual cliff is driven
    # by link_w_norm runaway (0.28 → 1.83 over 50 ep). weight_decay_link=
    # 1e-4 holds link_w_norm flat (0.19 → 0.17) and shrinks the 50-ep
    # cliff drop from -0.11 to -0.014 on wiki. Default OFF for backward
    # compatibility; pair with normbrake for the production cliff fix.
    weight_decay_link: float = 0.0

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
