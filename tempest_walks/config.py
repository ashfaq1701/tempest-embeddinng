"""Minimal production config for tempest-walks-v3.

Only locked architectural defaults — no experiment knobs. Every field
here is part of the production architecture. See CLAUDE.md for the
rationale behind each value.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Required (set per-dataset by the entry script after load_tgb).
    tgb_name: str
    max_node_count: int
    is_directed: bool

    # Model dimensions.
    d_emb: int = 128
    d_hidden_link: int = 128

    # Component 0 — time encoding + cold-start bits at the link MLP.
    time_enc_k: int = 16                       # → d_time = 2·k = 32
    cold_start_dt_clamp_factor: float = 100.0  # Δt clamped to factor × time_scale

    # Walks (Tempest, CPU). num_walks_per_node = 5 is Tempest paper default.
    max_walk_len: int = 20
    num_walks_per_node: int = 5
    walk_bias: str = "ExponentialWeight"

    # Ablation toggles for Lesson 28's Step-3 re-verification — under the
    # locked architecture both default to the production values
    # (encoder ON, tables trainable). Stripped after Step 5 decides.
    use_walk_encoder: bool = True
    freeze_tables: bool = False

    # Alignment loss — variant A (1/depth · (1+Δt/τ)^-β). β=0.5 fixed.
    temporal_decay_exp: float = 0.5
    alignment_time_scale: float = -1.0   # ≤ 0 ⇒ derive (train_span / L_REF=20)

    # Uniformity loss (Wang & Isola 2020).
    eta_uniform: float = 1.0
    uniformity_temperature: float = 2.0
    uniformity_cap: int = 20_000

    # Normbrake + link-MLP weight_decay — the locked cliff fix.
    # threshold = 1.5 × col_norm at ep 1-2, per-dataset (wiki: 3.87, review: 31.32).
    lambda_normbrake: float = 0.1
    normbrake_threshold: float = 3.87        # wiki default; override per dataset
    weight_decay_link: float = 1e-4

    # Negative sampling (training-time). hist_neg_ratio matches TGB eval mix.
    num_neg_per_pos: int = 10
    hist_neg_ratio: float = 0.5
    reservoir_size: int = 32                 # per-source Vitter-R reservoir

    # Optimization.
    emb_lr: float = 1e-3
    link_lr: float = 1e-3
    target_batch_size: int = 200
    num_epochs: int = 50
    early_stop_patience: int = 0             # 0 disables; >0 = patience on val MRR

    # Eval-time conveniences for review-scale datasets.
    monitor_sample_pct: float = 1.0          # < 1.0 = sampled per-epoch monitoring
    skip_final_full_eval: bool = False

    # Diagnostic instrumentation.
    log_debug: bool = False

    # System.
    tgb_root: str = "datasets"
    use_gpu: bool = False
    seed: int = 42
