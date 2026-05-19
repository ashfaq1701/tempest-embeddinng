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
