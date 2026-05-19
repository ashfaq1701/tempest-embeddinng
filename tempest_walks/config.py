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

    # Walk encoder (Phase 1 — GRU over per-position walk inputs feeds
    # the alignment loss; d_gru must equal d_emb because the cosine
    # compares target(seed) directly with the GRU output).
    use_walk_encoder: bool = True
    d_time: int = 16
    d_role: int = 8
    walk_encoder_dropout: float = 0.1

    # Cross-pair attention (Phase 3 — DyGFormer-style; pair-conditioned
    # walk summaries feed into the 4-channel link MLP).
    xpair_n_heads: int = 4
    xpair_dropout: float = 0.1
    link_dropout: float = 0.0

    # DyGFormer-style dynamic node encoder. Per-node history buffer +
    # transformer over recent interactions. node_h(u, t) is added to
    # target(u) at the link MLP input (additive residual; cold-start
    # rows get zero from the encoder and fall back to target(u) alone).
    use_node_encoder: bool = True
    k_history: int = 32                   # per-node history window size
    node_enc_n_heads: int = 4
    node_enc_n_layers: int = 1
    node_enc_dropout: float = 0.1
    node_enc_ff_dim: int = 256

    # Alignment loss
    temporal_decay_exp: float = 0.5      # β in (1 + Δt/time_scale)^(-β)
    alignment_time_scale: float = -1.0   # ≤ 0 ⇒ derive from training time range

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
