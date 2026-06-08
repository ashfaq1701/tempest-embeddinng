"""Convert a Tempest WalkData object into the per-position feature
tensors LinkPredGRU expects.

For each walk row (a single u and its K walks) Tempest returns:
    nodes      [W, L]   int32, -1 = pad, seed at p=lens-1 (bwd) or p=0 (fwd)
    timestamps [W, L]   int64, INT64_MAX at seed slot for bwd,
                                INT64_MIN at seed slot for fwd, -1 = pad
    lens       [W]      int64

The head wants, per (B, W, L):
    E_walks    [B, W, L, d_emb]  walk-node embedding lookups (detached)
    mask       [B, W, L]         True at every VALID position (seed + ctx)
    K_idx      [B, W, L]         hop distance from seed (≥ 0)
    t_feat     [B, W, L, d_T]    TimeEncoder(gap_norm)

Time normalisation (Option B — log compression):
  gap_norm := (log1p(gap) / log1p(T_full)).clamp(0, 1)
  - Non-seed positions: gap = t_query - t_edge_p   (≥ 0 by strict causality)
  - Seed positions:     gap = t_query - t_min      (the override; gives
                                                    the head a "where in
                                                    data lifetime" signal)
  T_full = (t_max_full - t_min) covers train + val + test.

  Why log over linear: at training the typical edge gap is small
  relative to T_full, so linear gap/T_full bunches most positions
  near 0 (~0.005-0.035 on wiki). log1p spreads them across the
  full [0, 1] range (~0.25-0.52 for the same gaps), giving the
  per-position MLP usable resolution. Verified empirically in the
  V0_fwd 2ep bake-off: Option B leads Option A by +0.009 val and
  +0.016 test at ep2 with otherwise-identical config.

Padding positions are masked AND get a safe placeholder value
(gap_norm=0, K_idx=0, E_walks=0).
"""
import torch

# Reflect Tempest sentinels — kept private here to avoid a hard import.
_INT64_MAX = (1 << 63) - 1
_INT64_MIN = -(1 << 63)


def make_head_inputs(
    walks_nodes_per_u: list,       # list of [W, L] LongTensor, one per u
    walks_ts_per_u: list,          # list of [W, L] LongTensor
    walks_lens_per_u: list,        # list of [W] LongTensor
    direction: str,                # "forward" | "backward"
    t_query_per_u: torch.Tensor,   # [B] long  — strict-causal query time
    t_min: int,
    T_full: float,                 # span across train + val + test
    embedding_table,               # EmbeddingTable
    time_encoder,                  # TimeEncoder
    device,
):
    """Stack per-u walks into [B, W, L, ...] tensors with consistent
    padding to the per-batch max (W_max, L_max) and return the dict
    expected by LinkPredGRU."""
    B = len(walks_nodes_per_u)
    W_max = max(t.shape[0] for t in walks_nodes_per_u)
    L_max = max(t.shape[1] for t in walks_nodes_per_u)

    # Allocate B-major tensors.
    nodes = torch.full(
        (B, W_max, L_max), -1, dtype=torch.long, device=device,
    )
    ts = torch.full(
        (B, W_max, L_max), -1, dtype=torch.long, device=device,
    )
    lens = torch.zeros((B, W_max), dtype=torch.long, device=device)

    for i in range(B):
        ni = walks_nodes_per_u[i].to(device).long()  # [W_i, L_i]
        ti = walks_ts_per_u[i].to(device).long()
        li = walks_lens_per_u[i].to(device).long()
        Wi, Li = ni.shape
        nodes[i, :Wi, :Li] = ni
        ts[i, :Wi, :Li] = ti
        lens[i, :Wi] = li

    # Build position-valid mask (every position p ∈ [0, lens[i,w]-1]).
    positions = torch.arange(L_max, device=device)  # [L_max]
    pos_lt_lens = positions.view(1, 1, L_max) < lens.unsqueeze(-1)
    not_pad_node = nodes >= 0
    valid = pos_lt_lens & not_pad_node  # [B, W_max, L_max]

    # K_idx (hop distance from seed):
    #   backward: seed at lens-1, K = lens-1 - p
    #   forward:  seed at 0,      K = p
    if direction == "backward":
        K_idx = (lens.unsqueeze(-1) - 1 - positions.view(1, 1, L_max))
    elif direction == "forward":
        K_idx = positions.view(1, 1, L_max).expand_as(nodes)
    else:
        raise ValueError(direction)
    K_idx = K_idx.clamp(min=0)  # padded slots → 0 (will be masked anyway)

    # Identify seed slot per row.
    if direction == "backward":
        seed_pos = (lens - 1).clamp(min=0)  # [B, W_max]
    else:
        seed_pos = torch.zeros_like(lens)

    # Option B — log compression: gap_norm = log1p(gap) / log1p(T_full).
    Tf = max(float(T_full), 1.0)
    Tf_log = float(torch.log1p(torch.tensor(Tf)).item())
    t_query = t_query_per_u.to(device).long().view(B, 1, 1).expand_as(nodes)
    # Bound raw_gap to T_full so forward-seed-slot INT64_MIN overflow
    # can't pollute autograd before the seed override fires.
    raw_gap = (t_query - ts).clamp(min=0).clamp(max=int(Tf)).float()
    gap_norm = (torch.log1p(raw_gap) / Tf_log).clamp(0.0, 1.0)

    # Seed override.
    seed_oh = torch.zeros_like(nodes, dtype=torch.bool)
    seed_oh.scatter_(2, seed_pos.unsqueeze(-1), True)
    seed_gap = (t_query_per_u.to(device).long() - t_min).clamp(min=0).float()
    seed_gap_norm = (torch.log1p(seed_gap) / Tf_log).clamp(0.0, 1.0).view(B, 1, 1)
    gap_norm = torch.where(
        seed_oh & valid, seed_gap_norm.expand_as(gap_norm), gap_norm,
    )

    # Time features.
    t_feat = time_encoder(gap_norm)  # [B, W, L, d_T]

    # Embedding lookup. clamp to a safe id for padded slots; mask later.
    nodes_safe = nodes.clamp(min=0)
    E_walks = embedding_table(nodes_safe).detach()  # [B, W, L, d_emb]

    # Zero out invalid slots so downstream sums see no leakage. The
    # mask covers max-pool; mean-pool divides by mask count.
    m = valid.unsqueeze(-1).float()
    E_walks = E_walks * m
    t_feat = t_feat * m

    return {
        "E_walks": E_walks,
        "mask": valid,
        "K_idx": K_idx,
        "t_feat": t_feat,
    }


def collect_per_u_walks(walks_for_each_u: list, K_total: int):
    """walks_for_each_u: list of (nodes, ts, lens) triples from Tempest
    for each unique u in the batch. K_total is the requested num walks
    per seed. Returns lists of len(walks_for_each_u) Tensors."""
    nodes_list, ts_list, lens_list = [], [], []
    for nodes, ts, lens in walks_for_each_u:
        nodes_list.append(nodes)
        ts_list.append(ts)
        lens_list.append(lens)
    return nodes_list, ts_list, lens_list
