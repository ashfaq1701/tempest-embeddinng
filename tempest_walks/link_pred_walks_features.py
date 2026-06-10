"""Convert a Tempest WalkData object into the per-position tensors the
DeepSphereSimpleHead consumes.

For each walk row (a single u and its K walks) Tempest returns:
    nodes      [W, L]   int32, -1 = pad, seed at p=lens-1 (bwd) or p=0 (fwd)
    timestamps [W, L]   int64, INT64_MAX at seed slot for bwd,
                                INT64_MIN at seed slot for fwd, -1 = pad
    lens       [W]      int64

The head wants, per (B, W, L):
    E_walks    [B, W, L, d_emb]  walk-node embedding lookups (detached, unit)
    mask       [B, W, L]         True at every VALID position (seed + ctx)
    elapsed    [B, W, L]         log-normalised gap in [0, 1] (recency signal)

Time normalisation (Option B — log compression):
  elapsed := (log1p(gap) / log1p(T_full)).clamp(0, 1),  gap = t_query - t_edge_p
  - T_full = (t_max_full - t_min) covers train + val + test.
  - Seed slots carry the INT64_MAX/MIN sentinel; (t_query - sentinel) wraps to a
    negative under int64 and is caught by clamp(min=0) -> elapsed 0 (most recent),
    so the seed (node u) gets the highest recency weight in the head's pool.
  - log spreads small gaps (most positions sit near 0 under linear gap/T_full)
    across [0, 1], giving the head usable resolution.

Padding positions are masked AND get a safe placeholder (elapsed=1, E_walks=0).
"""
import torch


def make_head_inputs(
    walks_nodes_per_u: list,       # list of [W, L] LongTensor, one per u
    walks_ts_per_u: list,          # list of [W, L] LongTensor
    walks_lens_per_u: list,        # list of [W] LongTensor
    t_query_per_u: torch.Tensor,   # [B] long  — strict-causal query time
    T_full: float,                 # span across train + val + test
    embedding_table,               # EmbeddingTable
    device,
):
    """Stack per-u walks into [B, W, L, ...] tensors with consistent padding to
    the per-batch max (W_max, L_max) and return the dict the head expects."""
    B = len(walks_nodes_per_u)
    W_max = max(t.shape[0] for t in walks_nodes_per_u)
    L_max = max(t.shape[1] for t in walks_nodes_per_u)

    nodes = torch.full((B, W_max, L_max), -1, dtype=torch.long, device=device)
    ts = torch.full((B, W_max, L_max), -1, dtype=torch.long, device=device)
    lens = torch.zeros((B, W_max), dtype=torch.long, device=device)

    for i in range(B):
        ni = walks_nodes_per_u[i].to(device).long()  # [W_i, L_i]
        ti = walks_ts_per_u[i].to(device).long()
        li = walks_lens_per_u[i].to(device).long()
        Wi, Li = ni.shape
        nodes[i, :Wi, :Li] = ni
        ts[i, :Wi, :Li] = ti
        lens[i, :Wi] = li

    # Position-valid mask (every position p in [0, lens[i,w]-1], non-pad node).
    positions = torch.arange(L_max, device=device)
    valid = (positions.view(1, 1, L_max) < lens.unsqueeze(-1)) & (nodes >= 0)

    # Log-normalised elapsed in [0, 1]. The seed-slot sentinel subtraction wraps
    # to negative under int64 and is caught by clamp(min=0) -> elapsed 0.
    Tf = max(float(T_full), 1.0)
    Tf_log = float(torch.log1p(torch.tensor(Tf)).item())
    t_query = t_query_per_u.to(device).long().view(B, 1, 1).expand_as(nodes)
    raw_gap = (t_query - ts).clamp(min=0).clamp(max=int(Tf)).float()
    elapsed = (torch.log1p(raw_gap) / Tf_log).clamp(0.0, 1.0)

    # Embedding lookup (detached — the head never moves E); zero padded slots.
    E_walks = embedding_table(nodes.clamp(min=0)).detach()  # [B, W, L, d_emb]
    E_walks = E_walks * valid.unsqueeze(-1).float()

    return {"E_walks": E_walks, "mask": valid, "elapsed": elapsed}
