"""TGN-style memory module with proper gradient flow (no BPTT past one batch).

Per-node memory state s[u] ∈ R^{d_emb} maintained across batches via
GRU updates. STRICT-CAUSAL by construction and IN-GRAPH for one batch
so the GRU + message-projection weights actually learn from BCE.

Lifecycle:
  Per epoch:    reset()                                                state ← 0
  Per batch:    apply_pending(unique_batch_nodes)  ← strict-causal:
                  state ← GRU(state, pending_raw_msg) for nodes with pending
                  → keeps state IN AUTOGRAD GRAPH for THIS batch's forward
                Score using state[u]
                Backward + optimizer step (gradients reach GRU + msg_proj)
                detach_state()  ← break the autograd link so next batch
                                   doesn't backprop through stale graph
                update_pending(batch.src, batch.tgt, ts, edge_feat_proj)
                  → build raw_msg from state, write into pending_msg
                    (state used here is also in-graph but post-backward,
                    so the gradient never reaches it — that's fine)

The key difference from the prior stub:
  * `state` is a regular Tensor attribute (not register_buffer) — assignment
    via `self.state = ...` preserves the autograd graph.
  * After backward, the trainer calls `memory.detach_state()` so the next
    batch's GRU update starts from a leaf tensor and doesn't BPTT through
    the entire history.

Strict-causal contract is identical to NodeHistory:
  reads (apply_pending + state) happen at start of batch, before scoring
  writes (update_pending) happen LAST in the post-scoring block
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class NodeMemory(nn.Module):
    """Per-node GRU-updated memory with raw-message store.

    Trainable weights: msg_proj (Linear), mlp_time (2-layer MLP),
                       gru (GRUCell).
    Non-trainable state: self.state, last_update_t, pending_msg, pending_t,
                         has_pending.

    The non-buffer attributes are recreated on `to(device)` via the
    `_to_overrides` hook so `.cuda()` etc. work.
    """

    def __init__(
        self,
        n_nodes: int,
        d_emb: int,
        d_time: int = 16,
        edge_feat_dim: int = 0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_emb = d_emb
        self.d_time = d_time
        self.edge_feat_dim = edge_feat_dim
        self.has_edge_feat = edge_feat_dim > 0

        # Trainable submodules
        msg_in = d_emb + d_time + (d_emb if self.has_edge_feat else 0)
        self.msg_proj = nn.Linear(msg_in, d_emb)
        self.mlp_time = nn.Sequential(
            nn.Linear(2, d_time),
            nn.GELU(),
            nn.Linear(d_time, d_time),
        )
        self.gru = nn.GRUCell(input_size=d_emb, hidden_size=d_emb)

        # Non-buffer state: plain tensors so assignment KEEPS autograd graph.
        # We initialise on CPU; `.to(device)` will move them via _apply().
        self.state = torch.zeros(n_nodes, d_emb)
        self.last_update_t = torch.zeros(n_nodes, dtype=torch.long)
        self.pending_msg = torch.zeros(n_nodes, d_emb)
        self.pending_t = torch.zeros(n_nodes, dtype=torch.long)
        self.has_pending = torch.zeros(n_nodes, dtype=torch.bool)

    def _apply(self, fn):
        """Override nn.Module's _apply so that .to(device) / .cuda() also
        move our non-buffer state tensors. nn.Module's default _apply only
        touches parameters and buffers."""
        super()._apply(fn)
        self.state = fn(self.state)
        self.last_update_t = fn(self.last_update_t)
        self.pending_msg = fn(self.pending_msg)
        self.pending_t = fn(self.pending_t)
        self.has_pending = fn(self.has_pending)
        return self

    # ------------------------------------------------------------------ #
    # Reset (once per training epoch)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def reset(self) -> None:
        self.state = torch.zeros_like(self.state)
        self.last_update_t.zero_()
        self.pending_msg.zero_()
        self.pending_t.zero_()
        self.has_pending.zero_()

    def detach_state(self) -> None:
        """After backward, detach state from the autograd graph so the next
        batch's GRU step starts from a leaf and doesn't BPTT through the
        entire epoch's history."""
        self.state = self.state.detach()
        # pending_msg may also be in-graph; detach for cleanliness
        self.pending_msg = self.pending_msg.detach()

    # ------------------------------------------------------------------ #
    # Apply pending messages BEFORE scoring (in-graph for one batch)
    # ------------------------------------------------------------------ #

    def apply_pending(self, node_ids: torch.Tensor) -> None:
        """For the queried nodes with a pending raw message from prior batch,
        advance state[u] via GRU(state[u], msg[u]). Keeps the resulting
        state[active] IN THE AUTOGRAD GRAPH so backward through this batch's
        link loss reaches the GRU weights.

        We need to:
        1. Gather state and msg for active nodes.
        2. Run GRUCell (in-graph).
        3. Write back to state using scatter (so non-active nodes keep their
           prior state values, also in-graph from earlier batches — but
           detach_state() is called between batches to break the long graph).
        """
        if node_ids.numel() == 0:
            return
        has_pending_subset = self.has_pending[node_ids]
        if not has_pending_subset.any():
            return
        active = node_ids[has_pending_subset]                   # [A]
        cur_state = self.state[active]                          # [A, d]
        cur_msg = self.pending_msg[active]                      # [A, d]
        new_state = self.gru(cur_msg, cur_state)                # in graph

        # Functional scatter so we don't lose graph via in-place op.
        new_full_state = self.state.clone()
        new_full_state[active] = new_state
        self.state = new_full_state

        # Update bookkeeping (no-grad — these are just indices/flags).
        with torch.no_grad():
            self.last_update_t[active] = self.pending_t[active]
            self.has_pending[active] = False
            self.pending_msg[active] = 0.0

    # ------------------------------------------------------------------ #
    # Update pending raw messages AFTER scoring
    # ------------------------------------------------------------------ #

    def update_pending(
        self,
        src: np.ndarray,                # [E] int64
        tgt: np.ndarray,                # [E] int64
        ts: np.ndarray,                 # [E] int64
        edge_feat_proj: Optional[torch.Tensor],   # [E, d_emb] or None
        time_scale: float,
    ) -> None:
        """Build raw messages for batch B's events and stash as pending for
        batch B+1's apply_pending. The state read here reflects events ≤ B−1
        (apply_pending already updated for the current batch's prior pending,
        but THIS batch's events haven't been folded into state yet — that
        happens next batch when apply_pending is called)."""
        device = self.state.device
        if src.shape[0] == 0:
            return
        src_t = torch.from_numpy(src).long().to(device)
        tgt_t = torch.from_numpy(tgt).long().to(device)
        ts_t = torch.from_numpy(ts).long().to(device)

        dt_u = (ts_t - self.last_update_t[src_t]).clamp_min(0).float()
        dt_v = (ts_t - self.last_update_t[tgt_t]).clamp_min(0).float()
        dt_u_norm = dt_u / max(time_scale, 1e-6)
        dt_v_norm = dt_v / max(time_scale, 1e-6)
        te_u = self.mlp_time(torch.stack([dt_u_norm, torch.log1p(dt_u_norm)], dim=-1))
        te_v = self.mlp_time(torch.stack([dt_v_norm, torch.log1p(dt_v_norm)], dim=-1))

        s_v_for_u = self.state[tgt_t]
        s_u_for_v = self.state[src_t]
        parts_u = [s_v_for_u, te_u]
        parts_v = [s_u_for_v, te_v]
        if self.has_edge_feat and edge_feat_proj is not None:
            parts_u.append(edge_feat_proj)
            parts_v.append(edge_feat_proj)
        raw_u = self.msg_proj(torch.cat(parts_u, dim=-1))
        raw_v = self.msg_proj(torch.cat(parts_v, dim=-1))

        # Latest-wins write per node (chronological order in src/tgt arrays).
        new_pending = self.pending_msg.clone()
        new_pending[src_t] = raw_u
        new_pending[tgt_t] = raw_v
        self.pending_msg = new_pending

        with torch.no_grad():
            self.pending_t[src_t] = ts_t
            self.pending_t[tgt_t] = ts_t
            self.has_pending[src_t] = True
            self.has_pending[tgt_t] = True

    # ------------------------------------------------------------------ #
    # Read interface for the link MLP
    # ------------------------------------------------------------------ #

    def read(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Gather state[node_ids]. Reads are in-graph for the current batch
        (so gradients from the link loss flow into the GRU weights via
        apply_pending's in-graph update)."""
        return self.state[node_ids]
