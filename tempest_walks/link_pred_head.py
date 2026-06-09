"""Walk-mediated link-prediction head — right-to-left GRU over per-position
similarity vectors.

Per (u, t, candidate v), for each of u's K walks and each walk position p:
    feat_p = [Hadamard(E_v, E_w_p), |E_v - E_w_p|, K_emb[hop], TimeEncoder(t)]
             (v's similarity-to-walk-node vector + hop + time; 2·d_emb+d_K+d_T)
A GRU runs RIGHT TO LEFT over the walk — from the seed slot (p = lens-1,
node u) outward to the oldest predecessor (p = 0). The GRU's per-position
hidden states are max+mean pooled over positions into a per-walk vector
(taking only the final, oldest-dominated state would discard the
seed-anchored trajectory), then mean-pooled over walks and passed through a
final MLP to a scalar logit.

Right-to-left + variable length is handled with a single cuDNN-fused
nn.GRU over packed sequences: each walk's valid block is reversed so the
seed (originally at p=lens-1) lands at index 0, then packed by length and
run in one call. This is equivalent to the former per-timestep masked
GRUCell loop — the pool is order-invariant, so the reversed per-position
states give the same pooled vector — but replaces L serialized kernel
launches with one.

Chosen on tgbl-wiki (2026-06-08): peaks at ep6 (val 0.7460 / test 0.7133)
vs the prior per-position-MLP head's ep1 peak 0.7422 / 0.7041 that then
collapsed to ~0.69 — the GRU peaks later AND holds a stable 0.733-0.746
band with no hard drift. Single-seed; the drift-shape fix is the
load-bearing win, the test delta is near the wiki noise band.

E[v] is EXPECTED to be detached upstream — this head's gradients update
only its own parameters; E is shaped by the alignment loss alone (u enters
only through its walks, whose seed slot is node u).

forward(E_v, walks) -> [B, C] logits.
"""
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import checkpoint


class TimeEncoder(nn.Module):
    def __init__(self, n_omega: int = 4):
        super().__init__()
        self.n_omega = n_omega
        omegas = 2.0 * math.pi * (2.0 ** torch.arange(n_omega).float())
        self.register_buffer("omegas", omegas, persistent=False)
        self.d_T = 4 + 2 * n_omega

    def forward(self, gap_norm: torch.Tensor) -> torch.Tensor:
        g = gap_norm.clamp(0.0, 1.0)
        raw = torch.stack([g, torch.exp(-g), g * g, torch.log1p(g)], dim=-1)
        ang = g.unsqueeze(-1) * self.omegas
        sin_cos = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return torch.cat([raw, sin_cos], dim=-1)


class WalkTower(nn.Module):
    """Per-position similarity features -> right-to-left (seed-first) fused
    nn.GRU over packed walks -> max+mean pool over per-position states ->
    mean over walks. out_dim = 2*d_pos (max ‖ mean); d_pos is the GRU
    hidden size."""

    def __init__(
        self,
        d_emb: int,
        max_walk_len: int,
        d_K: int = 16,
        d_pos: int = 96,
        d_T: int = 12,
        chunk_C: int = 0,
    ):
        super().__init__()
        self.chunk_C = int(chunk_C)
        self.K_emb = nn.Embedding(max_walk_len, d_K)
        in_dim = 2 * d_emb + d_K + d_T
        self.gru = nn.GRU(in_dim, d_pos, batch_first=True)
        self.out_dim = 2 * d_pos   # max + mean pool over per-position GRU states

    def _chunk_sim(self, E_walks, E_v_chunk):
        Ew = E_walks.unsqueeze(1)                 # [B,1,W,L,d]
        Ev = E_v_chunk.unsqueeze(2).unsqueeze(3)  # [B,chunk,1,1,d]
        return torch.cat([Ev * Ew, (Ev - Ew).abs()], dim=-1)

    def _process_chunk(self, E_walks, mask, K_idx, t_feat, E_v_chunk):
        B, W, L, _ = E_walks.shape
        chunk = E_v_chunk.shape[1]
        sim = self._chunk_sim(E_walks, E_v_chunk)            # [B,chunk,W,L,2d]
        Ke = self.K_emb(K_idx)                                # [B,W,L,d_K]
        feat = torch.cat([
            sim,
            Ke.unsqueeze(1).expand(B, chunk, W, L, Ke.shape[-1]),
            t_feat.unsqueeze(1).expand(B, chunk, W, L, t_feat.shape[-1]),
        ], dim=-1)                                            # [B,chunk,W,L,in]
        N = B * chunk * W
        feat_flat = feat.reshape(N, L, -1)
        mask_c = (
            mask.unsqueeze(1).expand(B, chunk, W, L).reshape(N, L).to(feat.dtype)
        )
        # Right-to-left (seed-first) recurrence, fused. The former masked
        # GRUCell loop ran p = L-1 .. 0: seed slot (p = lens-1) first, oldest
        # (p = 0) last. Reproduce that order by reversing each walk's valid
        # block so the seed lands at index 0, then run one packed nn.GRU.
        # Padding stays at the tail (never read by pack); the whole
        # per-position feature reverses in lockstep so channels stay aligned;
        # the pool below is order-invariant so H needs no un-reversal.
        lengths = mask_c.sum(dim=1).clamp_min(1).long()          # [N] valid count
        rev = (lengths - 1).unsqueeze(1) - torch.arange(L, device=feat.device)
        rev = rev.clamp_min(0)                                    # pad idx unread
        feat_flat = feat_flat.gather(
            1, rev.unsqueeze(-1).expand(N, L, feat_flat.shape[-1]))
        packed = pack_padded_sequence(
            feat_flat, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        H, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=L)

        # Masked max + mean pool over the per-position GRU states. Rows
        # are always non-empty (the seed slot is valid); the isinf guard
        # is defensive.
        m = mask_c.unsqueeze(-1)                              # [N, L, 1]
        H_neg = H.masked_fill(m == 0, float("-inf"))
        h_max = H_neg.max(dim=1).values                      # [N, hidden]
        h_max = torch.where(torch.isinf(h_max), torch.zeros_like(h_max), h_max)
        h_mean = (H * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        pooled = torch.cat([h_max, h_mean], dim=-1)          # [N, 2*hidden]
        return pooled.reshape(B, chunk, W, self.out_dim).mean(dim=2)  # [B,chunk,2*hidden]

    def forward(self, E_walks, mask, K_idx, t_feat, E_v):
        C = E_v.shape[1]
        if self.chunk_C <= 0 or self.chunk_C >= C:
            return self._process_chunk(E_walks, mask, K_idx, t_feat, E_v)
        out_parts = []
        for c0 in range(0, C, self.chunk_C):
            c1 = min(c0 + self.chunk_C, C)
            E_v_chunk = E_v[:, c0:c1]
            if self.training and torch.is_grad_enabled():
                out_parts.append(checkpoint(
                    self._process_chunk, E_walks, mask, K_idx, t_feat,
                    E_v_chunk, use_reentrant=False))
            else:
                out_parts.append(self._process_chunk(
                    E_walks, mask, K_idx, t_feat, E_v_chunk))
        return torch.cat(out_parts, dim=1)


class LinkPredGRU(nn.Module):
    """Walk-mediated GRU link-prediction head.

    A WalkTower (per-position similarity GRU, pooled) feeds a final MLP
    that maps the pooled walk features to a scalar logit. The walk seed
    slot is node u — kept and compared with each candidate v — so the
    tower carries the u-vs-v signal directly.

    Args:
        d_emb         : embedding width (model config).
        max_walk_len  : L; sizes the K (hop) embedding table.
        d_K           : hop-embedding width.
        d_pos         : GRU hidden size (per-walk vector is 2*d_pos after
                        the max+mean pool).
        chunk_C       : candidate-dim memory chunking (see WalkTower).

    forward(E_v, walks) -> [B, C] logits. E_v is the candidate embedding
    [B, C, d_emb], detached upstream; walks is the per-position feature
    dict (E_walks, mask, K_idx, t_feat).
    """

    def __init__(
        self,
        d_emb: int,
        max_walk_len: int,
        d_K: int = 16,
        d_pos: int = 96,
        chunk_C: int = 0,
    ):
        super().__init__()
        self.time_encoder = TimeEncoder()
        self.tower = WalkTower(
            d_emb=d_emb, max_walk_len=max_walk_len,
            d_K=d_K, d_pos=d_pos, d_T=self.time_encoder.d_T,
            chunk_C=chunk_C,
        )
        final_in = self.tower.out_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(final_in, final_in),
            nn.GELU(),
            nn.Linear(final_in, 1),
        )

    def forward(self, E_v: torch.Tensor, walks: dict) -> torch.Tensor:
        walk_features = self.tower(E_v=E_v, **walks)
        return self.final_mlp(walk_features).squeeze(-1)
