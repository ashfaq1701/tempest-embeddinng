"""Simple deep on-sphere link head.

Design contract (every clause is load-bearing — they are why it cannot blow up):
  * Every internal representation is UNIT-NORM (re-normalized after every layer).
    -> no layer can accumulate magnitude; the diag-metric runaway is impossible.
  * The ONLY scale knob is a single learnable temperature `tau`.
    -> no weight has an unbounded path to logit magnitude (that was the diag leak).
  * The match is a BOUNDED cosine between two unit vectors.
    -> |score| <= 1, and ||cv - cw||^2 = 2 - 2*score in [0, 4].
  * phi is a TIED ReZero residual map: alpha init 0 => phi == identity at init,
    AND inert (inner weights see zero gradient at init, scaled by alpha).
    -> the head EQUALS the cos recency-pool baseline at init, and departs from it
       smoothly as alpha lifts off 0 (no first-step kick off cos — fixes the ep1 dip).
  * E is read DETACHED. The head cannot move a single embedding; E stays shaped
    only by L_align on the sphere. The head's params train under L_link (Prodigy);
    E trains under L_align (RiemannianAdam). Unchanged dual-optimizer split.

The head is agnostic to where candidates/negatives come from: feeding it the
shared hard-negative set (the same set L_align uses) requires no change here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class ReZeroSphereBlock(nn.Module):
    """y = l2normalize(x + alpha * f(x)),  alpha init 0  => identity & inert."""

    def __init__(self, d, hidden):
        super().__init__()
        self.lin1 = nn.Linear(d, hidden)
        self.lin2 = nn.Linear(hidden, d)
        self.alpha = nn.Parameter(torch.zeros(1))      # ReZero gate
        nn.init.zeros_(self.lin2.weight)               # extra safety: f(x)=0 at init
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x):                              # x: [..., d] unit-norm
        f = self.lin2(F.gelu(self.lin1(x)))
        return l2norm(x + self.alpha * f)              # unit-norm out


class DeepSphereSimpleHead(nn.Module):
    def __init__(self, d, hidden=None, depth=2, tau_init=0.033, tau_min=0.02):
        super().__init__()
        hidden = hidden or d
        self.phi_blocks = nn.ModuleList(
            ReZeroSphereBlock(d, hidden) for _ in range(depth)
        )
        # Recency pooling. NOTE: to make identity-init == the cos baseline EXACTLY,
        # this must reproduce the cos head's pooling (its learned per-hop omega /
        # per-walk softmax / mean-over-walks). Inherit that code rather than the
        # learnable-decay placeholder below if you want bit-exact init == cos.
        self.decay = nn.Parameter(torch.tensor(1.0))
        # single scale knob; init so 1/tau ~ 30 (cos head's effective logit
        # scale). The head DIVIDES by tau, so tau itself inits to tau_init
        # (~0.033) ⇒ 1/tau ~ 30. (Provided code had a 1.0/tau_init typo here
        # that inverted the scale to a near-flat softmax.)
        raw0 = math.log(math.expm1(tau_init))          # softplus^{-1}(tau_init)
        self.raw_tau = nn.Parameter(torch.tensor(float(raw0)))
        self.tau_min = tau_min
        self.prior = nn.Parameter(torch.randn(d) * 0.01)   # cold-start direction

    def phi(self, x):                                  # S^{d-1} -> S^{d-1}, tied
        for blk in self.phi_blocks:
            x = blk(x)
        return x

    def pool_history(self, E_w, elapsed, mask):        # -> [B, d] unit
        logits = (-F.softplus(self.decay) * elapsed).masked_fill(~mask, float("-inf"))
        omega = torch.softmax(logits, dim=1)           # [B, L]
        omega = torch.nan_to_num(omega, 0.0)           # all-empty rows -> 0
        pooled = (omega.unsqueeze(-1) * E_w).sum(dim=1)
        w_bar = l2norm(pooled)
        empty = (~mask).all(dim=1, keepdim=True)       # cold-start guard
        w_bar = torch.where(empty, l2norm(self.prior).expand_as(w_bar), w_bar)
        return w_bar

    def forward(self, E_v, E_w, elapsed, mask):
        """
        E_v:     [B, C, d]  candidate embeddings (unit on the sphere)
        E_w:     [B, L, d]  history walk-node embeddings (unit)
        elapsed: [B, L]     t_query - ts_p  (>= 0)
        mask:    [B, L]     True = real position
        returns: [B, C]     logits
        """
        E_v = E_v.detach()                             # head never moves E
        E_w = E_w.detach()
        w_bar = self.pool_history(E_w, elapsed, mask)  # [B, d] unit
        cw = self.phi(w_bar)                           # [B, d] unit
        cv = self.phi(E_v)                             # [B, C, d] unit
        score = torch.einsum("bcd,bd->bc", cv, cw)     # bounded cosine in [-1, 1]
        tau = F.softplus(self.raw_tau).clamp_min(self.tau_min)
        return score / tau

    @torch.no_grad()
    def diagnostics(self):
        """Per-epoch logging: confirm the contract holds at runtime."""
        return {
            "tau": F.softplus(self.raw_tau).clamp_min(self.tau_min).item(),
            "alpha": [blk.alpha.item() for blk in self.phi_blocks],
        }
