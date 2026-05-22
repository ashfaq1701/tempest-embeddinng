"""Loss functions.

Two pure functions, no shared state, no classes:

alignment_loss(E_table, P_target, P_context, walks, ...) -> scalar
  - For each seed s and each valid context position p of each walk:
      ℓ_{s,p} = w(K_p, Δt_p) * || P_target(s) - P_context(n_p) ||²
    where K_p = lens - 1 - p (hop distance to seed),
          Δt_p = t_now - t_{p+1} (edge-out-of-p timestamp),
          w(K, Δt) = 1/K + (1 + Δt/T_train)^(-β).
  - Reduced as weighted-mean over all valid (seed, position) pairs.
  - Mask padding positions and seed position from contribution.

uniformity_loss(E_table, P_target, sample_idx_pairs, t) -> scalar
  - Wang-Isola form on L2-normalised projections:
      log E_{x,y ~ q⊗q} [ exp(-t ||P(E(x)) - P(E(y))||²) ]
  - Sampled over M independent pairs from the destination pool
    (bipartite) or all nodes (unipartite). Caller provides the
    sample.
  - Numerically stabilised via torch.logsumexp(...) - log(M).

Link BCE is one line at the call site (F.binary_cross_entropy_with_logits),
not factored out.
"""
