"""Model components: embedding table, projection heads, link head.

Three classes, no shared state:

EmbeddingTable
  - Single nn.Embedding(num_nodes, d_emb).
  - Lookup-only. Trained by alignment+uniformity through the
    projection heads.

ProjectionHead
  - Conditional architecture based on feature availability:
      E only          → MLP on E
      E + NF          → MLP on E + MLP on NF, concat, merge MLP
      E + EF          → MLP on E + MLP on EF, concat, merge MLP
      E + NF + EF     → MLP on E + MLP on NF + MLP on EF, concat,
                        merge MLP
  - Output is L2-normalised via F.normalize(..., p=2, dim=-1, eps=1e-12).
  - Two instances: P_target (for seed/downstream nodes) and
    P_context (for walk-internal/upstream nodes). EF channel only
    appears in P_context per convention β.

LinkHead
  - Inputs: E[u].detach(), E[v].detach() — stop-grad on E.
  - Pair features: concat of [E(u), E(v), E(u)*E(v), |E(u)-E(v)|].
  - 2-layer GELU MLP, hidden = d_emb, output scalar logit.
  - Output is a logit, scored with BCEWithLogitsLoss at the call site.
  - No node features, no edge features, no time features at scoring.
  - Symmetric scoring for undirected eval: average score(u,v) and
    score(v,u).
"""
