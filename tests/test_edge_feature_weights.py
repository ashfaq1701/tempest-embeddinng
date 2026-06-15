"""Validation for learnable edge-feature weighting in GeometricVelocityPerWalkAvgHead.

Edge features re-weight the per-walk LS fit through a ZERO-INIT Linear added to the
recency log-weights, so they are a no-op at init and grow only if they lower the loss.

  1. no-op at init — different edge-feature tensors (and zeros) give byte-identical
     logits at init; output is [B,C] and finite.
  2. gradients reach E_u, E_v, the walk tokens AND edge_proj.weight (so it can learn),
     even though e=0 at init.
  3. masked/padded steps stay w=0 — a masked token gets zero gradient despite a
     nonzero edge feature (e is added BEFORE the mask fill).
  4. once edge_proj is nonzero, the edge features DO move the output (mechanism live).
"""
import torch

from tempest_walks.link_pred_head import GeometricVelocityPerWalkAvgHead

EDGE_DIM = 5


def _common(B, C, K, L, d, seed):
    torch.manual_seed(seed)
    return (torch.randn(B, K, L, d), torch.rand(B, K, L) * 3 + 0.1,
            torch.randn(B, d), torch.randn(B, C, d), torch.rand(B, C))


def test_edge_features_noop_at_init():
    B, C, K, L, d = 4, 3, 3, 5, 8
    h = GeometricVelocityPerWalkAvgHead(d_emb=d, edge_dim=EDGE_DIM)
    tok, age, E_u, E_v, rec = _common(B, C, K, L, d, 0)
    mask = torch.ones(B, K, L, dtype=torch.bool)
    A, Bf, Z = (torch.randn(B, K, L, EDGE_DIM), torch.randn(B, K, L, EDGE_DIM),
                torch.zeros(B, K, L, EDGE_DIM))
    oA = h(tok, age, mask, A, E_u, E_v, rec)
    oB = h(tok, age, mask, Bf, E_u, E_v, rec)
    oZ = h(tok, age, mask, Z, E_u, E_v, rec)
    assert oA.shape == (B, C) and torch.isfinite(oA).all()
    assert torch.equal(oA, oB) and torch.equal(oA, oZ), "edge features not a no-op at init"
    print(f"\n[noop] max|out(A)-out(B)|={ (oA-oB).abs().max():.2e}  "
          f"max|out(A)-out(zeros)|={(oA-oZ).abs().max():.2e}  (both want 0)")


def test_grads_reach_all_inputs_and_edge_proj():
    B, C, K, L, d = 4, 3, 3, 5, 8
    h = GeometricVelocityPerWalkAvgHead(d_emb=d, edge_dim=EDGE_DIM)
    torch.manual_seed(1)
    E_u = torch.randn(B, d, requires_grad=True)
    E_v = torch.randn(B, C, d, requires_grad=True)
    tok = torch.randn(B, K, L, d, requires_grad=True)
    tef = torch.randn(B, K, L, EDGE_DIM)
    age = torch.rand(B, K, L) * 3 + 0.1
    mask = torch.ones(B, K, L, dtype=torch.bool)
    rec = torch.rand(B, C)
    h(tok, age, mask, tef, E_u, E_v, rec).sum().backward()
    roles = {"E_u": E_u.grad, "E_v": E_v.grad, "tok": tok.grad,
             "edge_proj.weight": h.edge_proj.weight.grad}
    ok = {k: (v is not None and v.abs().sum().item() > 0) for k, v in roles.items()}
    print(f"\n[grads] {ok}")
    for k, good in ok.items():
        assert good, f"{k} received no gradient"


def test_masked_steps_zero_grad():
    B, C, K, L, d = 4, 3, 3, 6, 8
    h = GeometricVelocityPerWalkAvgHead(d_emb=d, edge_dim=EDGE_DIM)
    torch.manual_seed(2)
    tok = torch.randn(B, K, L, d, requires_grad=True)
    tef = torch.randn(B, K, L, EDGE_DIM)            # nonzero even on masked steps
    mask = torch.rand(B, K, L) > 0.5
    mask[:, :, 0] = True
    assert (~mask).any()
    age = torch.rand(B, K, L) * 3 + 0.1
    h(tok, age, mask, tef, torch.randn(B, d), torch.randn(B, C, d),
      torch.rand(B, C)).sum().backward()
    gper = tok.grad.abs().sum(-1)
    masked_max = gper[~mask].abs().max().item()
    print(f"\n[mask] valid min|grad|={gper[mask].min():.2e}  masked max|grad|={masked_max:.2e}")
    assert (gper[mask] > 0).all()
    assert masked_max < 1e-6, "masked step received gradient (edge term leaked past mask?)"


def test_edge_features_live_after_step():
    B, C, K, L, d = 4, 3, 3, 5, 8
    h = GeometricVelocityPerWalkAvgHead(d_emb=d, edge_dim=EDGE_DIM)
    with torch.no_grad():                           # simulate a learned edge_proj
        h.edge_proj.weight.normal_(0, 0.5)
        h.edge_proj.bias.normal_(0, 0.1)
    tok, age, E_u, E_v, rec = _common(B, C, K, L, d, 3)
    mask = torch.ones(B, K, L, dtype=torch.bool)
    oA = h(tok, age, mask, torch.randn(B, K, L, EDGE_DIM), E_u, E_v, rec)
    oB = h(tok, age, mask, torch.randn(B, K, L, EDGE_DIM), E_u, E_v, rec)
    diff = (oA - oB).abs().max().item()
    print(f"\n[live] max|out(A)-out(B)| with nonzero edge_proj = {diff:.2e} (want > 0)")
    assert diff > 0, "edge features have no effect even with nonzero edge_proj (dead path)"


if __name__ == "__main__":
    test_edge_features_noop_at_init()
    test_grads_reach_all_inputs_and_edge_proj()
    test_masked_steps_zero_grad()
    test_edge_features_live_after_step()
    print("\nALL EDGE-WEIGHT CHECKS PASSED")
