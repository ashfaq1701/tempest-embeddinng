"""Pin HybridLinkHead's contract.

HybridLinkHead consumes 2*d_emb-dim inputs per side (caller concatenates
E[v].detach() with h_v). Tests:
  1. Output shape: matches input batch shape (no trailing dim).
  2. Detach invariant: gradient does NOT flow back to the E branch
     when the caller passes E.detach() (verified by checking that
     E.weight.grad is None after backward through the link head).
  3. h branch DOES receive gradient (the encoder's contribution must
     train).
  4. Bilinear is asymmetric (forward(eh_u, eh_v) != forward(eh_v, eh_u)).
"""

import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn

from tempest_walks.model import EmbeddingTable, HybridLinkHead


def test_output_shape():
    d_emb = 8
    P = 13
    head = HybridLinkHead(d_emb=d_emb)
    eh_u = torch.randn(P, 2 * d_emb)
    eh_v = torch.randn(P, 2 * d_emb)
    out = head(eh_u, eh_v)
    assert out.shape == (P,), f"expected ({P},), got {tuple(out.shape)}"
    assert torch.isfinite(out).all()


def test_detach_invariant_on_E_side():
    """When the caller passes E.detach() concatenated with a live h,
    backward should not populate E.weight.grad — but should populate
    h's source (here a plain Parameter)."""
    d_emb = 8
    num_nodes = 10
    P = 5
    E_table = EmbeddingTable(num_nodes=num_nodes, d_emb=d_emb)
    h_source = nn.Parameter(torch.randn(P, d_emb))   # stands in for encoder output
    head = HybridLinkHead(d_emb=d_emb)

    u_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    v_ids = torch.tensor([5, 6, 7, 8, 9], dtype=torch.long)
    e_u = E_table(u_ids).detach()
    e_v = E_table(v_ids).detach()
    eh_u = torch.cat([e_u, h_source], dim=-1)
    eh_v = torch.cat([e_v, h_source], dim=-1)
    out = head(eh_u, eh_v).sum()
    out.backward()

    assert E_table.E.weight.grad is None or torch.equal(
        E_table.E.weight.grad, torch.zeros_like(E_table.E.weight.grad),
    ), "E.weight received gradient through detached input"
    assert h_source.grad is not None and h_source.grad.abs().sum().item() > 0, (
        "h branch did not receive gradient — link head is broken on h side"
    )
    head_grad_norms = [
        p.grad.norm().item() for p in head.parameters() if p.grad is not None
    ]
    assert any(g > 0 for g in head_grad_norms), "head params no grad"


def test_asymmetry():
    """Bilinear is asymmetric; forward(u, v) should differ from
    forward(v, u). Undirected eval must average at the caller."""
    d_emb = 8
    head = HybridLinkHead(d_emb=d_emb)
    eh_u = torch.randn(3, 2 * d_emb)
    eh_v = torch.randn(3, 2 * d_emb)
    s_uv = head(eh_u, eh_v)
    s_vu = head(eh_v, eh_u)
    assert not torch.allclose(s_uv, s_vu, atol=1e-4), (
        "head is unexpectedly symmetric — caller may rely on asymmetry"
    )


if __name__ == "__main__":
    test_output_shape()
    test_detach_invariant_on_E_side()
    test_asymmetry()
    print("OK: all HybridLinkHead tests passed")
