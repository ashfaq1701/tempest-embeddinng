"""Pin CrossAttentionLinkHead's contract.

Tests:
  1. Output shape [P].
  2. Finiteness for normal inputs.
  3. All-padded rows (no valid tokens) produce a finite logit (no NaN
     from softmax over an all-masked row).
  4. Masking actually matters: padded positions do NOT contribute to
     the attention output (verified by perturbing a padded token's
     value and confirming the score is unchanged).
  5. Asymmetry: forward(u, v, ...) differs from forward(v, u, ...).
  6. Gradient flows to h_u, h_v, AND tokens (the encoder's BCE pathway).
"""

import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from tempest_walks.model import CrossAttentionLinkHead


def _make_inputs(P=6, T=10, d=16, all_pad_row: int = None):
    """Synthetic inputs. Optionally mark row `all_pad_row` as having
    zero valid tokens on the U side to exercise the empty-mask path."""
    h_u = torch.randn(P, d, requires_grad=True)
    h_v = torch.randn(P, d, requires_grad=True)
    u_tokens = torch.randn(P, T, d, requires_grad=True)
    v_tokens = torch.randn(P, T, d, requires_grad=True)
    # Variable valid-token counts per row.
    u_valid = torch.arange(T).unsqueeze(0).expand(P, T) < torch.tensor(
        [3, 7, 1, 10, 5, 8][:P]
    ).unsqueeze(1)
    v_valid = torch.arange(T).unsqueeze(0).expand(P, T) < torch.tensor(
        [10, 4, 6, 2, 1, 9][:P]
    ).unsqueeze(1)
    if all_pad_row is not None:
        u_valid[all_pad_row] = False
    return h_u, h_v, u_tokens, u_valid, v_tokens, v_valid


def test_output_shape_and_finiteness():
    head = CrossAttentionLinkHead(d_emb=16, n_heads=4)
    h_u, h_v, u_t, u_m, v_t, v_m = _make_inputs(P=6, T=10, d=16)
    out = head(h_u, h_v, u_t, u_m, v_t, v_m)
    assert out.shape == (6,)
    assert torch.isfinite(out).all()


def test_all_padded_row_produces_finite_logit():
    """If a row has zero valid tokens on the U side, the v_reads_u
    cross-attention must NOT NaN. Implementation substitutes a zero
    output for such rows and skips the attention call."""
    head = CrossAttentionLinkHead(d_emb=16, n_heads=4)
    h_u, h_v, u_t, u_m, v_t, v_m = _make_inputs(P=6, T=10, d=16, all_pad_row=2)
    out = head(h_u, h_v, u_t, u_m, v_t, v_m)
    assert torch.isfinite(out).all(), (
        "all-padded row produced NaN — the attention should be skipped "
        "for rows with key_padding_mask all True"
    )


def test_padded_token_does_not_affect_score():
    """Perturbing the value of a PADDED v-token should not change the
    score for any pair — attention's key_padding_mask must zero out
    softmax weights at padded positions."""
    torch.manual_seed(0)
    head = CrossAttentionLinkHead(d_emb=16, n_heads=4).eval()
    h_u, h_v, u_t, u_m, v_t, v_m = _make_inputs(P=4, T=8, d=16)
    with torch.no_grad():
        s_before = head(h_u, h_v, u_t, u_m, v_t, v_m).clone()
        # Find a padded position in v on row 1 (v_valid for row 1 has
        # only the first 4 positions valid; positions 4..7 are padded).
        # Perturb v_tokens[1, 6, :] (definitely padded).
        v_t_perturbed = v_t.clone()
        v_t_perturbed[1, 6, :] = 100.0
        s_after = head(h_u, h_v, u_t, u_m, v_t_perturbed, v_m).clone()
    delta = (s_after - s_before).abs().max().item()
    assert delta < 1e-4, (
        f"padded v-token affected the score (max delta {delta:.6f}) — "
        f"key_padding_mask is leaking"
    )


def test_asymmetry():
    head = CrossAttentionLinkHead(d_emb=16, n_heads=4).eval()
    h_u, h_v, u_t, u_m, v_t, v_m = _make_inputs(P=4, T=8, d=16)
    with torch.no_grad():
        s_uv = head(h_u, h_v, u_t, u_m, v_t, v_m)
        s_vu = head(h_v, h_u, v_t, v_m, u_t, u_m)
    assert not torch.allclose(s_uv, s_vu, atol=1e-3), (
        "head is symmetric — caller relies on asymmetry "
        "(undirected eval averages forward(u, v) and forward(v, u))"
    )


def test_gradient_flow():
    head = CrossAttentionLinkHead(d_emb=16, n_heads=4)
    h_u, h_v, u_t, u_m, v_t, v_m = _make_inputs(P=4, T=8, d=16)
    out = head(h_u, h_v, u_t, u_m, v_t, v_m).sum()
    out.backward()
    assert h_u.grad is not None and h_u.grad.abs().sum().item() > 0, "h_u no grad"
    assert h_v.grad is not None and h_v.grad.abs().sum().item() > 0, "h_v no grad"
    assert u_t.grad is not None and u_t.grad.abs().sum().item() > 0, "u_tokens no grad"
    assert v_t.grad is not None and v_t.grad.abs().sum().item() > 0, "v_tokens no grad"
    head_grad_norms = [
        p.grad.norm().item() for p in head.parameters() if p.grad is not None
    ]
    assert any(g > 0 for g in head_grad_norms), "head params no grad"


if __name__ == "__main__":
    test_output_shape_and_finiteness()
    test_all_padded_row_produces_finite_logit()
    test_padded_token_does_not_affect_score()
    test_asymmetry()
    test_gradient_flow()
    print("OK: all CrossAttentionLinkHead tests passed")
