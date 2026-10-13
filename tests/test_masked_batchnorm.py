import torch
import torch.nn as nn

from link_property_prediction.layers import MaskedBatchNorm


def _pair(num_features=3):
    # The reference is BatchNorm1d WITHOUT running stats -- batch statistics in both modes and
    # no buffers -- because that is exactly what MaskedBatchNorm is, plus the masking.
    torch.manual_seed(0)
    ours = MaskedBatchNorm(num_features)
    ref = nn.BatchNorm1d(num_features, track_running_stats=False)
    with torch.no_grad():
        ref.weight.copy_(torch.rand(num_features) + 0.5)
        ref.bias.copy_(torch.randn(num_features))
        ours.weight.copy_(ref.weight)
        ours.bias.copy_(ref.bias)
    return ours, ref


def test_matches_batchnorm1d_on_the_valid_entries():
    ours, ref = _pair()
    for step in range(5):
        feat = torch.randn(16, 25, 3, dtype=torch.float32) * (step + 1) + step
        valid = torch.rand(16, 25) > 0.4
        out = ours(feat, valid)
        torch.testing.assert_close(out[valid], ref(feat[valid]))
        assert (out[~valid] == 0).all()


def test_masked_entries_do_not_move_the_statistics():
    feat = torch.randn(4, 10, 3)
    valid = torch.rand(4, 10) > 0.5
    noisy = feat.clone()
    noisy[~valid] = 1e6
    ours_a, ours_b = MaskedBatchNorm(3), MaskedBatchNorm(3)
    torch.testing.assert_close(ours_a(feat, valid), ours_b(noisy, valid))


def test_state_dict_holds_only_the_affine():
    # No running_mean/running_var: forward never read them, so carrying them in the checkpoint
    # advertised a train/eval switch and a memory horizon the module does not have.
    assert sorted(MaskedBatchNorm(3).state_dict()) == ["bias", "weight"]
    assert not list(MaskedBatchNorm(3).buffers())
