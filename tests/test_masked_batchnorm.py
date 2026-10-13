import torch
import torch.nn as nn

from link_property_prediction.layers import MaskedBatchNorm


def _pair(num_features=3):
    torch.manual_seed(0)
    ours, ref = MaskedBatchNorm(num_features), nn.BatchNorm1d(num_features)
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
        expected = ref(feat[valid])
        torch.testing.assert_close(out[valid], expected)
        assert (out[~valid] == 0).all()
    torch.testing.assert_close(ours.running_mean, ref.running_mean)
    torch.testing.assert_close(ours.running_var, ref.running_var)

    # Eval normalises by the BATCH statistics as well, i.e. BatchNorm1d with
    # track_running_stats=False -- NOT by the running averages. That is the one variable.
    free = nn.BatchNorm1d(3, track_running_stats=False)
    with torch.no_grad():
        free.weight.copy_(ref.weight)
        free.bias.copy_(ref.bias)
    ours.eval()
    feat = torch.randn(8, 25, 3)
    valid = torch.rand(8, 25) > 0.4
    torch.testing.assert_close(ours(feat, valid)[valid], free(feat[valid]))


def test_train_and_eval_agree():
    ours, _ = _pair()
    feat = torch.randn(8, 25, 3)
    valid = torch.rand(8, 25) > 0.4
    ours.train()
    a = ours(feat, valid)
    ours.eval()
    b = ours(feat, valid)
    torch.testing.assert_close(a, b)


def test_masked_entries_do_not_move_the_statistics():
    feat = torch.randn(4, 10, 3)
    valid = torch.rand(4, 10) > 0.5
    noisy = feat.clone()
    noisy[~valid] = 1e6
    ours_a, ours_b = MaskedBatchNorm(3), MaskedBatchNorm(3)
    torch.testing.assert_close(ours_a(feat, valid), ours_b(noisy, valid))
    torch.testing.assert_close(ours_a.running_var, ours_b.running_var)
