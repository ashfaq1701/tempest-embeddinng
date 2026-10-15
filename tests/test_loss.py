import torch

from link_property_prediction.loss import (
    popularity_ordered_radii,
    radial_popularity_loss,
    rank_agreement,
)


def test_target_is_a_permutation_of_the_radii():
    radii = torch.tensor([0.3, 0.9, 0.1, 0.5])
    popularity = torch.tensor([7.0, 2.0, 4.0, 9.0])
    target = popularity_ordered_radii(radii, popularity)
    assert torch.equal(target.sort().values, radii.sort().values)


def test_radius_falls_as_popularity_rises():
    radii = torch.tensor([0.3, 0.9, 0.1, 0.5])
    popularity = torch.tensor([7.0, 2.0, 4.0, 9.0])
    target = popularity_ordered_radii(radii, popularity)
    # popularity order 9 > 7 > 4 > 2 gets radii 0.1 < 0.3 < 0.5 < 0.9.
    assert torch.equal(target, torch.tensor([0.3, 0.9, 0.5, 0.1]))


def test_tied_popularity_shares_its_slots_and_keeps_the_mean():
    radii = torch.tensor([0.2, 0.4, 0.6, 0.8])
    popularity = torch.tensor([5.0, 0.0, 0.0, 9.0])
    target = popularity_ordered_radii(radii, popularity)
    # The two unpopular nodes split the two largest slots, 0.8 and 0.6.
    assert torch.allclose(target, torch.tensor([0.4, 0.7, 0.7, 0.2]))
    assert torch.isclose(target.mean(), radii.mean())


def test_loss_is_zero_when_radii_already_follow_popularity():
    radii = torch.tensor([0.1, 0.5, 0.9])
    popularity = torch.tensor([9.0, 4.0, 1.0])
    assert radial_popularity_loss(radii, popularity) == 0.0


def test_gradient_moves_radii_toward_their_target_only():
    radii = torch.tensor([0.9, 0.1], requires_grad=True)
    popularity = torch.tensor([9.0, 1.0])
    radial_popularity_loss(radii, popularity).backward()
    # The popular node sits outside its target (0.1) and is pulled in; the other is pushed out.
    assert torch.allclose(radii.grad, torch.tensor([0.8, -0.8]))


def test_rank_agreement_spans_minus_one_to_one():
    popularity = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert rank_agreement(torch.tensor([0.9, 0.7, 0.5, 0.1]), popularity) == 1.0
    assert rank_agreement(torch.tensor([0.1, 0.5, 0.7, 0.9]), popularity) == -1.0


def test_rank_agreement_is_zero_when_popularity_is_constant():
    assert rank_agreement(torch.tensor([0.1, 0.5, 0.9]), torch.zeros(3)) == 0.0
