"""median_context_age must read ONLY context slots: seeds are structurally age 0 and
padding is -1, so including either silently drags the scale toward zero."""
import torch

from link_property_prediction import walk_tokens as wt


class _FakeTokens:
    """Stands in for WalkTokens; median_context_age only touches ages/mask/seed_mask."""

    def __init__(self, ages, mask, seed_mask):
        self.ages, self.mask, self.seed_mask = ages, mask, seed_mask


def _patch(monkeypatch, toks):
    monkeypatch.setattr(wt, "build_query_walk_tokens", lambda *a, **k: toks)


def _call():
    return wt.median_context_age(None, torch.device("cpu"), torch.zeros(1), torch.zeros(1),
                                 max_walk_len=2, num_walks_per_node=1)


def test_excludes_seed_and_padding(monkeypatch):
    # context ages 10,20,30; a seed slot at 0 and a padding slot at -1 that must not count.
    ages = torch.tensor([[0, 10, 20, 30, -1]], dtype=torch.int64)
    mask = torch.tensor([[True, True, True, True, False]])
    seed = torch.tensor([[True, False, False, False, False]])
    _patch(monkeypatch, _FakeTokens(ages, mask, seed))
    assert _call() == 20.0          # median(10,20,30); 0 and -1 would drag it down


def test_drops_zero_gap_ties(monkeypatch):
    ages = torch.tensor([[0, 0, 0, 4, 8]], dtype=torch.int64)
    mask = torch.ones(1, 5, dtype=torch.bool)
    seed = torch.tensor([[True, False, False, False, False]])
    _patch(monkeypatch, _FakeTokens(ages, mask, seed))
    # median(4, 8) -> 4.0: torch.median is the lower order statistic on even counts.
    assert _call() == 4.0           # the two age-0 ties are dropped first


def test_returns_none_when_all_cold(monkeypatch):
    # Seed slot only: no context token exists, so there is no age distribution to measure.
    ages = torch.tensor([[0, -1, -1, -1, -1]], dtype=torch.int64)
    mask = torch.tensor([[True, False, False, False, False]])
    seed = torch.tensor([[True, False, False, False, False]])
    _patch(monkeypatch, _FakeTokens(ages, mask, seed))
    assert _call() is None                      # caller falls back to mean_node_inter_arrival
