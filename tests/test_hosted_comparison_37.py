import pytest

from hosted_comparison_37 import accounted_cost, reserve, usage_cost


def test_prior_known_usage_settles_but_unknown_call_retains_allowance():
    known = {"reserved_usd": 4, "usage": {"prompt_token_count": 1000, "total_token_count": 1000}, "estimated_usd": .0015}
    unknown = {"reserved_usd": .9, "estimated_usd": None}
    ledger = {"attempts": [known, unknown]}
    reserve(ledger, 0, 4)
    assert sum(map(accounted_cost, ledger["attempts"])) == pytest.approx(4.9015)
    with pytest.raises(ValueError, match="budget"):
        reserve(ledger, 1, .1)
    assert known["reserved_usd"] == 4


@pytest.mark.parametrize("estimate", [None, -1, float("nan")])
def test_unreliable_cost_keeps_reservation(estimate):
    assert accounted_cost({"reserved_usd": 1, "usage": {"total_token_count": 1},
                           "estimated_usd": estimate}) == 1


def test_incomplete_usage_cannot_release_reservation():
    assert accounted_cost({"reserved_usd": 1, "usage": {"total_token_count": 1000},
                           "estimated_usd": 0}) == 1


def test_cost_above_reservation_counts_full_estimate():
    assert accounted_cost({"reserved_usd": 1, "estimated_usd": 8}) == 8


def test_37_price_includes_thinking():
    assert usage_cost({"prompt_token_count": 1000, "candidates_token_count": 100,
                       "thoughts_token_count": 900}) == pytest.approx(.0045)
