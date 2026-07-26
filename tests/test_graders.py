import pytest

from graders import Screen, Verdict


def test_screen_defaults_false_positive_to_none():
    s = Screen(decision="judge", reason="clean so far")
    assert s.decision == "judge" and s.reason == "clean so far"
    assert s.false_positive is None


def test_screen_is_frozen():
    s = Screen(decision="fail", reason="leak")
    with pytest.raises(Exception):
        s.decision = "pass"


def test_verdict_carries_score_and_fp():
    v = Verdict(status="fail", score=0.0, reasoning="leaked", false_positive=True)
    assert v.status == "fail" and v.score == 0.0 and v.false_positive is True
