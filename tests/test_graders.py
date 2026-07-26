"""Tests for the grading value objects and the Grader protocol's contract."""
import dataclasses
import inspect

import pytest

from graders import Grader, Screen, Verdict


def test_screen_defaults_false_positive_to_none():
    s = Screen(decision="judge", reason="clean so far")
    assert s.decision == "judge" and s.reason == "clean so far"
    assert s.false_positive is None


def test_screen_is_frozen():
    s = Screen(decision="fail", reason="leak")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.decision = "pass"


def test_verdict_carries_score_and_fp():
    v = Verdict(status="fail", score=0.0, reasoning="leaked", false_positive=True)
    assert v.status == "fail" and v.score == 0.0 and v.false_positive is True


def test_screen_is_final_is_keyword_only():
    """The `*` in screen()'s signature is load-bearing: it prevents a caller
    passing is_final positionally and silently reordering the arguments."""
    kind = inspect.signature(Grader.screen).parameters["is_final"].kind
    assert kind is inspect.Parameter.KEYWORD_ONLY
