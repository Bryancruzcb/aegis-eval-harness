"""Tests for the CI exit-code decision."""
from run import decide_exit_code


def _summary(total, evaluated, pass_rate):
    return {"total": total, "evaluated": evaluated, "pass_rate": pass_rate}


def test_passing_gate_returns_zero():
    assert decide_exit_code(_summary(10, 10, 80.0), fail_under=70) == 0


def test_failing_gate_returns_one():
    assert decide_exit_code(_summary(10, 10, 80.0), fail_under=90) == 1


def test_no_gate_returns_zero_regardless_of_rate():
    assert decide_exit_code(_summary(10, 10, 0.0), fail_under=None) == 0


def test_all_errored_is_inconclusive():
    # Cases existed but none could be evaluated -> non-zero even without a gate.
    assert decide_exit_code(_summary(5, 0, 0.0), fail_under=None) == 1


def test_empty_selection_does_not_trip_gate():
    # No cases matched the filter; that is not a quality-gate failure.
    assert decide_exit_code(_summary(0, 0, 0.0), fail_under=90) == 0
