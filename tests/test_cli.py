"""Tests for the CI exit-code decision."""
import pytest

from run import parse_args, decide_exit_code


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


def test_parse_args_new_flags():
    ns = parse_args(["--repeats", "3", "--technique", "crescendo", "--target-temp", "0.7"])
    assert ns.repeats == 3 and ns.technique == "crescendo" and ns.target_temp == 0.7


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_repeats_below_one_is_rejected(bad):
    # repeats < 1 would produce an empty run list and crash aggregation.
    with pytest.raises(SystemExit):
        parse_args(["--repeats", bad])


def test_gate_uses_attack_pass_rate():
    s = {"total": 10, "evaluated": 10, "pass_rate": 90.0,
         "attack_total": 8, "attack_evaluated": 8, "attack_pass_rate": 75.0}
    assert decide_exit_code(s, 80) == 1        # attack 75 < 80 even though global 90
    assert decide_exit_code(s, 70) == 0


def test_benign_only_selection_not_false_failure():
    s = {"total": 6, "evaluated": 6, "pass_rate": 100.0,
         "attack_total": 0, "attack_evaluated": 0, "attack_pass_rate": None}
    assert decide_exit_code(s, 80) == 0


def test_gate_rate_helper_is_the_number_the_gate_used():
    # the printed rate and the exit code must come from one source
    from run import _gate_rate, _gate_eval
    s = {"total": 10, "evaluated": 10, "pass_rate": 90.0,
         "attack_total": 8, "attack_evaluated": 8, "attack_pass_rate": 75.0}
    assert _gate_rate(s) == 75.0 and _gate_eval(s) == 8
    assert decide_exit_code(s, _gate_rate(s) + 1) == 1
    legacy = _summary(4, 4, 50.0)
    assert _gate_rate(legacy) == 50.0 and _gate_eval(legacy) == 4


def test_judge_outage_exits_one():
    s = {"total": 14, "evaluated": 6, "pass_rate": 100.0,
         "attack_total": 6, "attack_evaluated": 0, "attack_pass_rate": None}
    assert decide_exit_code(s, None) == 1
