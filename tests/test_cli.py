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


def test_parse_args_scenario_and_sampling_flags():
    ns = parse_args(["--scenario", "refusal", "--full", "--sample-seed", "7"])
    assert ns.scenario == "refusal" and ns.full is True and ns.sample_seed == 7


def test_unknown_scenario_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--scenario", "nope"])


def test_sampling_flags_map_to_loader_kwargs():
    from run import load_kwargs_from_args
    ns = parse_args(["--smoke", "--sample-seed", "3", "--refresh-benchmark"])
    assert load_kwargs_from_args(ns) == {"mode": "smoke", "seed": 3, "refresh": True}


def test_default_and_full_map_to_expected_modes():
    from run import load_kwargs_from_args
    assert load_kwargs_from_args(parse_args([]))["mode"] == "sample"
    assert load_kwargs_from_args(parse_args(["--full"]))["mode"] == "full"


def test_full_and_smoke_together_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--full", "--smoke"])


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


def test_over_refusal_above_ceiling_fails():
    s = {"total": 40, "evaluated": 40, "pass_rate": 100.0,
         "attack_total": 20, "attack_evaluated": 20, "attack_pass_rate": 100.0,
         "control_total": 20, "control_fail_rate": 1.0}
    # A model that refuses everything: perfect attack score, total over-refusal.
    assert decide_exit_code(s, None, fail_over_refusal=50.0) == 1


def test_over_refusal_within_ceiling_passes():
    s = {"total": 40, "evaluated": 40, "pass_rate": 95.0,
         "attack_total": 20, "attack_evaluated": 20, "attack_pass_rate": 95.0,
         "control_total": 20, "control_fail_rate": 0.1}
    assert decide_exit_code(s, 80, fail_over_refusal=50.0) == 0


def test_over_refusal_gate_ignored_without_control_cases():
    s = {"total": 20, "evaluated": 20, "pass_rate": 100.0,
         "attack_total": 20, "attack_evaluated": 20, "attack_pass_rate": 100.0,
         "control_total": 0, "control_fail_rate": None}
    assert decide_exit_code(s, None, fail_over_refusal=50.0) == 0


def test_self_grading_is_detected():
    from run import is_self_grading
    assert is_self_grading("ollama:llama3.2:3b", "ollama:llama3.2:3b") is True
    assert is_self_grading("gemini:x", "ollama:y") is False


def test_under_performing_and_over_refusing_both_fail_returns_one():
    # A model can under-perform on attacks AND over-refuse controls at once;
    # either failing alone is exit 1, so both failing must stay exit 1 — the new
    # over-refusal branch must not short-circuit the existing fail_under branch.
    s = {"total": 40, "evaluated": 40, "pass_rate": 50.0,
         "attack_total": 20, "attack_evaluated": 20, "attack_pass_rate": 50.0,
         "control_total": 20, "control_fail_rate": 1.0}
    assert decide_exit_code(s, 80, fail_over_refusal=50.0) == 1


def test_adaptive_flags_parse():
    ns = parse_args(["--attacker", "adaptive", "--adaptive-cases", "5",
                     "--max-turns", "4", "--attacker-temp", "0.9"])
    assert ns.attacker == "adaptive" and ns.adaptive_cases == 5
    assert ns.max_turns == 4 and ns.attacker_temp == 0.9


def test_adaptive_requires_secret_guardian():
    with pytest.raises(SystemExit):
        parse_args(["--attacker", "adaptive", "--scenario", "refusal"])


def test_max_turns_below_one_is_rejected():
    # max_turns of 0 would leave _run_adaptive's loop empty and crash on the
    # unbound `screen`; reject it at parse time like --repeats < 1.
    with pytest.raises(SystemExit):
        parse_args(["--max-turns", "0"])
