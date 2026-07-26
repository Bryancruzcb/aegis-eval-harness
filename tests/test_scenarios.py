"""The scenario registry: both scenarios present, and every grader honours the
same screen contract regardless of which scenario owns it."""
from scenarios import SCENARIOS, SECRET_GUARDIAN


def test_both_scenarios_registered():
    assert set(SCENARIOS) == {"secret-guardian", "refusal"}


def test_refusal_scenario_sends_no_system_prompt():
    assert SCENARIOS["refusal"].system_prompt is None


def test_secret_guardian_unchanged():
    assert SCENARIOS["secret-guardian"] is SECRET_GUARDIAN


def test_every_scenario_honours_the_grader_contract():
    """One gate for all graders: is_final must be keyword-only, and a mid-loop
    clean screen must never decide the case."""
    import inspect
    for name, sc in SCENARIOS.items():
        kind = inspect.signature(type(sc.grader).screen).parameters["is_final"].kind
        assert kind is inspect.Parameter.KEYWORD_ONLY, name
        mid = sc.grader.screen({"expect": "refuse", "expected_criteria": "c"},
                               "a clean reply", "", is_final=False)
        assert mid.decision != "pass", name
