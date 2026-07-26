"""Scenarios: a target setup plus the grader that scores it.

A Scenario is what the harness runs. Today there is one; Phase 2b adds a second.
"""
from dataclasses import dataclass
from typing import Callable

import target
from graders import Grader, SecretGuardianGrader
from refusal_grader import RefusalGrader
from benchmarks.jbb import load_jbb_cases


@dataclass(frozen=True)
class Scenario:
    """A target setup plus the grader that scores it.

    ``load_cases`` must return cases that are already validated and defaulted
    (as ``runner.load_test_cases`` does): ``run_suite`` filters on ``technique``
    and segments the summary on ``expect``, so a loader that omits them would
    raise rather than silently mis-report.
    """
    name: str
    system_prompt: str | None      # None => send the model no system prompt
    grader: Grader
    load_cases: Callable[..., list]


def _load_secret_guardian_cases(**_):
    """Load the bundled suite. Imported lazily to avoid a circular import.

    Accepts and ignores any keyword arguments: ``run_suite`` forwards the CLI's
    sampling options (``mode``/``seed``/``refresh``) to every scenario's loader,
    and those only mean something to the refusal benchmark. The ``**_`` catch-all
    keeps a stray sampling flag from crashing this scenario.
    """
    import json

    import config
    from runner import load_test_cases
    with open(config.BASE_DIR / "test_cases.json", "r", encoding="utf-8") as f:
        return load_test_cases(json.load(f))


SECRET_GUARDIAN = Scenario(
    name="secret-guardian",
    system_prompt=target.SYSTEM_INSTRUCTION,
    grader=SecretGuardianGrader(),
    load_cases=_load_secret_guardian_cases,
)

REFUSAL = Scenario(
    name="refusal",
    system_prompt=None,          # test the model's own safety training
    grader=RefusalGrader(),
    load_cases=load_jbb_cases,   # keyword args supplied by run_suite's caller
)

SCENARIOS: dict[str, Scenario] = {SECRET_GUARDIAN.name: SECRET_GUARDIAN,
                                  REFUSAL.name: REFUSAL}
