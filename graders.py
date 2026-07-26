"""Grading interfaces.

A Grader owns every scenario-specific grading decision, so the runner can drive
any scenario without branching on it.

``screen`` runs per assistant turn and is cheap and network-free. ``is_final``
tells it whether the conversation is over, because mid-loop only a ``fail`` is
actionable: a clean turn means "keep going", not "we are done".
"""
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class Screen:
    """The cheap check's answer for one turn.

    decision:
        "fail"  -> the case failed now; stop (honored on every turn)
        "pass"  -> the case passed; only honored when is_final
        "judge" -> undecided; only honored when is_final, else means "continue"
    """
    decision: Literal["pass", "fail", "judge"]
    reason: str
    false_positive: bool | None = None


@dataclass(frozen=True)
class Verdict:
    """The final graded outcome for one run of a case."""
    status: Literal["pass", "fail"]
    score: float | None
    reasoning: str
    false_positive: bool | None = None


class Grader(Protocol):
    def screen(self, case: dict, bot_text: str, attacker_text: str, *,
               is_final: bool) -> Screen:
        """Cheap, network-free check over the conversation so far."""
        ...

    def judge_system_prompt(self, case: dict) -> str:
        """System instruction for the LLM judge on this scenario."""
        ...

    def verdict(self, case: dict, screen: Screen, judged: dict | None) -> Verdict:
        """Final outcome, given the screen and the judge's answer (None if not judged)."""
        ...
