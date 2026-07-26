# AegisEval Phase 2a — The Scenario refactor

- **Date:** 2026-07-25
- **Status:** Draft for review
- **Phase:** 2a of 3 (2b = refusal benchmark; 3 = adaptive attacker)

## In plain terms

The harness tests one situation: *can a support bot be tricked into revealing its password?*
That situation is glued into the machine — spread across four files — so the harness can't test
anything else.

This phase takes it out of the guts and puts it in a **box**. The machine goes from "I test the
password bot" to "hand me a box, I'll run whatever is in it." Then we hand it the same password
test, in a box.

**Nothing new works when this is done.** Same tests, same results, same report. That is the
point: a change that is supposed to alter nothing is a change you can actually verify. If any
existing test's expected result has to change, the refactor is wrong.

One thing does get added: a missing test. Today nothing exercises a multi-turn *harmless* case,
which is exactly the combination this refactor could break — so the "our tests will catch it"
claim wasn't true until we write it.

## 1. Why this is its own phase

The original Phase 2 spec bundled this refactor with the refusal benchmark. A review found that
the refactor's stated safety net — "the 101 tests will catch any behavior change" — **is false**:
every multi-turn case in `test_cases.json` is `expect: "refuse"` and every benign fixture in
`tests/test_runner.py` is single-turn, so a multi-turn benign case (the combination most likely
to break) is untested. Splitting lets 2a prove behavior preservation on its own, with its own
review and its own PR, before any new capability lands on top.

## 2. Goals & non-goals

### Goals
1. `Scenario` and `Grader` exist; `secret-guardian` is one instance of each.
2. **Zero user-visible behavior change.** Same results, same JSON keys, same report.
3. All 101 existing tests pass **with no assertion changes** (import/call-shape edits only).
4. Close the multi-turn-benign test gap so the safety net is real.
5. Fix two latent defects the review surfaced that would silently break 2b (below).

### Non-goals (all deferred to 2b)
The JBB loader, the refusal grader, calibration, `expect: "comply"`, new summary keys, reporter
changes, and the judge-delimiter hardening. **No new CLI flag ships in 2a** — `--scenario` is
added in 2b when there is a second scenario to select.

## 3. Design

### 3.1 `Grader` protocol + `Screen` / `Verdict`

```python
@dataclass(frozen=True)
class Screen:
    decision: Literal["pass", "fail", "judge"]
    reason: str
    false_positive: bool | None = None

@dataclass(frozen=True)
class Verdict:
    status: Literal["pass", "fail"]
    score: float | None
    reasoning: str
    false_positive: bool | None = None

class Grader(Protocol):
    def screen(self, case: dict, bot_text: str, attacker_text: str, *,
               is_final: bool) -> Screen: ...
    def judge_system_prompt(self, case: dict) -> str: ...
    def verdict(self, case: dict, screen: Screen, judged: dict | None) -> Verdict: ...
```

**`is_final` is mandatory, and this is the blocker fix.** The current loop acts on exactly one
per-turn signal — a deterministic *fail* returns early (`runner.py:129-136`) — while the
"benign passes without a judge" decision happens *after* the loop ends (`runner.py:138`), keyed
on `expect`. Without turn context, `screen` cannot distinguish "clean, keep going" from "clean,
we're done", and an implementer honoring a mid-loop `pass` would silently stop multi-turn benign
cases after turn 1.

**Contract:** mid-loop (`is_final=False`) only `fail` is actionable; `pass` and `judge` mean
"continue". On the final turn all three are honored. This is stated in the runner's docstring
and pinned by the new test in §5.

### 3.2 `scenarios.py` (new)

```python
@dataclass(frozen=True)
class Scenario:
    name: str
    system_prompt: str | None
    grader: Grader
    load_cases: Callable[..., list[dict]]

SCENARIOS: dict[str, Scenario] = {"secret-guardian": SECRET_GUARDIAN}
```

`system_prompt: str | None` — `None` means send the model no system prompt at all. Unused in 2a
(secret-guardian always has one) but the target must honor it, because 2b's refusal scenario
depends on it. Verified feasible on both providers: `GenerateContentConfig.system_instruction`
is `Optional` in the installed `google-genai`, and the OpenAI/Ollama path simply skips the
prepend.

`SECRET_GUARDIAN.load_cases` wraps today's `load_test_cases`, so `test_cases.json` loading is
unchanged.

### 3.3 What moves — and the constraint that keeps the tests honest

**`run_deterministic_eval` stays in `evaluators.py`, unchanged, returning the same dict.**
It is a public, directly-tested function: 12 tests assert on its `passed` / `score` / `reasoning`
/ `secret_leak` / `profanity` keys. `SecretGuardianGrader.screen` **calls** it and composes the
echo guard and the `false_positive` label on top. Converting it into a `Screen` would force
rewriting those 12 assertions — exactly what goal 3 forbids.

The runner keeps owning the two accumulators (`running_bot`, `attacker_text`); `screen` receives
them as arguments, which reproduces `runner.py:128` exactly. `config.SYSTEM_SECRET` is read at
grader construction, and `evaluators.run_deterministic_eval` continues to read it at call time —
a single source, no duplication, because the grader passes nothing secret-related down.

`target.SYSTEM_INSTRUCTION` moves into `SECRET_GUARDIAN.system_prompt`;
`query_target_conversation` gains the prompt as a parameter.

### 3.4 Two latent defects fixed here (they would silently break 2b)

1. **`_base_fields` is a whitelist** (`runner.py:67-80`) copying exactly seven keys — so
   `technique` is *already* silently dropped from every result today, and any future per-case
   metadata would be too. Add `technique`, and add a documented `extra` passthrough for
   scenario-specific keys. Covered by a test asserting `technique` survives into the result.
2. **New parameters must be keyword-only.** `target.py:117-119` and `evaluators.py:226` forward
   positionally; inserting a parameter mid-signature would silently bind `provider` to it — no
   `TypeError`, just a wrong system prompt and a confusing `ProviderError`. Every new parameter
   added in 2a is `*`-keyword-only, and both wrapper call sites are converted to keywords.

Also: screen-decided results keep `eval_type = "deterministic"`, because
`aggregate_repeats`' take-worst tiebreak hardcodes that string (`runner.py:189`). A new
`eval_type` would silently degrade `--repeats` aggregation with no failing test.

### 3.5 Threading

`run_single_test`, `run_case_repeated`, `run_suite`, and the loader take a `scenario` parameter
**defaulting to `SCENARIOS["secret-guardian"]`**, so every existing call site — including the
~31 bare calls in the test suite — keeps working untouched. Only the ~31 injected `judge`/`target`
doubles in `tests/test_runner.py` need signature edits, for the new keyword-only parameters.

## 4. What does NOT change

`build_summary` and every summary key; `decide_exit_code`; the reporter template and
`print_terminal_summary`; `test_cases.json`; the CLI surface; `requirements.txt`.

## 5. Testing

The existing 101 tests are the contract. **If a Phase 1 behavioral test needs its assertions
changed, stop — that means the refactor altered behavior.** Signature edits to injected doubles
are expected and permitted.

New tests:
1. **The gap-closer:** a multi-turn `expect: "benign"` case whose replies are clean throughout →
   runs *all* turns (assert transcript length), and is decided only at the end. This is the case
   no existing test covers and the one the `is_final` contract exists for.
2. A multi-turn benign case that trips the deterministic check on turn 2 → fails at turn 2 with
   the partial transcript (proves `fail` is still actionable mid-loop).
3. `Screen`/`Verdict` construction and `SecretGuardianGrader.screen` behavior, table-driven,
   including the echo guard and the `false_positive` label — the same cases as the current runner
   tests, now at the grader level.
4. `technique` survives into the result dict (the `_base_fields` fix).
5. `target` sends no system message on either provider path when `system_prompt=None`.
6. Passing a scenario explicitly and omitting it produce identical results for the same case.

## 6. Risks

- **Silent behavior change during the move.** Mitigated by the "no assertion changes" rule, the
  new multi-turn benign tests, and keeping `run_deterministic_eval` intact.
- **Positional-argument binding.** Mitigated by keyword-only parameters (§3.4).
- **Scope creep into 2b.** The non-goals list is explicit; any `expect: "comply"`, loader, or
  reporter work in a 2a diff is a review rejection.

## 7. Acceptance criteria

- `Scenario` and `Grader` exist; `SCENARIOS["secret-guardian"]` is the default everywhere.
- A full run produces byte-identical results to `main` for the same seed and inputs (verify by
  running the suite against a stub target before and after).
- All 101 existing tests pass **with zero assertion changes**.
- The six new tests in §5 pass, including the multi-turn benign case.
- `technique` appears in result dicts; new parameters are keyword-only.
- No new CLI flag, no summary-key change, no reporter change.

## 8. Scope

~5 tasks: `Screen`/`Verdict`/`Grader` protocol → `SecretGuardianGrader` (composing, not
replacing, `run_deterministic_eval`) → `Scenario` + target/runner threading with defaults →
`_base_fields` + keyword-only fixes → the multi-turn benign tests and parity check.
