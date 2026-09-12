# GPT 5.6 Sol Ultra — review brief

Closed. Grader-quality review is in `HANDOFF-grok-gpt.md`. For the package
layout, use `docs/superpowers/plans/GPT-REVIEW-layout.md` instead.

Review diffs against `docs/superpowers/plans/2026-09-11-ml-engineer-program.md`.
Write findings into the GPT section of `docs/superpowers/plans/HANDOFF-grok-gpt.md`.

## Do not

- Mix MCC across contracts. 0.20 / 0.38 / 0.77 are the old refusal-as-jailbreak
  contract on different judges, not a grader upgrade.
- Adopt `SafetyGrader` from the 150-row MCC bump (dev MCC 0.510 → 0.777, FN 3 → 5).
  The gate is predeclared; FNR rose.
- Wipe `output/grader-quality/{dev,fresh}.json` (or `harmless.json`). Resume.
- Start a second `quality_eval.py` while `run.lock` is held.
- Print or commit JAILJUDGE / XSTest prompt text.

## Facts already established (do not re-litigate)

- Fresh row 48: candidate `JudgeParseError`; object was missing `reasoning`.
  Hash `2b492e01…` (79 chars). `json_object` retries were identical. Ollama
  `json_schema` unblocked it. See `docs/grader-quality-walkthrough.md`.
- Fresh row 80: candidate `TimeoutError` at 120 s. `JUDGE_TIMEOUT` bumped to
  180 (runtime is unfrozen; resume identity still passes).
- `--positive-control` uses the **live** attacker vs a leaky bot. Failed control
  exits 3 and withholds Compromise Rate. Offline tests stub the same path.
- README one-sentence product definition is at the top. README does not present
  0.20 / 0.38 / 0.77 as a grader upgrade.

## What to review

1. Parse retry + fence strip + schema pin (Phase 0) vs plan Tasks 0.1–0.4.
2. Decode-only identity amendment: `quality_eval.py` / `safety_grader.py` /
   `evaluators.py` hashes may change; prompt hashes must not.
3. Reporting: Compromise Rate withheld when `positive_control_passed` is false
   (HTML + terminal).
4. `fresh` is complete. `adopt_candidate` is **false** on both `dev` and `fresh`.
   Confirm the gate: FNR rose (dev 3→5, fresh 13%→26%). Fresh MCC interval
   includes 0. Do not drop the FNR clause to make MCC look good.
