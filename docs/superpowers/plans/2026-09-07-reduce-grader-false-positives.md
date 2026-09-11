# Reduce grader false positives implementation plan

Continuation completed: the user authorized $5 total hosted API spending and later funded the API account. All 150 hosted-comparison rows were evaluated; 149 resolved and one was blocked by the provider. On the matched cohort, Qwen MCC 0.527 exceeded Gemini 3.5 Flash MCC 0.306, with lower FPR and FNR. No hosted adoption. Known token-cost equivalent is $0.214077; conservative reservations are $3.984035. XSTest human-labeled harmless response files have been located and inventoried. Final results and rationale are in `docs/hosted-judge-comparison.md`; earlier blocked-stage statements below are historical.

> **For agentic workers:** Execute task-by-task with regression tests and an independent review. The user has authorized creating and executing this plan in the current task.

**Goal:** Measure and reduce screening-induced false positives without hiding partial harmful compliance or increasing missed jailbreaks.

**Architecture:** Add a paired screening ablation to the existing comparison command. Force the unchanged judge to score every response once, reuse that decision for both paths when applicable, and compare it with the existing deterministic screen. Preserve the production baseline until evidence supports adoption.

**Tech stack:** Python 3.11, pytest, existing provider clients, pinned JBB calibration data, local Ollama Qwen.

**Spec:** The user-approved five-step proposal in this task: isolate screening, compare stronger judges, inspect development errors, track false negatives, and validate on fresh labels.

## Constraints and decision rules

- Work on `codex/reduce-grader-false-positives`; preserve the prior uncommitted work and source snapshots.
- Keep the current rubric and partial-compliance verdict unchanged during screening ablation.
- Start with the existing 150 development rows. Both old splits have been inspected; neither is fresh final validation.
- For this controlled comparison, use the same judge output for shared escalation cases. Route-specific judge counts and timing are counterfactual cost estimates, not two separately timed runs.
- Gate adoption on lower FPR, higher MCC, no higher FNR on development, no loss of coverage, and disclosed inference cost. One observed improvement is provisional, not proof of generalization.
- Hosted model comparisons require the user's budget answer and a configured compatible model. No new paid calls while that answer is pending. Available local models smaller than Qwen must not be called stronger without evidence.
- Fresh human labels cannot be manufactured from AI judgments. If unavailable, prepare the validation specification and report that generalization remains unverified.
- Save hashes/predictions/error types, not harmful prompt or response text, in committed results.

## Task 1: Paired screening comparison and resumable execution

**Files:** `compare_graders.py`, `tests/test_compare_graders.py`.

**Interfaces:** `evaluate(..., *, judge_only=False)`; `evaluate_screening_modes(...) -> dict` with `two_stage` and `judge_only` results; CLI `--screening-ablation`, `--resume`.

- [x] Write failing tests: force an automatic compliance decision through a fake judge returning refusal; assert screened prediction is `True`, judge-only is `False`, and exactly one judge invocation occurs. Repeat for an ambiguous response and verify both paths reuse the identical decision.
- [x] Implement bypass using `Screen(decision="judge", reason="screening ablation")`; bypass only screening, never the production verdict mapping.
- [x] Implement paired evaluation: run screened mode once; reuse its result if it reached the judge, otherwise perform one forced-judge call. Record unique logical judge evaluations separately from estimated route costs. Provider retries are not individually counted.
- [x] Add resume validation against source/prompt hashes, model/provider, dataset/split, mode, and row hashes. Complete artifacts are a no-op; changed configuration must fail before model calls.
- [x] Run the targeted tests, then launch the local comparison:

```powershell
.\venv\Scripts\python.exe -u compare_graders.py --variant baseline=output/false-positive-ablation/refusal_grader_baseline.py --screening-ablation --provider ollama --model qwen2.5:latest --output output/false-positive-ablation/qwen-dev.json
```

To continue after interruption, append `--resume` to the same command. Do not restart completed rows.

## Task 2: Error analysis and stronger-judge feasibility

**Files:** `docs/false-positive-ablation-results.json`, `docs/false-positive-ablation-walkthrough.md`.

- [x] Decompose disagreements into screen compliance overturned, screen refusal overturned, newly introduced errors, and unchanged shared judge mistakes. Report counts, MCC, FPR/FNR, coverage, and unique logical evaluations/estimated route counts.
- [x] Check the hosted budget response and available model configuration. If funded, verify current provider pricing, cap requests/output tokens and projected cost, then run the identical ablation with one stronger candidate. Otherwise document the missing access/budget and keep this stage unexecuted.
- [x] Identify development-only error patterns without treating the already inspected comparison half as a fresh holdout. Make at most one separately frozen rubric candidate if the screening/model comparison exposes a specific correctable pattern; do not change multiple factors together or tune to named benchmark rows.

## Task 3: Adopt or reject, review, and validation handoff

**Files:** existing grader/CLI/calibration only if adoption is supported; tests; walkthrough; `docs/false-positive-validation-protocol.md`.

- [x] Apply the decision rule. If screening removal helps, expose the measured mode consistently in runtime and calibration with regression coverage. Otherwise retain production and explain why.
- [x] Define a fresh labeled validation set covering harmful and benign responses, independent labeling/adjudication, frozen rubric/model, and report FPR/FNR/MCC with coverage. Existing 300 rows are diagnostic data from now on.
- [x] Run `python -m pytest -q` and `git diff --check`; request an independent read-only code/evidence review and fix actionable findings.
- [x] Save measured results, selection rationale, reproducible commands, and any genuinely blocked stage. Do not claim the original numeric targets unless attained. Leave GitHub publication and resume edits outside this task.

No commits will bundle the prior session's uncommitted work during this experiment. Source changes remain reviewable on the dedicated branch.

## Execution outcome

Local work completed. JBB development rejected judge-only adoption: MCC 0.510 to 0.526, FP 41 to 37, but FN 3 to 4. Separate JAILJUDGE ID validation completed 200/200 with no errors; unchanged production scored MCC 0.641, FPR 16%, FNR 20%. These are separate cohort results, not a production improvement. Full suite: 327 tests passed. Independent follow-up review: no remaining actionable findings.

The stronger hosted model comparison remains unexecuted because no API budget answer was received. Population-level harmless-response validation remains unexecuted because independent labels are unavailable. These outstanding stages are not implied complete by the feasibility and protocol checkboxes above. No new paid calls, production adoption, GitHub publication or resume edits were made in this experiment.
