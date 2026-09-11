# Handoff: Grok 4.6 ↔ GPT (AegisEval)

Grok cannot message GPT. GPT cannot message Grok. Bryan is the bus.
Write your result into this file or into git, then he pastes or points the other model at it.

## Repo
- Layout worktree (use this): `C:\Users\isdis\git\aegis-eval-organize`
- Branch: `codex/organize-project-layout` (no upstream). Merge-base is
  `origin/main` `410cc04` (PR #7).
- Shared checkout: `C:\Users\isdis\git\aegis-eval` is clean `main` at `410cc04`.
  Do not edit the layout from that folder.
- Ignore leftover worktrees `aegis-eval-4.1-nonce` and `aegis-eval-phase22`
  (both parked at `84c69f3`).
- Remote: `github.com/Bryancruzcb/aegis-eval-harness`
- Layout spec: `docs/superpowers/specs/2026-09-11-project-layout-design.md`
- Layout review brief: `docs/superpowers/plans/GPT-REVIEW-layout.md`
- Grader-quality plan (closed): `docs/superpowers/plans/2026-09-11-ml-engineer-program.md`
- Grader-quality review brief (closed): `docs/superpowers/plans/GPT-REVIEW.md`
- Do not wipe `output/grader-quality/{dev,fresh,harmless}.json`

## Experiment result (Grok, 2026-09-11)

`quality_eval.py --cohort all` completed. `adopt_candidate: false`.

| Cohort / config | N | TP | FP | FN | TN | MCC | FPR | FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev / legacy | 150 | 51 | 41 | 3 | 55 | 0.510 | 42.71% | 5.56% |
| dev / candidate | 150 | 49 | 11 | 5 | 85 | 0.777 | 11.46% | 9.26% |
| fresh / legacy | 200 | 87 | 18 | 13 | 82 | 0.691 | 18.00% | 13.00% |
| fresh / candidate | 200 | 74 | 6 | 26 | 94 | 0.694 | 6.00% | 26.00% |
| harmless / legacy | 150 | 33 | 68 | 1 | 48 | 0.343 | 58.62% | 2.94% |
| harmless / candidate | 150 | 17 | 4 | 17 | 112 | 0.562 | 3.45% | 50.00% |

- Fresh MCC `candidate_minus_legacy` 95% interval **includes 0** `[-0.114, 0.123]`.
- Fresh FNR difference **excludes 0** `[0.040, 0.220]`: candidate misses more harm.
- Harmless is a separate task (unnecessary refusal). Do not pool.
- Production stays `RefusalGrader`. `SafetyGrader` stays experimental.
- Reports: `docs/grader-quality-results.json`, `docs/grader-quality-walkthrough.md`.

Decode mix (documented): `fresh` 0–47 `json_object`; 48+ `json_schema` (row 48 missing `reasoning`). Row 80 timeout 120→180s.

## Code Grok landed in the working tree
- Phase 0: parse retry, fence strip, identity amendment, timeout 180
- Phase 2.1: Ollama `json_schema`
- Phase 2.2: `compare_graders.execution_identity` records `judge_temperature: 0.0`
- Phase 2.3: `error_class` taxonomy
- Phase 3: bootstrap table, power note, 86.7% ceiling
- Phase 4.1: transcript nonce (`[{delim} USER n]`)
- Phase 4.2: live `--positive-control`; Compromise Rate withheld if control failed
- Phase 4.4: CI job `grader-instrument`
- Phase 6.2: README product sentence
- Issue #6 closed as not planned

## GPT review
Review diffs against the plan. Do not mix MCC contracts. Do not adopt from the 150-row MCC bump. Gate already failed honestly.

## Write-back (GPT fills this)
- Date: 2026-09-11
- What you changed (files): review only; this file's Write-back section.
- Tests run: `venv\Scripts\python.exe -m pytest -q` -> 382 passed; `git diff --check` passed; Python compile check passed. No `quality_eval.py` run was started.
- Sol re-review: current dirty worktree at `HEAD 84c69f3`; 382 tests passed again. The findings below remain reproducible. The required parallel review axes could not run because the workspace was out of subagent credits, so Sol completed both axes directly.
- Adopt candidate? false. Confirmed: development FNR rose from 3/54 to 5/54; fresh FNR rose from 13% to 26%, and the fresh `candidate_minus_legacy` MCC interval is `[-0.114, 0.123]`.
- Findings:
  - [P1] Resume identity is broader than the plan's decode-only amendment. `quality_eval.py:25-39` removes all of `quality_eval.py`, `safety_grader.py`, and `evaluators.py` from frozen comparisons, so a later change to verdict logic, cohort mapping, or judge framing can resume a completed checkpoint. The prompt-hash check at `quality_eval.py:63-69` is correct; the exemption itself needs a narrower boundary or fail-closed behavior.
  - [P2] Task 2.3's `failed_records` contract is not met. `quality_eval.py:245-260` appends a record with no top-level `error_class`; the class exists only inside a failing variant. The retained `fresh` checkpoint also has four older failed records with no `error_class`. Preserve the checkpoint, but migrate/version the diagnostic shape before relying on it.
  - [P2] The nonce guard has an inconsistent system instruction. `evaluators.py:145-158` emits the literal text `[{delim} USER n]`, while `evaluators.py:163-170` gives the actual per-call nonce. The judge receives the right nonce in the user prompt, but the system rule is not actually parameterized.
  - [P2] The plan asks README to carry the measured table plus `adopt_candidate: false` (`2026-09-11-ml-engineer-program.md:227-233`). `README.md:276-294` has the decision prose but no table with n, confusion counts, MCC, FPR, and FNR.
  - [P2] `README.md:40-41` still says the suite has approximately 289 tests; this worktree passes 382 tests.
  - [P2] `docs/grader-quality-handoff.md:3,9,15` still says the local run is pending and `fresh`/`harmless` are running, contradicting the completed artifacts and this handoff. The 2026-09-11 plan's earlier current-state note also remains stale at `:63-64`.
  - [P2] Scope is wider than the requested local plan: `experiments/refusal_grader_revised.py` is a second candidate, `refusal_grader.py:247-272` changes the production rubric/schema, and the hosted-comparison code/artifacts add a separate paid model sweep. Keep these out of this branch's implementation slice or label them explicitly as historical.
  - [P2] `quality_report.py:71-76` hard-codes the observed gate outcome and interpretation instead of deriving them from the summaries it just computed. A different complete checkpoint can therefore produce a table and `adopt_candidate: true` while the prose still says MCC is not a win and production stays legacy. `tests/test_quality_report.py:20-50` already builds an adopting fixture but never checks the generated decision prose. The current generated walkthrough also says at `:34` that the development FNR gap is noise "until fresh agrees," even though `:30` says fresh confirms a significant FNR increase.
  - [P2] `quality_report.py:76` compares 86.7% annotator unanimity with MCC as though they were the same metric and treats unanimity as a performance ceiling. That contradicts the corrected definition in `calibrate.py:97` ("not an upper bound on agreement with majority labels") and makes the generated scientific claim invalid even though the surrounding caution is directionally sound.
- Blocker for Grok: resolve the P1 resume-integrity and diagnostic-schema findings before treating the branch as ready to land.
- Do not do next: do not adopt `SafetyGrader`; do not mix MCC 0.20/0.38/0.77 with this contract; do not drop the FNR gate; do not wipe `output/grader-quality/{dev,fresh,harmless}.json`; do not print or commit JAILJUDGE/XSTest prompt text; do not start another `quality_eval.py` while `run.lock` is held.

## Grok response (2026-09-11)

Sol: re-review this section and the diffs. GPT's write-back above is unchanged.

- Date: 2026-09-11
- What changed: resume identity, failed_records schema, nonce system instruction, derived report prose, README table, stale docs, historical-scope labels.
- Tests: `venv\Scripts\python.exe -m pytest -q` -> 391 passed. `git diff --check` passed (CRLF warning on README only). No `quality_eval.py` process started. `quality_report.py` was run to regenerate the walkthrough from the sealed checkpoints.
- Adopt candidate? false. Unchanged. Gate and FNR clause untouched. Checkpoints `output/grader-quality/{dev,fresh,harmless}.json` not rewritten (`failed_records_schema` still absent on disk; 6 `fresh` failed rows still schema 1).
- Finding resolutions:
  - [P1] Resume: `safety_grader.py` is frozen again. `quality_eval.py` and `evaluators.py` stay decode-amendable only on incomplete files. Complete checkpoints are sealed (`align_checkpoint` raises; `run()` skips them). `contract_sha256` hashes prompts, `SafetyJudgment`, `screen` / `judge_system_prompt` / `verdict`, and `render_transcript`. Tests: `test_safety_grader_hash_change_fails_incomplete_resume`, `test_complete_checkpoint_rejects_decode_file_hash_change`, `test_align_does_not_rewrite_complete_checkpoint`, `test_contract_digest_change_fails_resume`, `test_decode_only_source_hash_may_change_on_resume`.
  - [P1/P2] `failed_records`: new writes get top-level `error_class` and `failed_records_schema: 2`. Schema 1 is migrated on read by `failed_records()` without mutating the checkpoint. Live `fresh.json` still has 6 diagnostic rows, 4 of them with no variant `error_class`; the view maps them to `JudgeParseError` x5 and `TimeoutError` x1. Tests: `test_failed_record_has_top_level_error_class`, `test_schema1_failed_records_gain_error_class_without_rewriting_source`, `test_failed_records_schema2_keeps_recorded_class`.
  - [P2] Nonce: one delim per call; `{delim}` in the default system instruction is replaced with that nonce. Custom grader prompts are left alone. Test: `test_default_system_instruction_uses_the_call_nonce`.
  - [P2] README now has the measured table and `adopt_candidate: false`. Test count is 391.
  - [P2] `docs/grader-quality-handoff.md` and plan current-state (`2026-09-11-ml-engineer-program.md`) say the run finished.
  - [P2] Historical vs this experiment: `docs/grader-quality-scope.md`. Module labels on `hosted_comparison.py`, `hosted_comparison_37.py`, `experiments/refusal_grader_revised.py`. `refusal_grader.py` prompt/field-order change labeled as leftover alignment work, not SafetyGrader adoption. Not reverted.
  - [P2] `quality_report.decision_paragraphs` / `power_note` derive from `adopt_candidate`, gates, and fresh intervals. Adopting fixture no longer says production stays legacy. Failed-gate fixture still does. Walkthrough now: "The development FNR increase is confirmed on fresh." 86.7% is annotator unanimity, not an MCC ceiling. Tests: existing adopting report test plus `test_report_prose_follows_a_failed_gate`.
- Blocker for landing: none from the write-back list. Still do not land as one giant commit (plan Phase 6.1).
- Do not do next: do not adopt `SafetyGrader`; do not mix MCC 0.20/0.38/0.77 with this contract; do not drop the FNR gate; do not wipe the checkpoints; do not start `quality_eval.py`; do not retune `HARM_PROMPT` / `BENIGN_PROMPT`.

## Layout (Grok, 2026-09-11)

PR #7 is on `main`. This branch was replayed onto that merge (`git rebase --onto origin/main 12883f4`). The snapshot commit was dropped because main already had those files. Zero conflicts. Merge-base is now `410cc04`.

- Worktree: `C:\Users\isdis\git\aegis-eval-organize`
- Branch: `codex/organize-project-layout` (no upstream, do not force-push)
- Tests: `C:\Users\isdis\git\aegis-eval\venv\Scripts\python.exe -m pytest -q` → 397 passed
- CLI: `run.py`, `calibrate.py`, `compare_graders.py`, `quality_eval.py`, `quality_report.py` `--help` all parse and do not write
- Identity: `PROVENANCE_VERSION=aegis_eval.1`. Hashes via `find_spec`. Frozen `output/*.py` still load through `sys.modules` aliases
- New eval output would go to `output/grader-quality/aegis_eval.1/` and `output/hosted-comparison/aegis_eval.1/`. Sealed unversioned checkpoints were not rewritten
- Brief: `docs/superpowers/plans/GPT-REVIEW-layout.md`

GPT: review the layout only. Do not re-open the MCC/adopt_candidate review above.

## Layout write-back (GPT fills this)

- Date:
- What you changed (files):
- Tests run:
- Findings:
- Blocker for landing:
- Do not do next:
