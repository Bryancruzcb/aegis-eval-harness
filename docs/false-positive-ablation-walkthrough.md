# Screening ablation and fresh validation

> Contract correction (2026-09-08): JBB judge-comparison includes 100 harmless XSTest examples. The legacy mapping of non-refusal to jailbreak misgrades benign compliance. The measurements below are preserved historical results under that contract; they do not establish a general model ranking. See [the contract plan](superpowers/plans/2026-09-08-grader-quality.md) and [error audit](grader-error-audit.json). Both JBB halves have been inspected during development.

Subsequent update: the user authorized a $5 hosted comparison and XSTest human-labeled harmless responses were found. See [hosted judge comparison](hosted-judge-comparison.md) for current status. The remaining-work section below records the earlier experiment's state.

## Decision

Keep the existing production rubric and screening. On the 150 JBB development responses, bypassing screening reduced false positives but increased missed jailbreaks. That fails the predeclared no-higher-FNR rule. No new production accuracy improvement is claimed from this experiment.

## What was implemented

- `compare_graders.py --screening-ablation` compares the original two-stage route with judging every response using the same frozen rubric and model. Shared escalation cases reuse a single judge output so model variation cannot create a routing difference.
- Checkpoints save predictions, hashes, errors and timings without response text. `--resume` verifies the configuration and completed row order before continuing. Independent review identified a missing dependency check; new runs also freeze evaluator/provider/harness hashes, package versions and retry settings.
- `analyze_screening.py` reports which screening mistakes were corrected and which correct decisions were lost, alongside confusion counts and coverage.
- `jailjudge_validation.py` selects a pinned local-only public test subset. It rejects malformed labels and unexpected file contents, removes empty records, excludes normalized exact JBB request/response overlap and keeps one row per normalized request.
- Fresh validation includes paired, class-stratified bootstrap uncertainty. Exact request deduplication does not establish independence of semantic request families.

## Controlled development result

Frozen source: `output/false-positive-ablation/refusal_grader_baseline.py`. Judge: local `qwen2.5:latest`, installed 7.6B Q4_K_M model. Source and model digests are in `false-positive-ablation-results.json`.

| Route | TP | FP | FN | TN | MCC | FPR | FNR | Resolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing two-stage | 51 | 41 | 3 | 55 | 0.510 | 42.71% | 5.56% | 150/150 |
| Judge every response | 50 | 37 | 4 | 59 | 0.526 | 38.54% | 7.41% | 150/150 |

There were zero evaluation errors. Bypassing screening corrected four false positives but lost one correctly detected jailbreak. Thirty false positives and three false negatives came from shared judge decisions, so removing screening cannot fix those errors. Seven of the screen's eleven false positives persisted when judged.

The experiment made 150 unique logical judge evaluations. The two-stage route would need 87 versus 150 for judge-only, an increase of 72.4%. Estimated summed route time increased from 390.673 to 684.762 seconds, about 75.3%. These are counterfactual routing costs from shared outputs, not independently timed runs. Internal provider retries are not separately counted.

The 0.60 MCC / below-20% FPR target was not reached on this development cohort. The baseline differs slightly from its previous run despite temperature zero; use the paired result above for this decision rather than mixing previous-run metrics into the comparison.

## Why no further rubric rewrite was adopted

This ablation isolates screening. It does not supply evidence that a specific additional prompt rewrite will resolve the thirty shared judge false positives without harming recall. The broader rubric from the earlier experiment already underperformed. No additional candidate was fitted to individual benchmark answers. Partial substantive harmful assistance still maps to failure in production and calibration.

## Fresh validation

The separately downloaded JAILJUDGE ID test file is pinned to commit `12c5739bf53fbe01709fcd875b949314f391c0a0`, SHA-256 `9d19d2db791b7f3a19993365421c9cdc40b1f241ae0e6dcacbddbcf86aa654da`.

Of 4,500 rows, selection excluded 16 empty-text rows, one exact JBB overlap and 677 duplicate requests. The remaining pools contained 2,409 negative and 1,397 positive labels. Seed 20260907 selected 100 from each class before predictions were examined. Both configurations were frozen before this validation; the development rejection above is not retuned based on its results.

The [dataset card](https://huggingface.co/datasets/zdxnlp/JailJudge) describes human annotation with GPT-4-assisted secondary review and adjudication. These are published labels with their own contract, not newly commissioned labels under our exact project rubric. Target-response model identities are absent from this file. Label semantics, possible semantic overlap, public benchmark contamination and the balanced sampling design limit interpretation. Report this cohort separately from JBB.

The custom license restricts transfers. Raw data remains in ignored `.cache/jailjudge/`; only local Ollama is allowed by the validation CLI. Per-row output contains no prompt/response text. This dataset does not satisfy the separate harmless-response validation requirement.

Validation completed with 200/200 resolved rows and zero errors. Aggregates and provenance are in `false-positive-validation-results.json`.

| Route | TP | FP | FN | TN | MCC | FPR | FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing two-stage | 80 | 16 | 20 | 84 | 0.641 | 16% | 20% |
| Judge every response | 80 | 15 | 20 | 85 | 0.651 | 15% | 20% |

Both point estimates meet the MCC/FPR targets on this cohort. They do not show improvement over JBB, since the responses and labeling contract differ. The existing production grader achieved its result without another rubric or screening change. Missing 20 of 100 labeled jailbreaks remains a material weakness.

The 95% paired class-stratified bootstrap intervals for the existing route are MCC 0.540–0.741, FPR 9–23%, and FNR 12–28%. Thus this sample does not establish that the population meets the MCC/FPR targets. Judge-only corrected one additional false positive. Its paired MCC difference interval was 0.000–0.032 and FPR difference interval was -3–0 percentage points. The observed FNR difference was zero in every resample because the configurations missed the same positive rows; this does not guarantee equal recall on future data.

The run used 200 unique logical judge evaluations. Estimated route counts were 186 versus 200, and summed route times were 1,366.049 versus 1,491.840 seconds. Total run wall time was about 25 minutes. These timings belong to this run and machine state, not the earlier JBB experiment.

The fresh result does not reverse the failed development adoption gate. It adds evidence that removing all screening has a small, dataset-dependent benefit and does not solve the judge's remaining mistakes.

## Verification

The complete suite passed **327 tests**. Regression coverage includes forced judging, shared decisions, partial harmful assistance, interrupted and completed resumes, dependency changes, strict dataset labels, overlap/deduplication, local-only inference and paired uncertainty. `git diff --check` passed. Independent review found the resume dependency gap; the fix and final evidence received a follow-up review with no remaining actionable findings.

Production `refusal_grader.py` still matches the experiment's baseline snapshot byte-for-byte. The active run's evaluator/provider/calibration/config/grader/selection dependency hashes were checked after completion and matched their captured values. The comparison harness itself received the documented resume-only fix after the active process had loaded its original code.

## Reproduce

From the repository using its virtual environment:

```powershell
.\venv\Scripts\python.exe compare_graders.py --variant baseline=output/false-positive-ablation/refusal_grader_baseline.py --screening-ablation --provider ollama --model qwen2.5:latest --output output/false-positive-ablation/qwen-dev-new.json
.\venv\Scripts\python.exe analyze_screening.py output/false-positive-ablation/qwen-dev.json --output docs/false-positive-ablation-results.json
.\venv\Scripts\python.exe compare_graders.py --variant baseline=output/false-positive-ablation/refusal_grader_baseline.py --screening-ablation --provider ollama --model qwen2.5:latest --jailjudge-id .cache/jailjudge/JAILJUDGE_ID.json --output output/false-positive-ablation/qwen-jailjudge-new.json
.\venv\Scripts\python.exe analyze_screening.py output/false-positive-ablation/qwen-jailjudge.json --output docs/false-positive-validation-results.json
.\venv\Scripts\python.exe -m pytest -q
git diff --check
```

The comparison commands create new runs; the analysis commands read the original measured artifacts. Add `--resume` only to continue an output created by the same frozen harness. The original runs predate dependency fingerprint enforcement and are retained as historical evidence, not silently upgraded checkpoints. The active validation harness was preserved as `output/false-positive-ablation/compare_graders_validation_run.py`; dependency hashes were captured during that run in `validation-dependencies.json`, not at launch. No evaluation dependency or grader changed during its inference. Download the pinned ID file and its license from the dataset repository separately; the CLI validates the exact file hash.

## Remaining work and ownership

- A stronger hosted judge comparison has not run. The API-budget question remains unanswered, so no new paid calls were made. The JAILJUDGE dataset is restricted to local inference even if a hosted budget is later authorized.
- Independently labeled harmless responses remain needed for population-level over-refusal measurement. The previous six authored smoke cases are only regression evidence.
- No manual coding is required from the user for these implemented tools. Hosted spending authorization and truly independent human annotation cannot be replaced by code or self-generated AI labels.
- No GitHub publication or resume metrics were changed for this experiment.
