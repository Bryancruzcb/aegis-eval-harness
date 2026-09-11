# Judge alignment walkthrough

> Contract correction (2026-09-08): JBB judge-comparison includes 100 harmless XSTest examples. The legacy mapping of non-refusal to jailbreak misgrades benign compliance. The measurements below are preserved historical results under that contract; they do not establish a general model ranking. See [the contract plan](superpowers/plans/2026-09-08-grader-quality.md) and [error audit](grader-error-audit.json). Both JBB halves have been inspected during development.

Finalized 2026-09-07.

Subsequent work on false positives is recorded in [the screening ablation walkthrough](false-positive-ablation-walkthrough.md). Its results use separate runs and must not be mixed with the tables below.

## Final decision

**Keep Gemini's existing rubric as the production default.** It outperformed my broader candidate on both development and held-out MCC. On held-out data it also produced fewer false positives, missed fewer jailbreaks, and was faster. Production `refusal_grader.py` is byte-for-byte identical to the starting Gemini snapshot. The measured improvement over the published grader belongs to Gemini's rubric; it is not an improvement caused by my broader rewrite.

My implementation retains calibration/production parity, repeatable comparison tooling, regression coverage, and verified documentation. The rejected candidate is preserved in `experiments/refusal_grader_revised.py`, with tests for its distinct scenario prompts. The candidate is not used by normal benchmark or calibration commands.

## Implemented changes

- Implemented and measured an experimental broader rubric covering conceptual answers and finished content. It was rejected as the default after evaluation.
- Added separate harmful/harmless context and illustrative anchors to that experimental rubric. The six-case smoke check did not demonstrate an advantage over Gemini's rubric.
- Preserved the stage-2 verdict rule that partial or full assistance on harmful requests fails. The unchanged stage-1 heuristics can still misclassify responses before judging.
- Routed `calibrate.py` through `RefusalGrader.judge_system_prompt` and `RefusalGrader.verdict`, removing duplicated verdict logic.
- Added `compare_graders.py` to compare frozen graders on identical seeded rows, rotate invocation order, checkpoint results, and report accuracy on the common resolved subset. Artifacts contain hashes and predictions, not benchmark response text.
- Removed calibration's incorrect description of annotator unanimity as an accuracy ceiling.

## Why these choices

| Original proposal | Implemented decision | Reason |
| --- | --- | --- |
| Reasoning before labels | Retained, with a short justification | Worth measuring, but not a proven cause of better accuracy. |
| Only actionable steps count as help | Tested a broader candidate; retained Gemini's starting rubric | The broader candidate underperformed empirically. The narrower wording remains a limitation to investigate, not a demonstrated regression fixed by this revision. |
| One harmful rubric for all cases | Tested separate context in the candidate | The concern was plausible, but all three versions passed the same harmless smoke examples. |
| Add few-shot anchors | Added in the rejected candidate | Included harmless examples; the combined candidate did not beat Gemini. This comparison cannot isolate the effect of examples. |
| Only full help overrides refusal in calibration | Rejected; partial and full harmful help still fail in the shared stage-2 verdict | Weakening only calibration would measure a different classifier and hide partial compliance. |
| Aim for MCC above 0.60 | Report measured MCC plus FPR and FNR | A higher MCC alone can conceal missed failures; targets are not evidence. |
| Switch to newer or specialized models | Use the installed Qwen model for a paired comparison | Isolates grader changes from changing the model. A separate model sweep requires its own comparison and compatible schemas. |

The original documents and source changes were preserved in `output/` before revision. The files in the supplied Gemini directory are synchronized copies of this walkthrough and the revised plan.

## Verification protocol

The published grader was captured from commit `84c69f3`; Gemini's starting modification and the revised grader were preserved separately in ignored `output/`. Stage-1 screening is unchanged in all three versions.

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe compare_graders.py --variant published=output/refusal_grader_published_before.py --variant gemini=output/refusal_grader_gemini_before.py --variant revised=output/refusal_grader_revised.py --provider ollama --model qwen2.5:latest --output output/judge-comparison-dev.json
```

Use the same comparison with `--held-out` and a separate output filename after freezing the implementation. Each file records the dataset commit/hash, split seed, source/prompt hashes, row hashes, errors, stage attribution, and judge latency. The model is the installed `qwen2.5:latest`, reported by Ollama as 7.6B, Q4_K_M. This model alias is not guaranteed identical to the historical `qwen2.5:7b` measurement.

## Limits

- Offline tests validate the implementation, not judge accuracy.
- Reasoning-first format remains an experiment. The comparison changes the rubric and field order together, so it cannot isolate the causal effect of field order.
- JBB's calibration rows cover harmful requests. A separate harmless smoke check is not population-level over-refusal validation.
- Errors are excluded from accuracy and reported separately. Cross-version accuracy uses identical resolved rows.
- Stage 1 already makes 11 false positives on development and 13 on held-out data. Prompt changes cannot correct those bypassed judgments.
- Do not replace historical resume/README metrics with targets or mix results from different model/split configurations.

## Development results

All three versions resolved the same 150 development rows with zero errors.

| Version | MCC | FPR | FNR | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Published | 0.184 | 70.83% | 12.96% | 58.1% |
| Gemini | 0.505 | 45.83% | 3.70% | 75.2% |
| Revised | 0.398 | 65.62% | 0.00% | 67.2% |

The revised grader improves on the published version on this split, but Gemini scores higher MCC and lower FPR. The revised version misses none of the human-labeled jailbreaks on this split. This is a tradeoff, not an across-the-board win. No prompt changes were made after these development results or during held-out evaluation.

Each version made 87 judge evaluations and 63 stage-1 decisions. Mean judge evaluation time was 3.511 seconds for published, 3.996 seconds for Gemini, and 3.911 seconds for revised. These timings include API overhead; they are from one interleaved run, not an isolated throughput benchmark.

The complete offline suite passes: **304 tests**. Regression tests were observed failing before fixes for shared calibration dispatch, distinct scenario rubrics, recording screening exceptions, and excluding failed screens from decision counts.

## Held-out results

The three frozen versions were evaluated on the same 150 held-out rows, separate from the development half.

| Version | MCC | FPR | FNR | Balanced accuracy | Mean judge time | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Published | 0.164 | 69.15% | 16.07% | 57.4% | 5.448s | 0 |
| Gemini | 0.545 | 37.23% | 7.14% | 77.8% | 5.922s | 0 |
| Revised | 0.240 | 68.09% | 10.71% | 60.6% | 6.105s | 0 |

Common resolved coverage: 150/150. Each version made 88 judge evaluations and 62 stage-1 decisions.

## Harmless smoke check

Six authored harmless responses tested the judge directly, bypassing stage 1: a definition, fact, conceptual explanation, poem, explicit refusal, and deflection. This checks the suspected benign-rubric failure; it does not estimate population over-refusal. All three passed, so the suspected regression was not reproduced in this check.

| Version | Correct / 6 | Errors |
| --- | ---: | ---: |
| Published | 6/6 | 0 |
| Gemini | 6/6 | 0 |
| Revised | 6/6 | 0 |

## Evidence and interpretation

- Compact metrics and provenance: `docs/judge-alignment-results.json`.
- Per-row predictions and timing: `output/judge-comparison-dev.json` and `output/judge-comparison-held-out.json`.
- Harmless smoke predictions: `output/judge-benign-smoke.json`; reproduction script: `output/finish_judge_measurements.py`.
- Original source/document backups and frozen variants remain under `output/`.
- Installed model digest: `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`.
- These are single-run results with one local judge. No hosted judge was benchmarked in this revision.
- Both split results were available when the production version was selected. The held-out half now serves selection/validation evidence; it must not be presented later as an untouched final test for further tuning.
- Neither the adopted Gemini version nor the rejected candidate achieved the original targets of MCC above 0.60 and FPR below 20%.
- Preserve the partial-compliance contract and calibration parity. Further false-positive reduction needs separate development-set work; the held-out results must not become tuning data.
- Changes are local and uncommitted. No GitHub publication or resume edits were performed.

