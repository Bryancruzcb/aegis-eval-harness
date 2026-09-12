# Scope of the local contract experiment

This file marks what is in the SafetyGrader contract experiment and what is
historical work that happens to sit in the same working tree.

## In slice

- `aegis_eval/workflows/grader_quality/{quality_eval,quality_data,quality_report,safety_grader}.py`
  (root `quality_eval.py` and `quality_report.py` are wrappers)
- Frozen cohorts in `output/grader-quality/{dev,fresh,harmless}.json`
- Reports in `docs/grader-quality-results.json` and
  `docs/grader-quality-walkthrough.md`
- Decode-only harness changes: parse retry, fence strip, Ollama `json_schema`,
  sealed resume identity, `failed_records` schema 2

`adopt_candidate` is false. Do not treat any other file in this tree as a
reason to swap production `RefusalGrader`.

## Historical, other contracts

These are not the frozen SafetyGrader experiment. Do not mix their MCC
figures with the table in README.

- `experiments/refusal_grader_revised.py` — rejected judge-alignment candidate
- `aegis_eval/workflows/hosted/hosted_comparison.py` and
  `experiments/hosted_comparison_37.py` — paid Gemini sweeps on JBB-dev under
  the old refusal contract, $5 cap, already spent
- `docs/calibration-history.md`, `docs/judge-alignment-*.md`,
  `docs/hosted-judge-*.md`, `docs/gemini-37-*.md`,
  `docs/false-positive-*.md`
- The `refusal_grader.py` prompt/schema field-order edit is leftover from
  that alignment pass. It is not an adoption of `SafetyGrader` and it is
  not part of the local Qwen contract numbers.

## Do not

- Adopt `SafetyGrader`
- Drop the FNR clause of `passes_gate`
- Quote 0.20 / 0.38 / 0.77 next to 0.510 / 0.777 as one progression
- Wipe `output/grader-quality/{dev,fresh,harmless}.json`
- Start a second `quality_eval.py` while `run.lock` is held
