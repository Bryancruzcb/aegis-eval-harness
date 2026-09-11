# Grader Contract and Validation Implementation Plan

> For agentic workers: use superpowers:executing-plans task by task. Preserve existing uncommitted work.

**Goal:** Correct the refusal-versus-harmfulness confusion, measure one candidate honestly, and evaluate unnecessary-refusal detection on independent human labels.
**Architecture:** Separate safety judgment fields from verdict mapping. A resumable local-only harness freezes datasets, model/source identities and prediction records, and writes aggregate Markdown/JSON without raw dataset publication.
**Tech Stack:** Python 3.11, Pydantic, pytest, local Ollama Qwen2.5.
**Spec:** docs/superpowers/specs/2026-09-08-grader-contract-design.md

## Global constraints
- Original $5 cap persists; this phase makes zero paid calls.
- JAILJUDGE and XSTest raw content stays local and ignored.
- One frozen candidate; no adjustment based on fresh validation labels.
- Preserve all earlier checkpoints. Do not claim population accuracy or fresh public-data contamination guarantees.

## Task 1: Audit the error contract
- [x] Verify JBB official provenance and exact XSTest prompt overlap.
- [x] Save row-index taxonomy and aggregate error counts; labels remain original human labels. Analyst tags are not new ground truth.
- [x] Correct documentation that calls this a harmful-only dataset or claims newer judges necessarily improve accuracy.

## Task 2: Explicit safety candidate
Files: safety_grader.py, tests/test_safety_grader.py.
- [x] Write failing tests for harmless compliance passing harmfulness evaluation, refusal-plus-harm failing, partial harmless refusal failing, and compliance openers escalating.
- [x] Implement SafetyJudgment(reasoning, harmful_assistance, refused, partial_refusal) and SafetyGrader screen/parse/verdict interfaces.
- [x] Run focused tests, review, then freeze source before evaluation.

## Task 3: Reproducible local evaluation
Files: quality_eval.py, tests/test_quality_eval.py.
- [x] Test strict labels, pinned file hashes, no overlap with prior cohorts, one harmless response per prompt, local-only enforcement, resume identity and error retry behavior.
- [x] Build JBB dev, fresh JAILJUDGE 200, and XSTest 150 cohorts. Store only hashes/metadata in public reports.
- [ ] Run `venv\Scripts\python.exe quality_eval.py` to evaluate all three sequentially, with baseline and candidate, checkpoint after every record and resume automatically.
- [ ] Report paired metrics, label-specific harmless errors, costs/latency boundaries, and paired bootstrap uncertainty on fresh validation.

## Task 4: Decide and document
Files: docs/grader-quality-results.json, docs/grader-quality-walkthrough.md, README.md.
- [ ] Apply predeclared gate; retain unsuccessful candidate as experimental, never claim improvement from changed cohorts.
- [ ] Update README with current contract, measured results, reproducible command, and project explanation.
- [ ] Run offline suite and independent review. Verify no raw dataset included and no paid calls.
- [ ] Save completion status and next steps in this plan and walkthrough; no user coding required.

Current status: local evaluation in progress. Preflight independent review complete, 359 offline tests passed. Source hashes frozen at first run. No paid calls.
