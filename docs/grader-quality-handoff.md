# Grader quality experiment: resume state

The plan is `docs/superpowers/plans/2026-09-08-grader-quality.md`. The local
run is complete. `adopt_candidate: false`. Do not claim production quality
from this file.

Checkpoints: `output/grader-quality/{dev,fresh,harmless}.json` (150/200/150).
Do not delete them. Do not start a second `quality_eval.py` while `run.lock`
is held. Complete checkpoints are sealed: decode-only identity amendment
applies only to incomplete files. Reporting is
`venv\Scripts\python.exe quality_report.py`.

Failed records on `fresh` are diagnostic. Schema 1 rows have no top-level
`error_class`; read them through `quality_eval.failed_records` rather than
rewriting the checkpoint.

Completed: original error audit; candidate schema/contract; pinned
fresh/harmless selection; hash/model/cross-cohort resume validation; local
comparisons; aggregate report; adoption gate (failed honestly). Scope of
what is in this experiment vs historical files:
`docs/grader-quality-scope.md`.

Important finding: old JBB-300 includes 100 harmless XSTest examples; the old
all-expect-refuse contract misclassifies benign compliance. On dev 150, 27 of
41 FP are on harmless prompts. Earlier model rankings are historical results
under that flawed contract. The candidate uses separate
harmful_assistance/refused/partial_refusal fields. Do not tune its prompt
after seeing fresh labels. No new paid calls.

Do not transmit JAILJUDGE examples to another model/agent for review. Review
code, aggregate results, and opaque row hashes only.

Development 150/150, no errors: legacy TP51 FP41 FN3 TN55 MCC 0.510;
candidate TP49 FP11 FN5 TN85 MCC 0.777. Candidate fails no-higher-FNR gate
(5 missed vs 3). Fresh 200/200: MCC 0.691 vs 0.694 (difference includes 0),
FNR 13% → 26% (difference excludes 0). Harmless 150/150 is a separate
over-refusal task. Production stays `RefusalGrader`.
