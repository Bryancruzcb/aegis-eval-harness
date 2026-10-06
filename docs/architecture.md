# AegisEval architecture

Implementation lives under `aegis_eval/`. Root `run.py`, `calibrate.py`, `compare_graders.py`, `quality_eval.py`, and `quality_report.py` are command wrappers.

```text
cli                         --> core, harness, reporter, benchmarks, workflows
workflows.hosted            --> core, harness, benchmarks, workflows.grader_quality
workflows.grader_quality    --> core, harness, benchmarks
reporter                    --> core
harness                     --> core, benchmarks
benchmarks                  --> core
core                        --> stdlib and third-party only
```

`tools/` and `experiments/` are never imported by `aegis_eval`.

Two evaluation problems share one CLI and must stay separate. Target grading asks whether the assistant leaks a secret, complies with a harmful request, or refuses a harmless one. Grader grading asks whether the judge agrees with human labels under a named contract.

Target grading lives in `aegis_eval/harness/graders.py` (`SecretGuardianGrader`) and `aegis_eval/harness/refusal_grader.py` (`RefusalGrader`). Grader grading lives in `aegis_eval/workflows/grader_quality/`. An error, including a rate limit or an outage, is not a safety failure. `build_summary` leaves those cases out of the pass rate. When nothing was evaluated, `pass_rate` is null, and the report shows a dash.

Layout, wrappers, identity, and migration order: `docs/superpowers/specs/2026-09-11-project-layout-design.md`.
