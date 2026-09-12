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

Layout, wrappers, identity, and migration order: `docs/superpowers/specs/2026-09-11-project-layout-design.md`.
