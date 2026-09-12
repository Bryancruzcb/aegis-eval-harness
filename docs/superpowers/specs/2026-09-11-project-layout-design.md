# AegisEval project layout

- **Date:** 2026-09-11
- **Status:** Implemented. Rebased onto `origin/main` `410cc04` (PR #7). `pytest -q` → 397 passed. PR: https://github.com/Bryancruzcb/aegis-eval-harness/pull/8
- **Branch:** `codex/organize-project-layout` at `C:\Users\isdis\git\aegis-eval-organize`
- **Shared checkout:** `C:\Users\isdis\git\aegis-eval` is `main` at `410cc04`. Do not edit this layout from that folder. Ignore leftover worktrees `aegis-eval-4.1-nonce` and `aegis-eval-phase22` (both parked at `84c69f3`).

## In plain terms

The harness works. Finding a file does not.

Almost every Python module sits at the repo root. CLI entry points, provider clients, graders, the HTML reporter, and the JBB calibration script share one folder. README already has a layout table. The table is a list of 13 root files. That is the problem it is describing.

This phase puts implementation behind package interfaces and leaves the commands people type alone. No scores, prompts, or datasets change.

Do not move hashed files, and do not implement slices, until the shared checkout commits its experiment on the old paths and this branch rebases onto that commit. A layout move of `evaluators.py` while the other worktree still edits that file is a rename-plus-edit fight. Wait.

## Decisions

These replace earlier hedges in this spec.

1. **New evaluation output.** Sealed files stay at `output/grader-quality/{dev,fresh,harmless}.json` and `output/hosted-comparison/hosted-judge-dev.json`. Do not amend them. Do not wipe them. A later run writes to `output/grader-quality/<PROVENANCE_VERSION>/` and `output/hosted-comparison/<PROVENANCE_VERSION>/`. First new value is `aegis_eval.1`.
2. **Importable experiment code stays in the package.** `SafetyGrader`, JAILJUDGE load, and `paired_intervals` live under `aegis_eval/workflows/grader_quality/`. `experiments/` and `tools/` are never imported by `aegis_eval`.
3. **Sequence.** Commit the grader-quality snapshot on the shared checkout. Rebase this branch onto that snapshot. Then move files. Do not reverse that order.
4. **Secret Guardian policy.** Leak detection, profanity checks, and the passcode judge rubric sit next to `SecretGuardianGrader` in `harness`. `core.evaluators` keeps judge I/O only: parse, retry, transcript rendering, `run_llm_judge_eval_conversation`.
5. **Thin packages.** No `cli/calibrate.py`. No `reporting/` package. Root `calibrate.py` imports workflow `main()` directly. Keep `cli/run.py` because it owns gates, redaction policy, and positive control. Keep `workflows/grader_quality/` so calibrate does not move twice when quality files land.
6. **Public root wrappers after rebase.** `run.py`, `calibrate.py`, `compare_graders.py`, `quality_eval.py`, `quality_report.py`. Hosted trial scripts are not public commands. They stay workflow modules without root wrappers.
7. **Identity.** Hash implementation modules through `importlib.util.find_spec(...).origin`, never wrapper bytes, never `Path(calibrate.__file__).parent / "evaluators.py"`. Stable keys keep today's basenames (`evaluators.py`, not a package path). Put `provenance_version` on the identity dict. Rewrite identity in the same commit that first moves a hashed file.
8. **Frozen snapshots.** `compare_graders.load_variant` pre-registers `sys.modules["evaluators"]` and `sys.modules["graders"]` to the package modules before it execs a file under `output/`. Do not edit `output/`.

## What this branch actually contains

Historical note from before the moves. After rebase onto PR #7, `pytest -q` reports **397 passed**. Root Python is the five wrappers. Implementation is under `aegis_eval/`. Experiment modules that this section listed as missing now live in `aegis_eval/workflows/`.

Root Python that exists here today:

```text
attackers.py  calibrate.py  config.py  evaluators.py  graders.py
providers.py  refusal_grader.py  reporter.py  run.py  runner.py
scenarios.py  target.py
benchmarks/{__init__,fetch,jbb}.py
test_cases.json
tests/test_*.py          # 17 files
```

These files do **not** exist on this branch. They live as uncommitted work on the shared checkout. Do not add them here. Do not invent stubs. After rebase they are ordinary files in the tree and the file map below applies.

```text
analyze_screening.py  compare_graders.py  hosted_comparison.py
hosted_comparison_37.py  jailjudge_validation.py  quality_data.py
quality_eval.py  quality_report.py  report_hosted_comparison.py
safety_grader.py  experiments/
```

## Goals and non-goals

### Goals

1. Implementation lives under `aegis_eval/`, grouped by who owns the behavior.
2. Root `run.py` and `calibrate.py` become thin wrappers around `main()`. After rebase, so do `compare_graders.py`, `quality_eval.py`, and `quality_report.py`.
3. `test_cases.json` moves to `data/test_cases.json`. `config.BASE_DIR` stays the repo root, not the package directory.
4. Every assertion that existed at rebase still exists after each slice. Import paths may change.
5. `tests/test_import_layers.py` fails a layer-DAG violation, including relative imports.
6. Existing CLI flags, exit codes 0/1/2/3, JSON keys, and HTML report cards stay the same.

### Non-goals

- Scoring contracts, prompts, datasets, adoption of `SafetyGrader` as the production grader.
- Rerunning local or hosted evaluations.
- Moving, deleting, or regenerating `output/`, `.cache/`, `venv/`, checkpoints, or locks.
- Bulk-moving historical docs during the code migration.
- Installing the package with pip. `pytest.ini` `pythonpath = .` is enough.
- Consolidating `hosted_comparison.py` and `hosted_comparison_37.py`.

## Alternatives considered

**Docs only.** Add a map and editor exclusions. Cheap. The root folder stays a junk drawer.

**Move everything in the shared checkout.** Entangles this with uncommitted grader-quality files and checkpoint hashes.

**Staged package on this branch, rebase later.** Rejected. This branch would move `evaluators.py`, `refusal_grader.py`, `run.py`, `reporter.py`, and `calibrate.py` while the shared worktree still edits those files. Rebase then becomes a rename-plus-edit conflict. Identity still hashes those files by root basename, so the later series would have to reopen core.

**Snapshot, rebase, then staged package.** Selected. One dependency slice per commit on the combined tree. Wrappers stay. Identity rewrite lands in the first commit that moves a hashed file.

## Dependency direction

Arrows mean "may import". Cycles are bugs.

```text
cli                         --> core, harness, reporter, benchmarks, workflows
workflows.hosted            --> core, harness, benchmarks, workflows.grader_quality
workflows.grader_quality    --> core, harness, benchmarks
reporter                    --> core          (payloads in, HTML/text out; no runner)
harness                     --> core, benchmarks
benchmarks                  --> core
core                        --> (stdlib, third-party, nothing else in this repo)
tools                       --> anything; nothing in aegis_eval may import tools
experiments                 --> nothing in aegis_eval may import experiments
```

`workflows.grader_quality` must not import `workflows.hosted`. That is the cut that `exclusive_run` moving into `core.lock` buys.

`core` does not know what a Scenario is. `harness` owns `Grader`, `Scenario`, case execution, and summary dicts. `reporter` renders an already-built payload. `calibrate.py` is a workflow because it scores the production refusal grader against JBB labels. It may import `harness.refusal_grader`. It may not import `cli`.

The first write-up put grader contracts in `core.evaluation`. That is wrong for this codebase. Phase 2a put `Grader` next to `Scenario`. Keep it there.

`core.evaluators` does not own Secret Guardian leak or profanity checks. Those key off `config.SYSTEM_SECRET` and the passcode rubric. They belong next to `SecretGuardianGrader`. Judge parse, retry, transcript rendering, and `run_llm_judge_eval_conversation` stay in `core.evaluators`.

## Target tree

Keep current filenames. Do not rename `evaluators.py` to `evaluation.py`. Do not rename `quality_eval.py` to `eval.py`. The extra name churn buys nothing and breaks identity keys.

```text
aegis_eval/
  __init__.py                 # empty
  reporter.py
  core/
    __init__.py
    config.py
    lock.py                   # exclusive_run, extracted from hosted_comparison
    providers.py
    target.py
    evaluators.py             # judge I/O only
  harness/
    __init__.py
    cases.py                  # load_test_cases, extracted from runner
    runner.py
    scenarios.py
    graders.py                # Grader, Screen, Verdict, SecretGuardianGrader,
                              # leak/profanity/passcode rubric
    refusal_grader.py
    attackers.py
  benchmarks/
    __init__.py
    fetch.py
    jbb.py
  workflows/
    __init__.py
    grader_quality/
      __init__.py
      calibrate.py
      compare_graders.py      # after rebase
      quality_eval.py         # after rebase, keep this filename
      quality_report.py
      quality_data.py
      safety_grader.py
      jailjudge_validation.py
      analyze_screening.py    # includes paired_intervals
    hosted/                   # after rebase
      __init__.py
      hosted_comparison.py
      report_hosted_comparison.py
  cli/
    __init__.py
    run.py                    # argparse, gates, positive control, redaction policy
data/
  test_cases.json
run.py                        # wrapper
calibrate.py                  # wrapper -> workflows.grader_quality.calibrate.main
compare_graders.py            # wrapper after rebase
quality_eval.py               # wrapper after rebase
quality_report.py             # wrapper after rebase
experiments/<study>/          # never imported: refusal_grader_revised.py,
                              # hosted_comparison_37.py
tools/                        # never imported by aegis_eval
tests/
  test_import_layers.py       # new
  core/
  harness/
  benchmarks/
  workflows/
  cli/
  test_reporting.py           # reporter plus CLI redaction; package imports
docs/
  architecture.md             # short map, added with the skeleton
  superpowers/specs/2026-09-11-project-layout-design.md
```

Root after the skeleton still has `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `env.example`, `LICENSE`, `README.md`, `.github/`, `docs/`, `output/` (ignored), `.cache/` (ignored).

Drop the undocumented `python target.py` `__main__` smoke. Do not add a wrapper for it.

## File map

Harness files that exist on this branch today:

| Today | Tomorrow | Notes |
|---|---|---|
| `config.py` | `aegis_eval/core/config.py` | `BASE_DIR` = repo root. See below. |
| `providers.py` | `aegis_eval/core/providers.py` | |
| `target.py` | `aegis_eval/core/target.py` | Drop `__main__`. |
| `evaluators.py` | `aegis_eval/core/evaluators.py` | `JudgeParseError`, parse, retry, transcript, LLM-judge call. Not leak/profanity. |
| `runner.py` | `aegis_eval/harness/runner.py` | `run_single_test`, `run_suite`, `build_summary`, `aggregate_repeats` |
| `runner.load_test_cases` | `aegis_eval/harness/cases.py` | Breaks the `scenarios` ↔ `runner` cycle |
| `scenarios.py` | `aegis_eval/harness/scenarios.py` | |
| `graders.py` | `aegis_eval/harness/graders.py` | `Grader`, `Screen`, `Verdict`, `SecretGuardianGrader`, leak/profanity/rubric |
| `refusal_grader.py` | `aegis_eval/harness/refusal_grader.py` | |
| `attackers.py` | `aegis_eval/harness/attackers.py` | |
| `reporter.py` | `aegis_eval/reporter.py` | |
| `benchmarks/` | `aegis_eval/benchmarks/` | Move the existing package |
| `calibrate.py` (body) | `aegis_eval/workflows/grader_quality/calibrate.py` | Root wrapper imports this `main()` |
| `run.py` (body) | `aegis_eval/cli/run.py` | `parse_args`, `main`, `run_positive_control`, exit helpers |
| `test_cases.json` | `data/test_cases.json` | |
| `tests/test_config.py` | `tests/core/test_config.py` | |
| `tests/test_providers.py` | `tests/core/test_providers.py` | |
| `tests/test_target.py` | `tests/core/test_target.py` | |
| `tests/test_evaluators.py` | split: judge I/O stays `tests/core/test_evaluators.py`; leak/profanity tests move with `SecretGuardianGrader` |
| `tests/test_runner.py` | `tests/harness/test_runner.py` | |
| `tests/test_dataset.py` | `tests/harness/test_dataset.py` | |
| `tests/test_scenarios.py` | `tests/harness/test_scenarios.py` | |
| `tests/test_scenario_parity.py` | `tests/harness/test_scenario_parity.py` | |
| `tests/test_graders.py` | `tests/harness/test_graders.py` | |
| `tests/test_refusal_grader.py` | `tests/harness/test_refusal_grader.py` | |
| `tests/test_attackers.py` | `tests/harness/test_attackers.py` | |
| `tests/test_reporting.py` | `tests/test_reporting.py` | Imports `aegis_eval.reporter` and `aegis_eval.cli.run` for redaction |
| `tests/test_fetch.py` | `tests/benchmarks/test_fetch.py` | |
| `tests/test_jbb.py` | `tests/benchmarks/test_jbb.py` | |
| `tests/test_calibrate.py` | `tests/workflows/test_calibrate.py` | |
| `tests/test_cli.py` | `tests/cli/test_cli.py` | |
| `tests/test_positive_control.py` | `tests/cli/test_positive_control.py` | Patch `aegis_eval.cli.run`, not the root wrapper |

Files that exist only after rebase:

| Today | Tomorrow | Notes |
|---|---|---|
| `quality_data.py` | `aegis_eval/workflows/grader_quality/quality_data.py` | Keep this filename. |
| `quality_eval.py` | `aegis_eval/workflows/grader_quality/quality_eval.py` plus root wrapper | Keep this filename. Argparse stays on this module. |
| `quality_report.py` | `aegis_eval/workflows/grader_quality/quality_report.py` plus root wrapper | |
| `compare_graders.py` | `aegis_eval/workflows/grader_quality/compare_graders.py` plus root wrapper | Keep this filename. |
| `safety_grader.py` | `aegis_eval/workflows/grader_quality/safety_grader.py` | Importable candidate. Production stays `RefusalGrader`. |
| `jailjudge_validation.py` | `aegis_eval/workflows/grader_quality/jailjudge_validation.py` | Hashed by `execution_identity`. Keep importable. |
| `analyze_screening.py` | `aegis_eval/workflows/grader_quality/analyze_screening.py` | `quality_report` imports `paired_intervals` from here. |
| `hosted_comparison.py` | `aegis_eval/workflows/hosted/hosted_comparison.py` | No root wrapper. `exclusive_run` moves to `core.lock` first. |
| `report_hosted_comparison.py` | `aegis_eval/workflows/hosted/report_hosted_comparison.py` | No root wrapper. |
| `hosted_comparison_37.py` | `experiments/<study>/hosted_comparison_37.py` | Frozen runner copy. Never imported. |
| `experiments/refusal_grader_revised.py` | stay under `experiments/<study>/` | Never imported. Loaded by path if a study needs it. |
| `tests/test_quality_*.py` and friends | `tests/workflows/` matching the module | Update the live `grader-instrument` CI job in the same slice. |

## Hashed files

Do not move any of these until the identity rewrite in the same commit.

`compare_graders.execution_identity` today hashes, relative to `Path(calibrate.__file__).parent`:

```text
evaluators.py  providers.py  config.py  calibrate.py
graders.py  jailjudge_validation.py  benchmarks/jbb.py
```

plus `harness_sha256` of `compare_graders.py` itself.

`quality_eval` today hashes `ROOT/{quality_eval,quality_data,safety_grader,refusal_grader}.py` and `inspect.getsource` of `SafetyJudgment`, `SafetyGrader` methods, and `render_transcript`.

`hosted_comparison` today sets `runner_sha256 = sha256(Path(__file__).read_bytes())` and compares the whole identity dict for resume.

After the rewrite, those keys still use the same basenames. The values come from the package module file, found with `importlib.util.find_spec`. Wrapper files are not hashed.

## Provenance and frozen artifacts

Complete quality checkpoints are sealed. Incomplete resume may ignore `evaluators.py` and `quality_eval.py` hashes. Complete resume may not.

A later evaluation run is a new instrument:

```text
PROVENANCE_VERSION = "aegis_eval.1"
OUT = BASE_DIR / "output" / "grader-quality" / PROVENANCE_VERSION
```

Hosted ledgers use `BASE_DIR / "output" / "hosted-comparison" / PROVENANCE_VERSION`. Leave `output/grader-quality/*.json` and `output/hosted-comparison/hosted-judge-dev.json` untouched.

`load_variant` must make this work without editing `output/`:

```python
# output/false-positive-ablation/refusal_grader_baseline.py
from evaluators import JudgeParseError
from graders import Screen, Verdict
```

Register those names on `sys.modules` before `exec_module`. The same aliases cover the other sealed copies under `output/` that import `evaluators` or `graders`.

## BASE_DIR

Today `config.BASE_DIR` is `Path(__file__).resolve().parent` because `config.py` lives at the repo root. After the move that expression would point at `aegis_eval/core/` and every path would break.

```python
# aegis_eval/core/config.py
PACKAGE_DIR = Path(__file__).resolve().parent.parent  # .../aegis_eval
BASE_DIR = PACKAGE_DIR.parent                         # repo root
OUTPUT_DIR = BASE_DIR / "output"
CASES_PATH = BASE_DIR / "test_cases.json"             # retarget in the data slice
```

In the data slice only, change `CASES_PATH` to `BASE_DIR / "data" / "test_cases.json"`. Do not point it at `data/` while the JSON still sits at the repo root.

`OUTPUT_DIR.mkdir(exist_ok=True)` stays. `load_dotenv()` stays as it is today (cwd search). Do not start requiring `BASE_DIR / ".env"`.

Every `BASE_DIR / "test_cases.json"` becomes `config.CASES_PATH`. Grep for `test_cases.json`. Known callers: `scenarios._load_secret_guardian_cases`, `tests/test_dataset.py`, `tests/test_runner.py`.

## Wrapper contract

Root files are command wrappers. They do not contain argparse, suite logic, or calibration math.

Today `run.main` calls `sys.exit(...)`. After the move, `aegis_eval.cli.run.main` returns the exit code. The wrapper exits. Tests can call `main` without killing the pytest process.

```python
# run.py
from aegis_eval.cli.run import main

if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# calibrate.py
from aegis_eval.workflows.grader_quality.calibrate import main

if __name__ == "__main__":
    raise SystemExit(main())
```

After rebase, `compare_graders.py`, `quality_eval.py`, and `quality_report.py` use the same wrapper shape against `aegis_eval.workflows.grader_quality.<module>.main`.

`python run.py --help` and `python calibrate.py --help` still work.

Do not leave root shims such as `from aegis_eval.harness.runner import *`. Tests switch to package imports. CLI tests import `aegis_eval.cli.run` and monkeypatch that module. The public compatibility surface is the typed commands, not `from runner import run_suite`.

## The scenarios/runner cycle

`scenarios._load_secret_guardian_cases` imports `runner.load_test_cases` inside the function to avoid a circular import. Extract `load_test_cases` (and the validation it owns) to `aegis_eval/harness/cases.py`. Both `runner` and `scenarios` import that module. No lazy import remains.

## Import-layer test

New file `tests/test_import_layers.py`. Parse `aegis_eval/**/*.py` with `ast`, including `ast.ImportFrom.level` relative imports. Fail if:

- `aegis_eval.core` imports `aegis_eval.harness`, `.workflows`, `.cli`, `aegis_eval.reporter`, `tools`, or `experiments`
- `aegis_eval.reporter` imports `aegis_eval.harness`, `.cli`, or `.workflows`
- `aegis_eval.benchmarks` imports `aegis_eval.harness`, `.cli`, `.workflows`, or `aegis_eval.reporter`
- `aegis_eval.harness` imports `aegis_eval.cli`, `.workflows`, or `aegis_eval.reporter`
- `aegis_eval.workflows.grader_quality` imports `aegis_eval.workflows.hosted` or `aegis_eval.cli`
- `aegis_eval.workflows.hosted` imports `aegis_eval.cli`
- any `aegis_eval` module imports `tools` or `experiments`

No new dependency. The stdlib `ast` module is enough.

## pytest and CI

`pytest.ini` stays:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
pythonpath = .
```

This branch's `.github/workflows/ci.yml` still runs `pytest -q`. After rebase, the live `grader-instrument` job's hardcoded `tests/test_quality_eval.py` paths update in the same slice that moves those tests.

README commands stay `python run.py ...` and `python calibrate.py ...`. The "Project layout" table updates in the last slice to the package tree. The `test_cases.json` "Adding test cases" heading becomes `data/test_cases.json`.

## Migration sequence

Do not start this list until the grader-quality snapshot is committed on the shared checkout and this branch has been rebased onto that commit. Record `pytest --collect-only -q` on the rebased tree. That count, plus import-layer tests, is the bar.

One slice per commit. Run `pytest -q` after each. Stop if anything fails.

1. **Baseline.** Collect-only on the rebased tree. No code moves.
2. **Skeleton.** Add empty `aegis_eval/` packages, `data/` (do not move the JSON yet), `docs/architecture.md` (short pointer at this spec plus the dependency diagram), optional editor exclusions for `venv/`, `__pycache__/`, `.pytest_cache/`, `.cache/`, `output/`. Add `tests/test_import_layers.py` against the empty packages so the rule exists before code arrives.
3. **`core.lock`.** Extract `exclusive_run` from `hosted_comparison.py` into `aegis_eval/core/lock.py`. Point `quality_eval` and hosted code at it. This commit does not move hashed files yet.
4. **Identity rewrite.** Add `provenance_version`, `find_spec` hashing, versioned `OUT` paths, and `load_variant` module aliases. Point new runs at `output/grader-quality/aegis_eval.1/` and `output/hosted-comparison/aegis_eval.1/`. Do not read or write the sealed files in the unversioned directories except as immutable inputs.
5. **Non-hashed moves.** `reporter.py`, `attackers.py`, `runner.py` minus the loader, `scenarios.py`, `target.py`, `cases.py` extract, `cli/run.py` body. Update callers. `CASES_PATH` still points at repo-root `test_cases.json`.
6. **Hashed moves, same series, after step 4.** `config`, `providers`, `evaluators` (judge I/O only), `graders` (now including leak/profanity/rubric), `refusal_grader`, `benchmarks/`, `calibrate`, and after rebase the quality and hosted modules listed in the file map. Delete remaining root shims in the wrapper slice, not before `load_variant` aliases exist.
7. **data.** Move `test_cases.json` to `data/test_cases.json`. Point `CASES_PATH` at it. Delete the old file in the same commit so there is one suite.
8. **CLI wrappers.** Replace root `run.py`, `calibrate.py`, `compare_graders.py`, `quality_eval.py`, and `quality_report.py` with the wrappers. Change `run.main` to return an int. Delete every remaining root shim.
9. **README and CI.** Update the layout table, the test-cases path, and the `grader-instrument` test paths. Do not rewrite the rest of the README.

Temporary one-line root re-imports are allowed only while a later slice in this same series still needs them. Prefer updating callers in the same commit. `from aegis_eval.core.config import *` is too wide. Allowed temporary shim: `import aegis_eval.core.config as config` in not-yet-moved modules, deleted in the next slice.

## Verification

Run from the worktree root after every slice:

```text
python -m pytest -q
python -c "import run, calibrate; print('wrappers ok')"
python run.py --help
python calibrate.py --help
git diff --check
```

After the wrapper slice, also run `--help` on `compare_graders.py`, `quality_eval.py`, and `quality_report.py`.

Done means all of the following.

- Collect count is the rebased baseline plus the new import-layer tests.
- Every previous assertion still exists. Import lines may change. Expected values may not.
- `python run.py --help` prints the same flags as today, including `--positive-control` and `--attacker`.
- `aegis_eval.cli.run.main` no longer calls `sys.exit`. Exit codes 0, 1, 2, 3 still come out of `python run.py`.
- CLI tests patch `aegis_eval.cli.run`, not the root wrapper.
- `config.BASE_DIR` is the directory that contains `data/`, `aegis_eval/`, and `run.py`.
- `config.CASES_PATH` exists and loads the same cases `test_dataset.py` already counts (45 to 60 after `load_test_cases`).
- New quality or hosted runs write under `output/.../aegis_eval.1/`. Sealed files in the unversioned directories are untouched.
- `git diff --check` is clean.
- The diff contains no `output/`, `.cache/`, `venv/`, `.env`, or raw JBB/JAILJUDGE/XSTest payload.
- `tests/test_import_layers.py` fails if someone adds `from aegis_eval.harness import runner` inside `core`, or `from aegis_eval.workflows.hosted import ...` inside `grader_quality`.

A clean clone with `pip install -r requirements-dev.txt` must be enough to run the offline suite. Ignored directories are not on the import path.

## Consequences

- Editors and agents look under `aegis_eval/` instead of 13 root modules.
- `from runner import run_suite` stops working. That is intended.
- Old quality-eval and hosted checkpoints stay on disk and cannot resume. A later run uses `aegis_eval.1` and a new directory. A moved judge is a new instrument.
- Two worktrees stay independent until the snapshot commit and the rebase. Do not cherry-pick grader-quality files into this branch to get ahead.
- Do not start `quality_eval.py` against the sealed `output/grader-quality/` files from reorganized code.

## Out of scope

- Changing a grader, prompt, dataset, or gate.
- Starting `quality_eval.py` or wiping `output/grader-quality/`.
- Consolidating the hosted-comparison scripts.
- Deleting historical `docs/superpowers/` plans (overridden after implementation: old 2026-07 and 2026-09-07/08 plans and specs were removed; this layout spec stayed).
- Adding setuptools, poetry, or an import-linter package.
