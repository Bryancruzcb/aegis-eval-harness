# AegisEval project layout and module-boundary design

## Status

High-level approach approved on 2026-09-11. This written design is proposed for review before implementation planning begins.

## Goal

Make the repository navigable by placing implementation methods behind clear module interfaces, separating runnable workflows from immutable evaluation evidence, and preserving existing command-line entry points.

## Context

The active repository has accumulated the original evaluation harness, grader-quality research, hosted-comparison workflows, scripts, reports, and local artifacts in one flat layout. The public harness modules, research modules, and command entry points are currently interleaved at the repository root. `output/` is intentionally ignored because it contains local data and run artifacts, but it also contains durable helper code and historical source copies. The live grader-quality checkout is shared and contains uncommitted work, so this refactor must not alter that checkout or reuse its checkpoints.

## Constraints

- Work in `codex/organize-project-layout`, not the shared `codex/reduce-grader-false-positives` checkout.
- Preserve the current `output/`, `.cache/`, virtual environment, checkpoints, locks, and source snapshots exactly; do not move, delete, rename, stage, or regenerate them in this refactor.
- Preserve existing root-level commands as thin compatibility wrappers, including `python run.py`, `python calibrate.py`, `python compare_graders.py`, `python quality_eval.py`, and `python quality_report.py`.
- Keep existing behavior, command arguments, exit codes, report formats, and test semantics unchanged.
- Add no runtime dependency solely for this reorganization.
- Do not make an old checkpoint resumable after its implementation has moved. A later evaluation run must use a new provenance version.

## Alternatives considered

### Documentation-only cleanup

Add editor exclusions and a project map but leave the implementation flat. This is low risk but leaves module ownership and imports difficult to follow.

### Immediate wholesale move in the shared checkout

Move all files while grader-quality work remains uncommitted. This would entangle the refactor with active research, risk source-identity changes, and create an unreviewable merge conflict surface.

### Staged package migration in an isolated branch

Create a package layout, move one dependency slice at a time, retain root compatibility wrappers, and verify behavior after each slice. This provides the desired organization without changing the shared experiment. This is the selected approach.

## Decision

Adopt a staged package layout. Root files become small compatibility command wrappers only; implementation moves into `aegis_eval/` modules grouped by responsibility. Reusable behavior lives behind module interfaces, while one-off analyses remain explicit workflows or tools rather than becoming dependencies of the core harness.

### Target layout

```text
aegis_eval/
  core/                 # configuration, provider adapters, target and judge evaluation
  harness/              # scenarios, attackers, case execution, result aggregation
  reporting/            # terminal and HTML rendering from result payloads
  benchmarks/           # benchmark fetch and case conversion
  workflows/
    grader_quality/     # calibration, candidate grading, cohort evaluation, reports
    hosted/             # hosted trial orchestration and cost-ledger reporting
  cli/                  # main() functions used by compatibility wrappers
data/
  test_cases.json       # repository-owned static evaluation cases
experiments/            # versioned, non-imported study snapshots and notes
tools/                  # explicit one-off analysis commands
tests/
  core/
  harness/
  reporting/
  benchmarks/
  workflows/
docs/
  architecture.md
  experiments/
  plans/
  specs/
```

The package names describe ownership, not implementation mechanics. `core` has no dependency on `harness`, `reporting`, `workflows`, `tools`, or `experiments`. `harness` depends on `core` and returns plain result payloads. `reporting` consumes those payloads without importing harness execution. Each workflow may depend on `core` and a small shared workflow utility, but workflow modules must not import command wrappers or experimental snapshots.

### Module interfaces

- `core.providers` owns provider client construction and retry classification.
- `core.evaluation` owns deterministic and LLM-judge evaluation parsing; grader contracts live next to the code that consumes them.
- `core.target` owns provider-neutral target conversation requests.
- `harness.execution` owns case loading, execution, repeats, and summary aggregation.
- `harness.scenarios` owns scenario registration and scenario-specific grader selection.
- `reporting.results` renders or redacts an already-computed payload; it does not decide evaluation outcomes.
- `workflows.grader_quality` owns cohort selection, provenance validation, candidate comparison, and gate reporting. Shared workflow constants and checkpoint validation move to a neutral module so evaluation and reporting do not import each other.
- `workflows.hosted` owns hosted trial configuration and cost ledgers. Common locking/checkpoint mechanics are extracted only when both workflows genuinely use the same interface.
- `tools` contains manually invoked analyses such as screening diagnostics. It is never imported by production or evaluation modules.
- `experiments` preserves frozen candidate/source variants as labeled study inputs. It is never imported transitively by the package.

### Artifact and documentation rules

- `output/` remains an ignored local-artifact root. It may contain generated checkpoints, reports, and data only after the migration boundary.
- Durable scripts or source snapshots currently buried in `output/` are first inventoried and copied into a versioned `experiments/<study>/` or `tools/` location only after their owning study is committed. No raw private data is unignored or committed by this refactor.
- Historical documents are not bulk-moved during the code migration. Add a concise `docs/architecture.md` and a documentation index first; organize historical reports by study only in a separately reviewable documentation change with updated links.
- Editor exclusions may hide ignored runtime directories (`venv/`, `__pycache__/`, `.pytest_cache/`, `.cache/`, `output/`, `.superpowers/`) without changing their contents.

## Migration sequence

1. Establish a clean baseline and capture the current public command behavior with focused smoke tests.
2. Add the package skeleton, data directory, architecture map, and editor-only exclusions without moving artifacts.
3. Migrate the core, harness, reporting, and benchmark slices already present on the clean base, updating imports and tests together; leave root wrappers intact.
4. Stop at the workflow boundary. Do not move grader-quality or hosted workflow files that exist only as uncommitted shared-checkout work.
5. After the shared work has its own committed artifact snapshot, rebase this branch and migrate the grader-quality and hosted workflow slices in separate commits.
6. Move one-off analyses to `tools/` and preserve experiment variants under `experiments/` only after their referenced studies have a committed snapshot.
7. Update README commands, CI paths, and the test tree; run the complete offline suite and command compatibility checks. Do not attempt to resume an old checkpoint from the reorganized code.

## Verification

- Every existing unit test passes after each migration slice.
- The complete offline test suite passes from the package layout.
- Each preserved root command reaches the same `main()` function and retains its exit behavior.
- `git diff --check` passes and the refactor contains no generated artifact, secret, raw benchmark response, cache, or virtual-environment file.
- Static import checks confirm that `core` does not depend on workflow, tool, experiment, reporting, or CLI modules.
- A clean checkout exposes source through the target layout and does not require ignored local directories for imports or tests.

## Out of scope

- Changing scoring contracts, model prompts, datasets, checkpoint contents, or adoption decisions.
- Rerunning local or hosted evaluations.
- Consolidating the nearly identical hosted trial behavior before its current experiment state is versioned.
- Deleting historical artifacts, documentation, or experiments.
