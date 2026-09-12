# GPT 5.6 Sol Ultra — layout review brief

Review `codex/organize-project-layout` against
`docs/superpowers/specs/2026-09-11-project-layout-design.md`.
Write findings into the **Layout write-back** section of
`docs/superpowers/plans/HANDOFF-grok-gpt.md`.

Work in `C:\Users\isdis\git\aegis-eval-organize`. That worktree is already
rebased onto `origin/main` `410cc04` (PR #7). Do not open
`C:\Users\isdis\git\aegis-eval` for this review. Do not rebase again unless
`origin/main` has moved.

## Do not

- Mix MCC across contracts. Do not adopt `SafetyGrader`.
- Wipe `output/grader-quality/{dev,fresh,harmless}.json` or
  `output/hosted-comparison/hosted-judge-dev.json`.
- Start `quality_eval.py`. New runs would write under
  `output/grader-quality/aegis_eval.1/` and would not resume the sealed files.
- Edit files under `output/`. Frozen grader snapshots still say
  `from evaluators import` / `from graders import`. Aliases in
  `compare_graders.load_variant` exist so those files keep loading.
- Re-review the grader-quality experiment. That review is closed in
  `HANDOFF-grok-gpt.md`.
- Force-push.

## Facts already established

- PR #7 is on `main`. This branch's merge-base is that merge commit.
- Snapshot commit `12883f4` was dropped during the rebase. Main already had
  those files.
- `pytest -q` on this worktree: **397 passed** (2026-09-11).
- Root commands: `run.py`, `calibrate.py`, `compare_graders.py`,
  `quality_eval.py`, `quality_report.py`. Each is a wrapper.
- `PROVENANCE_VERSION` is `aegis_eval.1`. Identity hashes use
  `importlib.util.find_spec(...).origin` with stable basenames
  (`evaluators.py`, not a package path).
- Import DAG is enforced by `tests/test_import_layers.py`.
- Historical 2026-07 and 2026-09-07/08 superpowers plans/specs were deleted
  on request. This layout spec stayed. Experiment walkthroughs and JSON stayed.
- Upstream: `origin/codex/organize-project-layout`. PR: https://github.com/Bryancruzcb/aegis-eval-harness/pull/8

## What to review

1. Wrappers still expose the same flags and exit codes. `--help` must not
   write reports or start evals.
2. Frozen `output/*.py` imports still resolve through `sys.modules` aliases.
   Do not rewrite those files to package imports.
3. `config.BASE_DIR` is the repo root after `config.py` moved.
4. New quality/hosted output goes under `aegis_eval.1/`. Sealed unversioned
   checkpoints are untouched.
5. Layer DAG: `core` imports nothing else in this repo. `grader_quality`
   does not import `hosted`. `aegis_eval` does not import `experiments/` or
   `tools/`.
6. Secret Guardian leak/profanity/passcode rubric lives next to
   `SecretGuardianGrader` in `aegis_eval/harness/graders.py`.
7. Diff vs `origin/main` contains no `output/`, `.cache/`, `venv/`, `.env`,
   or raw JBB/JAILJUDGE/XSTest payload.

## How to run

```text
cd C:\Users\isdis\git\aegis-eval-organize
C:\Users\isdis\git\aegis-eval\venv\Scripts\python.exe -m pytest -q
```

This worktree has no local `venv`. Use the shared venv. Do not install into it
for this review.

## After the review

Fill the layout write-back in `HANDOFF-grok-gpt.md`, commit, and push.

If **Blocker for landing** is empty and pytest is green, merge
https://github.com/Bryancruzcb/aegis-eval-harness/pull/8 into `main`
(GitHub merge commit is fine). Bryan is not coming back to merge it.

If there is a blocker, leave the PR open and write the blocker in the
write-back. Do not merge.
