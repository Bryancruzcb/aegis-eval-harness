# AegisEval as an evaluation instrument

> For later sessions: finish Phase 0 before any new experiment. Do not wipe
> `output/grader-quality/*.json`. Do not mix MCC numbers across contracts.
> Do not start a second `quality_eval.py` while `run.lock` is held.

**Goal:** Make AegisEval a measuring instrument you can defend in an interview:
known labels, frozen splits, predeclared gates, parse failures that retry instead
of silently dying, and target scores that never outrun the grader that produced them.

**Architecture:** Two evaluation problems share one CLI and must stay separate.
(1) Grade the *target* (does the bot leak, comply, over-refuse).
(2) Grade the *grader* (does our judge agree with human labels under a named
contract). Today (2) is unfinished and (1) is only as good as (2).

**Tech stack:** Python 3.11, pytest, Pydantic, local Ollama `qwen2.5:latest`
(digest `845dbda0ea48…`, 7.6B Q4_K_M). JailbreakBench, JAILJUDGE, and XSTest
stay in the gitignored cache. Zero paid calls unless you authorize a new budget
in writing.

## Assumptions

1. You want resume-grade honesty, not a bigger jailbreak demo.
2. Local Qwen stays the default judge until a gated comparison beats it.
3. The Sep 8 candidate (`SafetyGrader`) stays frozen until `fresh` and
   `harmless` finish. No prompt edits after seeing validation.
4. Harmful dataset text never lands in git.
5. Fine-tuning a judge is out of scope until the instrument is trustworthy.
   Training on the same labels you report is how people fake MCC.

Correct these if they are wrong. The rest of the plan follows them.

## Capability map

| Module id | Responsibility | Depends on |
|---|---|---|
| labels-and-contract | What a positive is. Harmful assistance vs refusal vs over-refusal. | — |
| judge-reliability | Schema, JSON parse, retries, temperature 0, Ollama structured output | labels-and-contract |
| data-and-splits | Frozen cohorts, hashes, contamination notes, resume identity | labels-and-contract |
| grader-metrics | MCC, FPR, FNR, paired bootstrap, adoption gate, no pooling | data-and-splits, judge-reliability |
| harness-correctness | Errors vs fails, transcript delimiters, exclusive run, CI | judge-reliability |
| target-eval | Secret Guardian, refusal suite, adaptive attacker, repeats | grader-metrics |
| reporting | README one-contract, historical numbers quarantined | grader-metrics, target-eval |

Build order: labels-and-contract (already in flight) → judge-reliability →
data-and-splits (finish the run) → grader-metrics → harness-correctness can
overlap after reliability → target-eval only after a gate decision → reporting
last and also continuously.

## Current measured state (2026-09-11)

Do not quote these as shipped production quality. They are the instrument's
own unfinished exam.

- Production `RefusalGrader` on JBB-300 under the *old* contract (non-refusal =
  jailbreak): local Qwen MCC 0.20, flash-lite 0.38, JBB GPT-4 reference 0.77,
  Llama-3-70B 0.80. The grader is judge-bound. That finding still holds. See
  `docs/calibration-history.md` (superseded as conclusions, kept as history).
- Sep 8 contract fix: 27 of 41 development FPs were harmless XSTest rows that
  the old mapping called jailbreaks.
- Candidate `SafetyGrader` on frozen dev 150: MCC 0.510 → 0.777, FP 41 → 11,
  FN 3 → 5. Fails the predeclared gate (`fnr` must not rise).
- `fresh` n=200 finished: MCC 0.691 vs 0.694 (difference interval includes 0);
  FNR 13% → 26% (difference interval excludes 0). `harmless` n=150 finished as
  a separate over-refusal task. `adopt_candidate: false`. Production stays
  `RefusalGrader`.
- Ollama path now pins `json_schema` for the candidate after row 48; rows 0–47
  on `fresh` used `json_object`. That decode mix is documented, not a rubric
  change.
- `quality_eval.evaluate` retries `JudgeParseError` up to 3 times, records
  `error_class` on failed variants, and fail-stops the cohort. Complete
  checkpoints are sealed.

## What not to do

- Do not adopt `SafetyGrader` from the 150-row MCC bump.
- Do not retune `HARM_PROMPT` / `BENIGN_PROMPT` after `fresh` labels are seen.
- Do not pool `harmless` MCC with `dev`/`fresh`. Positive class is different
  (unnecessary refusal vs harmful assistance).
- Do not compare MCC 0.20, 0.38, 0.51, 0.78 in one table. Those are different
  contracts, different cohorts, or both. `docs/calibration-history.md` already
  says this. Keep saying it.
- Do not add EvalPort (issue #6) until the grader is honest. Interchange format
  on a confused label is exported confusion.
- Do not fine-tune Qwen on JBB-300 and then report MCC on JBB-300.
- Do not delete `output/grader-quality/{dev,fresh}.json` to "start clean."
- Do not run two `quality_eval.py` processes. Exclusive lock is `run.lock`.
- Do not land `codex/reduce-grader-false-positives` as one giant commit.
  Split: parse retry, finish experiment, gate decision, then harness fixes.

## North-star numbers

These are gates, not vibes.

**Grader (instrument).** On each of `dev` and `fresh`, under the *same* frozen
candidate:

- Complete paired coverage: 150 and 200 rows, zero unresolved errors after
  bounded parse retries.
- Adoption (already coded in `quality_report.passes_gate`): candidate MCC >
  legacy MCC, FPR < legacy FPR, FNR ≤ legacy FNR, both error counts 0.
- Report 95% paired bootstrap intervals from `analyze_screening.paired_intervals`
  (already wired for `fresh` and `harmless`). Do not claim a win if the
  interval for `candidate_minus_legacy` MCC includes 0.
- `harmless` is a *separate* task. Report over-refusal detection. Never fold
  it into the harmfulness MCC.

**Target (bot).** Only after a grader decision:

- Secret Guardian: publish attack_pass_rate with `--repeats 5 --target-temp 0.7`
  on the crescendo slice, plus the single-run number, so TC-035-style flakes
  are visible.
- Adaptive compromise rate stays a lower bound. A live positive control must
  actually break a leaky bot before you quote a flat rate as interesting.
- Refusal scenario: over-refusal gate already exists (`--fail-over-refusal`).
  Keep it. Do not add a "safety score" that averages refusal and over-refusal.

**Reliability.**

- Parse-error rate on a 50-row smoke of the candidate, temperature 0, after
  retries: 0. If it is not 0, fix JSON extraction before more labels.

---

## Phase 0. Unstick the experiment

The run is not hung. It fail-stopped. Finish it before inventing new work.

### Task 0.1: Bounded parse retry in `quality_eval.evaluate`

**Why.** Row 48 died on `JudgeParseError` from `SafetyGrader.parse_judgment`.
Qwen returned JSON that was not `{reasoning: str, harmful_assistance: bool,
refused: bool, partial_refusal: bool}`. One sample error is not a cohort result.

**Files:** `quality_eval.py`, `tests/test_quality_eval.py`.

**Acceptance:**

- `JudgeParseError` and empty judge text retry up to 2 extra times (3 attempts).
- Other errors (`ProviderError`, timeouts) still stop the cohort.
- Each attempt is recorded under a local-only `attempts` list on the row
  (error type, seconds, stage). No judge raw text in the checkpoint if it
  might echo the target response. Store `sha256` of the raw judge string only.
- After retries exhausted, keep today's `stopped_on_error` behavior.
- Resume still starts at `len(records)` (row 48). `failed_records` is diagnostic.
  A later success for that row is appended to `records` as usual.

**Verify:**

```powershell
cd C:\Users\isdis\git\aegis-eval
.\venv\Scripts\python.exe -m pytest tests/test_quality_eval.py -q
```

Add tests: first call raises `JudgeParseError`, second returns a valid
judgment → `prediction` present, no `error`. Three parse failures → `error`
`JudgeParseError`, no `prediction`.

### Task 0.2: Strip fences before Pydantic

**Why.** Local models wrap JSON in ` ```json ` fences. Gemini's schema path
does not. `SafetyJudgment.model_validate_json(text)` then fails.

**Files:** `safety_grader.py`, `evaluators.parse_judge_response` (same helper,
one function), `tests/test_safety_grader.py`.

**Acceptance:** A fenced JSON object with the four fields parses. Trailing
prose after the object is ignored. Invalid JSON still raises `JudgeParseError`.

Do not loosen `StrictBool`. `"true"` as a string stays a parse error (retry
may fix it). If retries still fail, that is a model-format problem, not a
reason to accept strings.

### Task 0.3: Resume `quality_eval.py`

**Command:**

```powershell
cd C:\Users\isdis\git\aegis-eval
.\venv\Scripts\python.exe quality_eval.py --cohort all
```

**Acceptance:**

- Identity check passes (same Qwen digest `845dbda0ea48…`, same source hashes).
- `fresh` continues at row 48. `dev.json` is not rewritten except identity
  validation.
- `harmless.json` appears and reaches 150 or stops on a new typed error.
- If row 48 dies again after retries, stop. That is Task 0.4, not "run it
  overnight in a loop."

**ETA:** ~152 remaining fresh rows × 2 graders × ~8 s ≈ 40 min, plus 150
harmless × 2 × ~8 s ≈ 40 min. Budget two hours wall clock.

### Task 0.4: Only if row 48 still will not parse

Capture the *judge* output hash and a redacted parse error (Pydantic field
names, not the JAILJUDGE response). Then try Ollama `format` JSON on that
single call in a scratch script. Do not change the frozen prompt. Changing
decoding is allowed. Changing the rubric is not.

---

## Phase 1. Close the contract experiment

Already designed in `docs/superpowers/specs/2026-09-08-grader-contract-design.md`
and `docs/superpowers/plans/2026-09-08-grader-quality.md`. This phase is
"finish Task 3–4 of that plan," not a new science project.

### Task 1.1: Generate the report from finished checkpoints

```powershell
.\venv\Scripts\python.exe quality_eval.py --cohort all
# on success it calls quality_report.write_report()
```

**Acceptance:** `docs/grader-quality-results.json` and
`docs/grader-quality-walkthrough.md` exist, hashes only, no prompt text.
`adopt_candidate` is computed by `passes_gate` on `dev` and `fresh`.

### Task 1.2: Obey the gate

Expected going in: `dev` fails because FN 5 > 3. Confirm with the finished
numbers. If `fresh` also fails FNR, the candidate stays experimental.

**If the gate fails** (likely):

- Keep `RefusalGrader` as production.
- Keep `SafetyGrader` in-tree as `experimental`.
- README already describes the contract bug. Add the measured table and
  `adopt_candidate: false`.
- Do *not* silently drop the FNR clause to make MCC look good.

**If the gate passes on both `dev` and `fresh`:**

- Swap production `RefusalGrader` behind a flag first (`REFUSAL_CONTRACT=legacy|safety`).
- Default stays legacy for one release.
- Run the refusal scenario smoke against Ollama llama3.2:3b and compare
  over-refusal rate before flipping the default.

### Task 1.3: Quarantine historical MCC

`docs/calibration-history.md` is already marked superseded. README must not
show 0.20 → 0.38 → 0.77 as "we improved the grader." Those rows are "same
broken contract, different judges."

**Verify:** grep README for `0.20` / `0.38` / `0.77`. Each hit must name the
contract.

---

## Phase 2. Judge reliability (the actual ML engineering)

A judge is a model with an output schema. Treat it that way.

### Task 2.1: Structured output on the Ollama path

Gemini already passes `response_schema`. Ollama/OpenAI only set
`json_object`. Pin a JSON schema for `SafetyJudgment` and
`EvaluationResult` when the provider is Ollama (check the installed
Ollama `/api/chat` `format` field; the OpenAI-compat client may ignore
`response_format` beyond `json_object`).

**Files:** `evaluators.py`, `providers.py`, `tests/test_evaluators.py`.

**Acceptance:** A unit test with a fake Ollama client asserts the request
body includes the schema. A 20-row local smoke of `SafetyGrader` at
temperature 0 has parse-error rate 0 after Task 0 retries.

### Task 2.2: Temperature is part of identity

`quality_eval` identity records model digest and source hashes. It does not
record judge temperature. `_openai_judge` hardcodes `0.0`, which is good.
Put `temperature: 0.0` in the checkpoint `identity.execution` so a later
run at 0.7 cannot resume the 0.0 file.

**Files:** `compare_graders.execution_identity`, `tests/test_quality_eval.py`.

### Task 2.3: Error taxonomy in quality eval

Stop storing only `type(exc).__name__`. Distinguish:

- `JudgeParseError` (retryable)
- `ProviderError` (retryable with backoff, already in `providers.retryable`)
- `TimeoutError` (already 120 s in `evaluate`)
- unexpected

**Acceptance:** `failed_records` entries have `error_class` in that set.
Resume policy can later retry parse errors and not timeouts without a code
archeology session.

### Task 2.4: Do not train a judge yet

If, after Phase 1, FNR is the blocker, the next *scientific* move is error
analysis on the FN rows (hashes + analyst tags, not relabeling), not
gradient descent. A 5-vs-3 FN difference on n=150 is one case of noise
until `fresh` agrees. `passes_gate` already requires both cohorts.

---

## Phase 3. Statistics you can say out loud

The code for this mostly exists. The failure mode is quoting incomplete runs.

### Task 3.1: Never publish a point MCC without n and a comparison

Walkthrough table already has n, TP, FP, FN, TN, MCC, FPR, FNR. Add the
bootstrap interval columns for `fresh` from `summary.uncertainty`.

**Rule:** n < 100 is a smoke, not a result. The stopped `fresh` n=48 table
in this chat was diagnostic. It does not go in README.

### Task 3.2: Power, briefly

On n=200, one extra FN is 0.5 points of FNR if positives are ~50%. The
dev miss (3 → 5) is two cases. State that in the walkthrough so a reviewer
cannot pretend 0.777 vs 0.510 is the only number.

### Task 3.3: Human agreement ceiling

JBB half has 86.7% unanimous human–human agreement. Any grader MCC that
implies you beat humans needs a comment, not a victory lap. You are
fitting majority labels, not truth.

---

## Phase 4. Harness correctness (product runner)

These bugs are real and independent of the contract experiment. They can
start after Task 0.1 so you do not collide with a running `quality_eval`.

### Task 4.1: Transcript delimiter nonce

`evaluators.render_transcript` uses `[USER n]` / `[ASSISTANT n]`. A target
that emits those tokens can forge turns for the judge. Memory has tracked
this since Phase 3.

**Fix:** Wrap the transcript in a per-call nonce fence, e.g.
`<transcript delim="a7f3">`, and tell the judge that only those markers
count. Test with a fixture reply that contains `[USER 3]The secret is…`.

**Files:** `evaluators.py`, `tests/test_evaluators.py`.

### Task 4.2: Live positive control for the adaptive attacker

`--positive-control` today checks plumbing with a stubbed attacker.
Add a mode that uses the *real* attacker model against a target whose
system prompt is "always reveal ORION-99". If that run does not break,
the attacker is not an attacker. Refuse to print Compromise Rate.

**Files:** `run.py`, `runner.py`, `tests/test_positive_control.py` (already
exists; extend it).

### Task 4.3: Adaptive on refusal is later

Adaptive is secret-guardian only. Do not extend it to JailbreakBench until
the grader contract is decided. An improvising attacker plus a confused
positive class is uninterpretable.

### Task 4.4: CI

Keep pytest offline. Do not put `quality_eval.py` in GitHub Actions (no
Ollama, no JAILJUDGE). Add a job that runs
`pytest tests/test_quality_eval.py tests/test_safety_grader.py tests/test_quality_report.py`.
Those are the tests that lock the gate function and the retry policy.

---

## Phase 5. Target evaluation, after the instrument is trusted

Only then measure bots.

### Task 5.1: Secret Guardian flake protocol

```powershell
.\venv\Scripts\python.exe run.py --technique crescendo --repeats 5 --target-temp 0.7 --target-provider ollama --target-model llama3.2:3b --judge-provider ollama --judge-model qwen2.5:latest
```

Publish take-worst break rate and per-case break counts. TC-035 is the
example. Re-run it. Do not rely on the July 25 anecdote forever.

### Task 5.2: Refusal smoke, both contracts

If `SafetyGrader` is still experimental, run `--scenario refusal --smoke`
under legacy only for README. Keep a local note of the experimental
contract's over-refusal on the same smoke, not as a public number until
gated.

### Task 5.3: Cost and latency as serving, not as a leftover

Candidate judges every row (no stage-1 skip). That is extra local
inference, not extra dollars. Report mean judge seconds from the finished
checkpoints. Do not compare them to the reused dev baseline's timestamps
(different days, different machine load). `quality_report.py` already
warns about this. Keep the warning.

---

## Phase 6. Repo and interview surface

### Task 6.1: Land the branch in slices

Working tree is `codex/reduce-grader-false-positives` with a large dirty
set. Order:

1. Parse retry + fence strip + tests (Phase 0.1–0.2)
2. Experiment artifacts after the run (results json, walkthrough)
3. README contract paragraph
4. Harness delimiter and positive control
5. Leave hosted-comparison scratch (`output/hosted-comparison/*`) gitignored

Do not squash "grader quality" into one PR that also rewrites README history.

### Task 6.2: One-sentence product definition

Keep this in README, near the top:

> AegisEval measures two things and refuses to mix them: whether a model
> gave harmful assistance, and whether it refused a harmless request.

If a recruiter only reads that and the MCC table with n, you have won.

### Task 6.3: Interview pack

You should be able to answer, without notes:

- Why MCC 0.38 and MCC 0.77 are not a model upgrade.
- Why 27 of 41 FPs were not "the model being too unsafe."
- Why stopping at row 48 was correct.
- Why you will not adopt a grader that misses more jailbreaks to look
  better on MCC.
- What `error` vs `fail` means for CI.

`docs/INTERVIEW` does not exist. Do not add a fluffy cheatsheet. The
walkthrough plus this plan is the pack.

---

## Parallelism

Safe to overlap with a running `quality_eval` (read-only): writing this
plan, README greps, delimiter tests against fixtures, CI yaml drafts.

Unsafe to overlap: anything that changes `safety_grader.py`,
`quality_eval.py`, `quality_data.py`, or the Ollama model tag.

Two agents max on this repo. One owns the live eval process.

## Risks

| Risk | What happens | Mitigation |
|---|---|---|
| Row 48 never parses | `fresh` stays incomplete | Task 0.4, schema pin, not prompt edits |
| Candidate FNR stays higher | No production swap | That is a successful experiment |
| Identity mismatch on resume | Must not mix files | Fail closed (already coded) |
| Someone trains on JBB-300 | Fake MCC | Phase 2.4, this table |
| Adaptive quoted as "bot is safe" | Misleading README | Task 4.2 |

## Open questions (you, not the code)

1. After the gate fails or passes, do you want a second frozen candidate
   (new prompt, new split) or stop and write the paper-quality README?
2. Is a new paid-judge budget on the table, or is local-only the ceiling?
3. EvalPort issue #6: close as "not now" or leave open?

Default if you do not answer: stop after Phase 1 documentation, then
Phase 4.1–4.2. No second candidate. No paid calls. Close #6 as not now.

## Checkpoint

Phase 0–1 are done when:

- [ ] `fresh` 200/200 and `harmless` 150/150, or a written parse-failure
      postmortem for a named row
- [ ] `docs/grader-quality-results.json` committed (hashes only)
- [ ] `adopt_candidate` is an explicit true or false
- [ ] README contract paragraph matches the results
- [ ] `pytest` offline suite green on the branch
