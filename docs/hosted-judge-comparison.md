# Hosted judge comparison

> Contract correction (2026-09-08): JBB judge-comparison includes 100 harmless XSTest examples. The legacy mapping of non-refusal to jailbreak misgrades benign compliance. The measurements below are preserved historical results under that contract; they do not establish a general model ranking. See [the contract plan](superpowers/plans/2026-09-08-grader-quality.md) and [error audit](grader-error-audit.json). Both JBB halves have been inspected during development.

Update 2026-09-08: the additional Gemini 3.7 Flash trial is complete. See [the three-model comparison](gemini-37-comparison.md). On 149 matched rows its MCC was 0.321, FPR 65.62%, FNR 5.66%; Qwen remains the recommendation. Total known token-cost estimates across all trials are now $0.282645. The 3.7 runner settles verified usage while retaining unknown-call allowances; its budget-accounted total is $1.224901 within the original $5 cap. The sections below preserve the completed 3.5 experiment and its historical accounting.

## Scope and authorization

The user authorized a total API spending cap of $5 for this comparison. The experiment uses the same 150 JBB development responses and frozen production rubric as `output/false-positive-ablation/qwen-dev.json`. JAILJUDGE data is not sent to hosted providers.

The existing local baseline remains a previous run, not a simultaneous model trial. Judge-specific schemas, thinking settings and serving infrastructure differ. This measures the deployed judge configurations rather than isolating model weights alone. Neither this development set nor the previously inspected JBB half is an untouched final test.

## Candidate selection

The configured Gemini key listed Gemini 3.8 Flash and 3.5 Flash. No OpenAI key was configured. A model listing alone does not guarantee usable quota or reliable requests.

Gemini 3.8 Flash was tried first at its introductory $0.75 input / $3.75 output price per million tokens. A basic probe succeeded, but grading calls repeatedly returned HTTP 503 high-demand errors. That trial was stopped; its few successful rows are not an accuracy benchmark or the basis for model ranking.

The fallback is Gemini 3.5 Flash, priced at $1.50 input / $9 output per million tokens. [Google's pricing page](https://ai.google.dev/gemini-api/docs/pricing) includes thinking tokens in output charges. Settings are temperature 0, low thinking, 2,048 maximum output tokens, native JSON schema and the unchanged grader prompt. The fallback uses the existing two-stage route; it does not rerun the screening-bypass experiment.

## Spending and recovery

`hosted_comparison.py` keeps one spending ledger in `output/hosted-comparison/hosted-judge-dev.json`. It carries forward all reservations from the interrupted 3.8 trial, including diagnostic probes. It reserves an input allowance equal to UTF-8 prompt bytes plus 8,192 tokens of overhead and the full output allowance before every request. No reservations are refunded, including failed or interrupted attempts. This is a conservative cost allowance at the verified rates, not a provider-enforced account billing limit.

The client makes one SDK attempt per call. Up to two explicitly reserved retries are permitted for transient errors. Explicit daily quota errors stop without short retries. An exclusive process lock prevents two CLI runs from spending against stale ledger balances. Infrastructure failures remain in the failure history and do not advance the resume position. Terminal ungradable responses advance as explicit error rows and yield complete_with_errors after all examples are attempted. The run refuses a changed source/configuration fingerprint. The final status/quota-handling fix was audit-migrated into the checkpoint fingerprint; it changes no grading request or saved prediction.

Usage-based estimates include prompt, candidate and thinking tokens. A failed request without usage metadata has unknown billing, covered by its retained reservation. The provider billing console remains authoritative; absence of usage is not proof of zero charge.

Independent review identified resume coverage and concurrent-budget bugs in the first implementation. Both were fixed and regression-tested before the fallback run. The active 3.8 process was stopped before launching the locked 3.5 run. Its historical ledger and runner revision notes remain in ignored output storage.

## Result

All 150 JBB development examples were evaluated. Gemini produced usable grades for 149 and returned `BlockedReason.PROHIBITED_CONTENT` for one. That response remains an explicit evaluation error. The run status is `complete_with_errors`, not incomplete or fully resolved.

The accuracy comparison uses the same 149 resolved examples for both judges:

| Two-stage judge | TP | FP | FN | TN | MCC | FPR | FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local Qwen 2.5 | 51 | 41 | 2 | 55 | 0.527 | 42.71% | 3.77% |
| Gemini 3.5 Flash | 49 | 62 | 4 | 34 | 0.306 | 64.58% | 7.55% |

Qwen resolved 150/150, while Gemini resolved 149/150. The excluded example was a local Qwen false negative. Consequently Qwen's full-cohort result remains MCC 0.510, FPR 42.71%, FNR 5.56%; its higher MCC in the matched table is a cohort change, not an additional improvement. The ungraded Gemini response is not counted as safe, harmful or correct.

**Choose local Qwen for this refusal grader.** Gemini 3.5 made 21 more false positives and missed two more jailbreaks on the matched cohort, and lost one row of coverage. It fails the predeclared adoption gate. This is a result for the frozen rubric and settings on development data, not a claim that Qwen is generally the more capable model. Gemini 3.8 remains unranked because capacity failures prevented a useful trial.

Each route made 63 deterministic decisions and 87 logical judge evaluations. Qwen had 87 successful judge responses and Gemini had 86. Mean successful judge-evaluation times were 4.490 seconds for Qwen and 4.722 seconds for Gemini. These exclude failed attempts and retry waits and come from different run periods. The generic `judge_mean_seconds` field includes terminal errors; use the separately reported `successful_judge_mean_seconds` for this comparison.

No defaults were edited. The existing environment selects `gemini-flash-lite-latest`, which was not the hosted model evaluated here. To explicitly use the measured local configuration, pass `--judge-provider ollama --judge-model qwen2.5:latest` to calibration or the runner.

## Spending and completion

The user added $10 to the Gemini account, while this experiment retained its original $5 authorization. Total known usage across both model trials is **$0.214077 at published token rates**, not a verified billing charge. Earlier free-tier calls may be unbilled. All failed/interrupted allowances remain reserved: **$3.984035**, including 32 attempts without usage metadata. The run stayed below its cap. We did not inspect the account's remaining credit balance.

The earlier daily-quota interruption and 64-row partial result are preserved in `hosted-judge-partial-results.json`. They are superseded by `hosted-judge-results.json`. The latter records all 150 evaluated rows, matched-cohort metrics, the unresolved row, model/settings hashes, usage estimates and provider-error history. Per-row predictions and the shared ledger remain in ignored output storage. No additional billing setup or quota wait is needed to complete this comparison.

A completed rerun command exits without making new requests. Its exit code is 1 because one row was ungradable; it does not mean there are still unattempted examples. Further tuning or another model trial requires a separately frozen comparison and must account for the existing budget.

## Verification

The offline suite passes **333 tests**. Independent review verified persistent budget carry, process locking, infrastructure-error recovery and explicit terminal-error coverage. Regression coverage distinguishes daily quota exhaustion from retryable minute limits. The production rubric remains byte-identical to the frozen baseline. The grader's accuracy limitations remain; this task establishes which tested configuration to retain rather than claiming the original JBB accuracy targets were met.


## Harmless-response data found

The original [XSTest repository](https://github.com/paul-rottger/xstest) provides 1,250 responses to 250 safe prompts across five target-model configurations. Its model-completion files include two annotation columns and final labels. [Paper section 4.2](https://aclanthology.org/2024.naacl-long.301.pdf) describes manual labeling by the authors and discussion of disagreements.

The downloaded files are pinned to commit `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d`; hashes and label counts are in `xstest-data-inventory.json`. Raw responses remain in ignored `.cache/xstest/`. Prompt licensing is CC-BY-4.0; the repository says model completions retain their original model-provider license terms. No raw files were published or sent to a hosted judge.

This resolves the search for existing human-labeled harmless responses. It does not yet measure AegisEval on them. Before evaluation, freeze a mapping where full compliance passes and full/partial refusal fails for harmless cases; preserve the original three-way labels for per-class analysis. XSTest accepts direct answer attempts without requiring factual correctness, which differs from judging general answer quality. Group evaluation splits and uncertainty by prompt because five responses share each prompt. These are diagnostic challenge cases, not a sample of ordinary user traffic.

## Reproduce

```powershell
.\venv\Scripts\python.exe hosted_comparison.py
.\venv\Scripts\python.exe -m pytest -q
```

The comparison command resumes its existing ledger automatically. Do not delete or reset the ledger to bypass the cap. No repository default, GitHub publication or resume claim is changed by running it.
