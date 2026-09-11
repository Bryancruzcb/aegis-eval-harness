# Gemini 3.7 Flash comparison

> Contract correction (2026-09-08): JBB judge-comparison includes 100 harmless XSTest examples. The legacy mapping of non-refusal to jailbreak misgrades benign compliance. The measurements below are preserved historical results under that contract; they do not establish a general model ranking. See [the contract plan](superpowers/plans/2026-09-08-grader-quality.md) and [error audit](grader-error-audit.json). Both JBB halves have been inspected during development.

Status: complete_with_errors; all 150 rows attempted. Requested 2026-09-08.

Run `venv\Scripts\python.exe hosted_comparison_37.py` from the repository root. Checkpoint: `output/hosted-comparison/gemini-3.7-flash-dev.json`. Resume with the same command. This experiment freezes the previous runner in a separate file so the completed 3.5 checkpoint and its source identity remain reproducible.

Use the exact same 150 JBB development examples, frozen production rubric, two-stage screening, JSON schema, temperature 0, low thinking, and 2048 output-token cap. Compare all three models on their common resolved rows and report coverage separately. No JAILJUDGE data may be sent to Gemini. No default or resume claims change automatically.

Pricing verified against https://ai.google.dev/gemini-api/docs/pricing on 2026-09-08: standard input $0.75/million, output including thinking $3.75/million through December 2026.

Budget remains $5 total. The prior completed ledger includes both 3.5 and 3.8 attempts. Preserve its original reservations and reported costs. For enforcement, settle calls with known token usage to their recorded cost estimate; retain full reservations for unknown usage. Starting accounted amount is approximately $1.15633 ($0.214077 known usage plus $0.9422555 unknown-call reservations). Reserve before every new call. The sum of all historical reservations is not actual spending and may exceed $5 after settlement. Token estimates are not billing receipts. Both the old and new runner locks are held during execution.

Adoption gate: improved MCC and lower false-positive rate, no higher missed-jailbreak rate, and complete coverage. A capacity or quota failure pauses the run; resume without discarding attempts. A terminal ungradable response is recorded as an error, never a prediction. Development results are not a fresh holdout estimate.

## Measured results

Both hosted trials attempted 150 rows. This table compares the same 149 examples resolved by all three configurations.

| Judge | TP | FP | FN | TN | MCC | False positives | Missed jailbreaks | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen_2_5 | 51 | 41 | 2 | 55 | 0.527 | 42.71% | 3.77% | 150/150 |
| gemini_3_5_flash | 49 | 62 | 4 | 34 | 0.306 | 64.58% | 7.55% | 149/150 |
| gemini_3_7_flash | 50 | 63 | 3 | 33 | 0.321 | 65.62% | 5.66% | 149/150 |

Gemini 3.7 adoption gate passed: False. The gate uses the Qwen/3.7 pair (149/150 resolved), independently of the three-way table. Error rows are excluded from accuracy and remain explicit coverage failures. These are development results for the fixed grader; they do not rank general model capability.

Gemini 3.7 known token cost: $0.068568. All trials known token cost: $0.282645. Budget-accounted total including unknown-call allowances: $1.224900 of $5. These are estimates, not a billing receipt or account balance.

Successful judge-evaluation means (different run periods; failed attempts and retry waits excluded): qwen_2_5: 4.490s, gemini_3_5_flash: 4.722s, gemini_3_7_flash: 2.163s.

The production grader and defaults were not changed. Raw checkpoints remain local. Full machine-readable details are in `gemini-37-results.json`.

### Interpretation

Retain Qwen for this grader. On the matched 149 examples, Gemini 3.7 produced 22 more false positives and missed one more jailbreak than Qwen. Compared with Gemini 3.5 it caught one additional jailbreak but added one false positive; the modest MCC increase does not establish a statistically reliable improvement. Both Gemini models left the same row (zero-based index 77) ungraded.

Qwen on its full 150-row cohort remains MCC 0.510, FPR 42.71%, FNR 5.56%. Its 0.527 MCC in the matched table reflects excluding a false negative, not a new improvement. This was a completed experiment with an explicit error, not a mid-run interruption.

Verification: 340 offline tests passed, independent review checked spending and report logic, and production rubric SHA remained identical to the frozen baseline. The completed resume command exited without changing the ledger or making new inference calls.
