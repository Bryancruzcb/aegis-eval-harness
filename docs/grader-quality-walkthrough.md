# Grader contract experiment

The historical JBB evaluation confused non-refusal with harmful assistance. JBB judge-comparison includes 100 harmless XSTest examples among its 300 rows. Of the historical local development run's 41 false positives, 27 were on these harmless prompts; 11 others came from the compliance-opening shortcut. Human labels were not changed. Analyst error tags are diagnostic hypotheses, not independent labels or adjudications.

The frozen candidate separates harmful assistance, full refusal and partial refusal. Every response receives a local content judgment. Development reuses the legacy baseline; fresh and harmless cohorts independently run both configurations. The result therefore measures a combined contract, prompt and screening change, not a model upgrade.

| Cohort / configuration | N | TP | FP | FN | TN | MCC | FPR | FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev / legacy | 150 | 51 | 41 | 3 | 55 | 0.510 | 42.71% | 5.56% |
| dev / candidate | 150 | 49 | 11 | 5 | 85 | 0.777 | 11.46% | 9.26% |
| fresh / legacy | 200 | 87 | 18 | 13 | 82 | 0.691 | 18.00% | 13.00% |
| fresh / candidate | 200 | 74 | 6 | 26 | 94 | 0.694 | 6.00% | 26.00% |
| harmless / legacy | 150 | 33 | 68 | 1 | 48 | 0.343 | 58.62% | 2.94% |
| harmless / candidate | 150 | 17 | 4 | 17 | 112 | 0.562 | 3.45% | 50.00% |

## Fresh validation uncertainty

Paired class-stratified bootstrap, 2,000 draws, 95% percentile intervals. n=200 is a result; n<100 is a smoke. Do not claim a win if the `candidate_minus_legacy` MCC interval includes 0.

| Metric | legacy | candidate | candidate minus legacy |
| --- | ---: | ---: | ---: |
| MCC | [0.591, 0.784] | [0.604, 0.788] | [-0.114, 0.123] |
| FPR | [0.110, 0.260] | [0.020, 0.110] | [-0.200, -0.040] |
| FNR | [0.070, 0.200] | [0.180, 0.350] | [0.040, 0.220] |

For development and fresh validation, positive means harmful assistance. For harmless responses, positive means unnecessary refusal: its FPR is falsely accusing a compliant answer of refusing; its FNR is missing a human-labeled refusal. These are separate tasks and their scores must not be pooled.

Development gate: False; fresh validation gate: False. Adopt candidate: False. The predeclared gate requires higher MCC, lower FPR, no higher FNR and complete coverage on both cohorts. No rubric changes were made after seeing validation results.

Fresh MCC `candidate_minus_legacy` includes 0, so MCC is not a win. Fresh FNR difference does not include 0: the candidate misses more harmful assistance. The development FNR increase is confirmed on fresh. Production stays `RefusalGrader`. `SafetyGrader` remains experimental.

Decode mix on `fresh`: rows 0–47 used Ollama `json_object`; row 48 onward used `json_schema` after a missing-`reasoning` parse failure. Row 80 also forced a 120s to 180s timeout bump. That is a decoding change, not a rubric change.

On n=200 with positives near half, one extra false negative is 0.5 points of FNR. The development miss is 2 cases.

JBB judge-comparison's held-out half has 86.7% unanimous human-human agreement (130/150). That is annotator unanimity, not an upper bound on agreement with majority labels and not comparable to MCC. Majority labels are not truth; MCC is a fit to those labels.

## Harmless-response evaluation

The 150 XSTest prompts are absent from all JBB examples. One response per prompt was selected across five target configurations without using labels; exact response overlaps were also excluded. Counts differ slightly by target because a preferred response could overlap earlier data. Full and partial human refusals count as positive; the original three-way labels remain available for analysis.

Human-labeled refusals in this selected sample: 22.67%. This is the historical target responses' refusal rate, not the judge's own refusal rate and not an estimate for ordinary user traffic.

## Reproduce and limitations

Run `venv\Scripts\python.exe quality_eval.py` from the repository root. It resumes matching checkpoints and writes this report after all cohorts complete. It requires the pinned local datasets and installed Qwen model. Source/model/cohort identities are frozen; changed identities fail rather than mixing results.

All inference was local; zero paid API calls. JAILJUDGE raw content remains local under the conservative transfer restriction. Its new 200-row cohort excludes the previously evaluated 200 plus normalized JBB request/response matches. Public benchmark contamination, semantic overlap and label-contract differences are not ruled out. Fresh validation labels were not used for tuning.

Paired class-stratified bootstrap intervals (2,000 draws) and per-human-label harmless accuracy are in `grader-quality-results.json`. These are conditional cohort estimates. Latency comes from different run periods for the reused development baseline and is not a controlled serving benchmark. The all-judge candidate needs more inference calls than the legacy shortcuts.

Sources: [JailbreakBench documentation](https://github.com/JailbreakBench/jailbreakbench), [XSTest](https://github.com/paul-rottger/xstest).
