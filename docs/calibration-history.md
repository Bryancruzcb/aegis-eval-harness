# Historical calibration documentation

Superseded 2026-09-08. The text below preserves previous reported results, not current conclusions. JBB judge-comparison contains 100 benign examples; non-refusal does not mean harmful assistance. Both JBB halves have been inspected during development. Claims below of a harmful-only cohort, untouched holdout, or monotonic improvement with stronger judges are incorrect. See grader-quality-walkthrough.md for the corrected experiment.

### Calibrating the refusal grader

The refusal grader is a measuring instrument, so its own accuracy is measured
against JBB's **300 human-labelled `judge-comparison` rows**. `calibrate.py`
runs stage 1 over a held-out 150-row half — the grader's marker/length
thresholds are fixed constants set during design, not fit to this half, and
`--dev` reports the other (seeded) half for comparison — escalates the undecided
rows to the judge, and prints its confusion matrix beside JBB's own published
`StringClassifier` baseline. The positive class is **jailbroken (=
not refused)**, matching JBB's convention, and the judge is fed each row's
behaviour (`goal`), not the jailbreak wrapper — the same way the harness queries.

```bash
python calibrate.py                 # score the held-out half
python calibrate.py --dev           # score the seeded dev half instead
python calibrate.py --limit 20      # a quick 20-row check
```

Output is a confusion matrix and derived stats for the grader and the naive
baseline. Measured on the **held-out 150-row half** (one pass, temperature 0).
The table below is historical and predates the September 6 rubric revision.
See the [judge-alignment walkthrough](docs/judge-alignment-walkthrough.md) for
the paired comparison of the published, Gemini, and revised graders.
The [screening ablation walkthrough](docs/false-positive-ablation-walkthrough.md)
records the later routing experiment and separate public-label validation.
The [hosted judge comparison](docs/hosted-judge-comparison.md) reports all 150
attempted Gemini 3.5 Flash evaluations, including one provider-blocked response,
and the matched comparison supporting retention of local Qwen for this rubric.
The grader is the same two stages both times; only the escalation **judge**
changes. The reference rows are the judges JBB ships in the same file, scored on
the identical rows:

| classifier | agreement | balanced acc. | MCC | FPR / FNR |
|------------|-----------|---------------|-----|-----------|
| refusal grader — `qwen2.5:7b` judge (local, free) | 52.0% | 58.8% | 0.20 | 0.68 / 0.14 |
| refusal grader — `gemini-flash-lite` judge | 58.0% | 66.1% | **0.38** | 0.66 / 0.02 |
| naive JBB `StringClassifier` | 56.0% | 64.2% | 0.34 | 0.68 / 0.04 |
| _ref: GPT-4 judge_ | 89.3% | 89.0% | 0.77 | 0.10 / 0.13 |
| _ref: Llama-3-70B judge_ | 90.0% | 90.9% | 0.80 | 0.13 / 0.05 |
| _ref: Llama Guard 2_ | 86.0% | 85.6% | 0.71 | 0.13 / 0.16 |
| _ref: HarmBench_ | 76.7% | 78.5% | 0.55 | 0.29 / 0.14 |

> Also measured, on a smaller free-tier sample: with **`gemini-3.5-flash`** as the
> judge over a **30-row** subset, the grader scored **MCC 0.36** (balanced accuracy
> 68%) against naive's 0.19 on those same rows — consistent with `flash-lite` and
> comfortably above the baseline. A full 150-row run with a stronger hosted judge
> needs a paid-tier key (a free tier can't sustain the ~90 judge calls); the
> judge-bound conclusion holds either way.

Human–human agreement on this half is **86.7%** (130/150 unanimous). This
unanimity rate is not a ceiling on agreement with majority labels. The
**majority-class baseline** (always answer "not
jailbroken") scores **62.7%**, a property of the dataset's ~63/37 label split.

**The grader is judge-bound, and that is the finding.** Stage 1 auto-decides 41%
of rows on markers alone and escalates the other 59% to the judge, so the
grader's accuracy is dominated by the judge model behind it — and it climbs
monotonically with judge quality: a free 7B local judge lands *below* the naive
baseline (MCC 0.20), a small hosted judge (`gemini-flash-lite`) edges *past* it
(0.38), and the reference rows show strong LLMs reaching 0.55–0.80 on the same
data. With the 7B judge, 51 of the grader's 64 false positives come from the
judge itself (a capable one resolves them) and 13 from stage 1 calling a
non-standard refusal "complied" — the residual the length/marker floor cannot
reach. The two-stage framework is sound; the judge is the lever. Re-run the
one-liner above with `--judge-provider gemini` (add `--judge-delay 5` to stay
under a free-tier rate cap) to measure your own.

**Scope of this calibration** — a single number will otherwise be read as
validating everything, so:

1. calibrated on **harmful-split responses only** — the over-refusal half of the
   grader is unvalidated;
2. `target_response` rows are 2024-vintage outputs, so the length gate and marker
   lists are tuned out-of-distribution relative to 2026 targets, and the stage-1
   short-circuit rate will not transfer — re-report that rate from real runs;
3. JBB is public and frozen, so a high refusal rate partly measures memorization.
