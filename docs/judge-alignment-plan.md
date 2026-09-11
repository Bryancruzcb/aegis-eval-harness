# Judge alignment implementation plan

Revised 2026-09-06; finalized 2026-09-07 after completing evaluation.

## Objective

Correct the refusal rubric without redefining partial harmful compliance as safe. Measure the result against the published and Gemini versions on identical rows. MCC 0.60 and FPR below 20% are aspirations, not acceptance criteria or claimed outcomes.

## Implementation

1. Preserve the published and Gemini grader snapshots and original documents under ignored `output/`.
2. Keep reasoning first as an experimental output format, with a short justification. Do not claim field order guarantees better reasoning or accuracy.
3. Implement and measure a candidate that defines assistance relative to the request, including conceptual answers and finished content. Treat this as an experiment, not an automatic replacement for Gemini's rubric.
4. In the candidate, supply distinct harmful and harmless context and harmless illustrative anchors for a refusal and for partial compliance behind a disclaimer. Check harmless conceptual answers directly.
5. Preserve the existing verdict contract: any partial/full harmful assistance overrides a refusal flag. On harmless requests retain the refusal flag. Route calibration through the production grader's prompt and verdict rather than duplicating the rule.
6. Add regression tests for scenario-specific prompts, harmless conceptual answers, and calibration/production parity. Run the complete offline suite.
7. Compare frozen published, Gemini, and revised versions on the same seeded 150 development rows using an available local judge. Interleave versions, fix temperature, record model identity, prompt/source hashes, row indices, predictions, errors, stage attribution, and end-to-end judge latency. Keep benchmark text and model responses out of committed artifacts.
8. Freeze the implementation before evaluating the 150 held-out rows. Report both FPR and FNR, MCC, balanced accuracy, coverage, and latency. A held-out run estimates generalization; it does not prove zero overfitting. Add a small harmless live smoke set, explicitly separate from population-level over-refusal calibration.
9. Update the walkthrough with actual commands, results, and limitations. Sync both revised documents to the supplied Gemini paths. Preserve historical README measurements unless new evidence warrants a clearly labeled addition.
10. Adopt the measured winner and retain rejected experiments for inspection. Do not leave a poorer candidate as the default merely because its wording appears better.

## Completed outcome

All ten steps are complete. The broader candidate was implemented, tested, and measured, but rejected as the default. Gemini's existing rubric scored higher MCC on both development (0.505 versus 0.398) and held-out (0.545 versus 0.240). It also had lower held-out FPR, lower FNR, and lower mean judge latency than the candidate. All three versions passed the six harmless smoke examples; the suspected benign regression was not reproduced by that check.

Production retains the exact starting Gemini grader, with calibration now calling its production prompt and verdict. The rejected candidate remains in `experiments/refusal_grader_revised.py`. The comparison utility, calibration parity fix, 304 passing tests, evidence artifacts, and documentation are retained. No prompts were tuned on held-out data.

Final selection considered both split results. Treat this held-out half as selection/validation evidence from now on; further tuning requires fresh evaluation data for an untouched final test.

## Scope and decision rules

- Do not tune stage-1 thresholds on the held-out rows. Report its errors separately because prompt changes cannot fix decisions that bypass the judge.
- Do not change model defaults or substitute a specialized classifier without a compatible adapter and evidence. Use the installed model for this run; a modern-model sweep is deferred.
- Do not claim prompt improvement from unit tests, successful JSON parsing, a small smoke sample, or historical scores with different models.
- If live inference is unavailable or fails, record that result and provide reproducible commands. Never describe stage-1-only execution as full calibration.
- Review changes locally. Publishing and resume edits are separate work.
