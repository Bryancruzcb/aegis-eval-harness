from aegis_eval.workflows.grader_quality.analyze_screening import diagnostics


def row(actual, before, after, stage="screen"):
    return {"actual": actual, "variants": {
        "two_stage": {"prediction": before, "stage": stage},
        "judge_only": {"prediction": after, "stage": "judge"}}}


def test_diagnostics_distinguishes_corrections_from_missed_jailbreaks():
    result = diagnostics([
        row(False, True, False), row(True, True, False),
        row(True, False, True), row(False, False, True),
        row(False, True, True, "judge"), row(True, False, False, "judge"),
    ])
    assert result["screen_false_positives_fixed"] == 1
    assert result["screen_true_positives_lost"] == 1
    assert result["screen_false_negatives_fixed"] == 1
    assert result["screen_true_negatives_lost"] == 1
    assert result["shared_judge_false_positives"] == 1
    assert result["shared_judge_false_negatives"] == 1


def test_diagnostics_does_not_treat_judge_error_as_a_correction():
    r = row(False, True, False)
    r["variants"]["judge_only"] = {"stage": "judge", "error": "TimeoutError"}
    result = diagnostics([r])
    assert result["new_judge_errors"] == 1
    assert result["screen_false_positives_fixed"] == 0


def test_paired_intervals_preserve_identical_predictions():
    from aegis_eval.workflows.grader_quality.analyze_screening import paired_intervals
    records = [{"actual": actual, "variants": {
        name: {"prediction": actual} for name in ("two_stage", "judge_only")}}
        for actual in (True, False, True, False)]
    result = paired_intervals(records, draws=100)
    assert result["judge_only_minus_two_stage"]["mcc"] == [0.0, 0.0]
    assert result["judge_only_minus_two_stage"]["fpr"] == [0.0, 0.0]
