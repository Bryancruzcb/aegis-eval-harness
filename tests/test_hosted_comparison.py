import pytest

from hosted_comparison import reserve, usage_cost


def test_budget_reserves_before_call_and_never_exceeds_cap():
    ledger = {"attempts": []}
    reserve(ledger, 0, 4.9)
    with pytest.raises(ValueError, match="budget"):
        reserve(ledger, 1, .2)
    assert len(ledger["attempts"]) == 1


def test_thinking_tokens_are_included_in_cost():
    assert usage_cost({"prompt_token_count": 1000, "candidates_token_count": 100,
                       "thoughts_token_count": 900}) == pytest.approx(.0105)


def test_failed_row_does_not_advance_resume_position():
    from hosted_comparison import checkpoint_record, NAMES
    payload = {"records": []}
    row = {"row": 0, "actual": False, "variants": {name: {
        "error": "ServerError", "stage": "judge", "seconds": 1} for name in NAMES}}
    assert not checkpoint_record(payload, row)
    assert len(payload["records"]) == 0
    assert payload["summary"]["common_resolved"] == 0
    row = {"row": 0, "actual": False, "variants": {name: {
        "prediction": False, "stage": "judge", "seconds": 1} for name in NAMES}}
    assert checkpoint_record(payload, row)
    assert len(payload["records"]) == 1
    assert len(payload["failed_records"]) == 1
    assert payload["summary"]["common_resolved"] == 1


def test_second_process_cannot_acquire_budget_lock(tmp_path):
    import subprocess
    import sys
    from hosted_comparison import exclusive_run
    path = tmp_path / "run.lock"
    code = "from pathlib import Path; import sys; from hosted_comparison import exclusive_run\nwith exclusive_run(Path(sys.argv[1])): print('acquired')"
    with exclusive_run(path):
        result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True)
        assert result.returncode != 0
        assert "acquired" not in result.stdout
    with exclusive_run(path):
        pass


def test_daily_quota_is_not_retried_like_a_minute_limit():
    from hosted_comparison import retryable_attempt
    assert not retryable_attempt({"code": 429, "quota_ids": ["GenerateRequestsPerDayPerProjectPerModel-FreeTier"]})
    assert retryable_attempt({"code": 429, "quota_ids": ["GenerateRequestsPerMinutePerProjectPerModel-FreeTier"]})
    assert retryable_attempt({"code": 503})


def test_terminal_ungradable_response_counts_as_error_not_prediction():
    from hosted_comparison import checkpoint_record, NAMES
    payload = {"records": []}
    record = {"row": 0, "actual": True, "variants": {
        NAMES[0]: {"prediction": False, "stage": "judge", "seconds": 1},
        NAMES[1]: {"error": "JudgeParseError", "stage": "judge", "seconds": 1}}}
    assert not checkpoint_record(payload, record, terminal=True)
    assert len(payload["records"]) == 1
    assert payload["summary"]["common_resolved"] == 0
    assert payload["summary"]["variants"][NAMES[1]]["errors"] == 1
