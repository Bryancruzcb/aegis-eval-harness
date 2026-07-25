"""Tests for summary math and HTML report escaping."""
from runner import build_summary
from reporter import generate_html_report


# --- Summary construction ---

def test_summary_excludes_errors_from_pass_rate():
    results = [
        {"status": "pass", "latency_seconds": 1.0},
        {"status": "fail", "latency_seconds": 1.0},
        {"status": "error", "latency_seconds": 0.0},
    ]
    summary = build_summary(results, "target-x", "judge-y", total_time_seconds=5.0)
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["errors"] == 1
    # Pass rate is over evaluated (pass+fail) cases only, not errors.
    assert summary["pass_rate"] == 50.0


def test_empty_summary_has_all_keys():
    # An over-narrow filter yields zero cases; the reporter must not KeyError.
    summary = build_summary([], "target-x", "judge-y", total_time_seconds=0.0)
    for key in [
        "timestamp", "target", "judge", "total", "passed", "failed",
        "errors", "pass_rate", "total_time_seconds", "average_latency_seconds",
    ]:
        assert key in summary
    assert summary["total"] == 0
    assert summary["pass_rate"] == 0.0


# --- HTML escaping (XSS from model output) ---

def test_html_report_escapes_model_output():
    payload = {
        "results": [{
            "id": "TC-XSS", "category": "security", "prompt": "p",
            "response": "<script>alert('xss')</script>",
            "expected_criteria": "c", "description": "d", "tags": ["x"],
            "score": 0.0, "reasoning": "r", "passed": False,
            "status": "fail", "eval_type": "deterministic", "latency_seconds": 0.1,
        }],
        "summary": {
            "timestamp": "t", "target": "x", "judge": "y", "total": 1,
            "passed": 0, "failed": 1, "errors": 0, "pass_rate": 0.0,
            "total_time_seconds": 0.1, "average_latency_seconds": 0.1,
        },
    }
    path = generate_html_report(payload, file_name="test_xss_report.html")
    with open(path, encoding="utf-8") as f:
        html = f.read()
    assert "<script>alert('xss')</script>" not in html
    assert "&lt;script&gt;" in html


# --- Transcripts, metrics, and guarded new fields ---

def _payload(**over):
    res = {"id": "T", "category": "security", "prompt": "p", "response": "<b>hi</b>",
           "expected_criteria": "c", "description": "d", "tags": ["x"], "score": 1.0,
           "reasoning": "r", "passed": True, "status": "pass", "eval_type": "llm_judge",
           "latency_seconds": 0.1,
           "transcript": [{"role": "user", "content": "ask"},
                          {"role": "assistant", "content": "<script>x</script>"}],
           "repeat_count": 3, "break_count": 1, "expect": "refuse"}
    res.update(over)
    summary = {"timestamp": "t", "target": "x", "judge": "y", "total": 1, "passed": 1,
               "failed": 0, "errors": 0, "pass_rate": 100.0, "total_time_seconds": 0.1,
               "average_latency_seconds": 0.1, "attack_total": 1, "attack_pass_rate": 100.0,
               "grader_fp_rate": None, "overall_break_rate": 0.33, "total_breaks": 1,
               "benign_total": 0, "benign_false_positives": 0}
    return {"results": [res], "summary": summary}


def test_transcript_rendered_and_escaped():
    path = generate_html_report(_payload(), file_name="test_multiturn.html")
    html = open(path, encoding="utf-8").read()
    assert "&lt;script&gt;" in html and "<script>x</script>" not in html
    assert "broke 1/3" in html


def test_none_rate_renders_dash():
    path = generate_html_report(_payload(), file_name="test_none_rate.html")
    html = open(path, encoding="utf-8").read()
    assert "—" in html  # grader_fp_rate is None


def test_existing_escaping_payload_without_new_keys_still_renders():
    # legacy-shaped payload (no transcript/repeat_count/attack_* keys) must not raise
    payload = {"results": [{"id": "L", "category": "security", "prompt": "p",
               "response": "<script>a</script>", "expected_criteria": "c", "description": "d",
               "tags": [], "score": 0.0, "reasoning": "r", "passed": False, "status": "fail",
               "eval_type": "deterministic", "latency_seconds": 0.1}],
               "summary": {"timestamp": "t", "target": "x", "judge": "y", "total": 1,
               "passed": 0, "failed": 1, "errors": 0, "pass_rate": 0.0,
               "total_time_seconds": 0.1, "average_latency_seconds": 0.1}}
    path = generate_html_report(payload, file_name="test_legacy.html")
    html = open(path, encoding="utf-8").read()
    assert "&lt;script&gt;" in html
